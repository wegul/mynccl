#include "mpib_socket.h"

#include <cstring>

static ncclResult_t mpibSocketSetFlags(struct mpibSocket *sock) {
  if (sock->socketDescriptor < 0)
    return ncclInvalidArgument;
  int flags = fcntl(sock->socketDescriptor, F_GETFL);
  if (flags < 0)
    return ncclSystemError;
  if (sock->asyncFlag || sock->abortFlag) {
    if (fcntl(sock->socketDescriptor, F_SETFL, flags | O_NONBLOCK) < 0)
      return ncclSystemError;
  }
  int one = 1;
  setsockopt(sock->socketDescriptor, IPPROTO_TCP, TCP_NODELAY, (char *)&one,
             sizeof(int));
  return ncclSuccess;
}

const char *mpibSocketToString(const union mpibSocketAddress *addr, char *buf,
                               int numericHostForm) {
  const struct sockaddr *saddr = &addr->sa;
  char host[NI_MAXHOST], service[NI_MAXSERV];
  int flag = NI_NUMERICSERV | (numericHostForm ? NI_NUMERICHOST : 0);
  if (buf == NULL || addr == NULL)
    goto fail;
  if (saddr->sa_family != AF_INET && saddr->sa_family != AF_INET6)
    goto fail;
  if (getnameinfo(saddr, sizeof(union mpibSocketAddress), host, NI_MAXHOST,
                  service, NI_MAXSERV, flag))
    goto fail;
  sprintf(buf, "%s<%s>", host, service);
  return buf;
fail:
  if (buf)
    buf[0] = '\0';
  return buf;
}

uint16_t mpibSocketToPort(union mpibSocketAddress *addr) {
  return ntohs(addr->sa.sa_family == AF_INET ? addr->sin.sin_port
                                             : addr->sin6.sin6_port);
}

ncclResult_t mpibSocketInit(struct mpibSocket *sock,
                            const union mpibSocketAddress *addr, uint64_t magic,
                            enum mpibSocketType type,
                            volatile uint32_t *abortFlag, int asyncFlag,
                            int customRetry) {
  if (sock == nullptr)
    return ncclInvalidArgument;
  memset(sock, 0, sizeof(*sock));
  sock->socketDescriptor = -1;
  sock->acceptSocketDescriptor = -1;
  sock->abortFlag = abortFlag;
  sock->asyncFlag = asyncFlag;
  sock->magic = magic;
  sock->type = type;
  sock->customRetry = customRetry;
  sock->state = mpibSocketStateInitialized;

  if (addr) {
    memcpy(&sock->addr, addr, sizeof(union mpibSocketAddress));
  } else {
    memset(&sock->addr, 0, sizeof(sock->addr));
    sock->addr.sin.sin_family = AF_INET;
    sock->addr.sin.sin_addr.s_addr = INADDR_ANY;
    sock->addr.sin.sin_port = 0;
  }
  sock->salen = (sock->addr.sa.sa_family == AF_INET)
                    ? sizeof(struct sockaddr_in)
                    : sizeof(struct sockaddr_in6);

  sock->socketDescriptor = socket(sock->addr.sa.sa_family, SOCK_STREAM, 0);
  if (sock->socketDescriptor < 0)
    return ncclSystemError;
  return mpibSocketSetFlags(sock);
}

ncclResult_t mpibSocketListen(struct mpibSocket *sock) {
  if (sock == nullptr)
    return ncclInvalidArgument;
  int opt = 1;
  setsockopt(sock->socketDescriptor, SOL_SOCKET, SO_REUSEADDR, &opt,
             sizeof(opt));
#if defined(SO_REUSEPORT)
  setsockopt(sock->socketDescriptor, SOL_SOCKET, SO_REUSEPORT, &opt,
             sizeof(opt));
#endif
  SYSCHECK(bind(sock->socketDescriptor, &sock->addr.sa, sock->salen), "bind");
  socklen_t size = sock->salen;
  SYSCHECK(getsockname(sock->socketDescriptor, &sock->addr.sa, &size),
           "getsockname");
  SYSCHECK(listen(sock->socketDescriptor, 16384), "listen");
  sock->acceptSocketDescriptor = sock->socketDescriptor;
  sock->state = mpibSocketStateReady;
  return ncclSuccess;
}

ncclResult_t mpibSocketGetAddr(struct mpibSocket *sock,
                               union mpibSocketAddress *addr) {
  if (sock == nullptr || addr == nullptr)
    return ncclInvalidArgument;
  if (sock->state != mpibSocketStateReady)
    return ncclInternalError;
  memcpy(addr, &sock->addr, sizeof(*addr));
  return ncclSuccess;
}

static ncclResult_t mpibSocketConnectCheck(struct mpibSocket *sock,
                                           int errCode) {
  if (errCode == 0) {
    sock->state = mpibSocketStateConnected;
    return ncclSuccess;
  }
  if (errCode == EINPROGRESS || errCode == EWOULDBLOCK || errCode == EAGAIN) {
    sock->state = mpibSocketStateConnectPolling;
    return ncclSuccess;
  }
  WARN("mpibSocketConnect: connect failed with %s", strerror(errCode));
  sock->state = mpibSocketStateError;
  return ncclSystemError;
}

ncclResult_t mpibSocketConnect(struct mpibSocket *sock) {
  if (sock == nullptr)
    return ncclInvalidArgument;
  int ret = connect(sock->socketDescriptor, &sock->addr.sa, sock->salen);
  return mpibSocketConnectCheck(sock, (ret == -1) ? errno : 0);
}

static ncclResult_t mpibSocketFinalizeConnect(struct mpibSocket *sock) {
  int sent = sock->finalizeCounter;
  const char *ptr = reinterpret_cast<const char *>(&sock->magic);
  while (sent < (int)sizeof(sock->magic)) {
    int n = send(sock->socketDescriptor, ptr + sent, sizeof(sock->magic) - sent,
                 MSG_NOSIGNAL | MSG_DONTWAIT);
    if (n < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
        break;
      return ncclSystemError;
    }
    sent += n;
  }
  sock->finalizeCounter = sent;
  if (sent == (int)sizeof(sock->magic)) {
    sock->state = mpibSocketStateReady;
  }
  return ncclSuccess;
}

static ncclResult_t mpibSocketFinalizeAccept(struct mpibSocket *sock) {
  int received = sock->finalizeCounter;
  while (received < (int)sizeof(sock->magic)) {
    int n = recv(sock->socketDescriptor, sock->finalizeBuffer + received,
                 sizeof(sock->magic) - received, MSG_DONTWAIT);
    if (n == 0)
      return ncclRemoteError;
    if (n < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
        break;
      return ncclSystemError;
    }
    received += n;
  }
  sock->finalizeCounter = received;
  if (received == (int)sizeof(sock->magic)) {
    uint64_t magic;
    memcpy(&magic, sock->finalizeBuffer, sizeof(magic));
    if (magic != sock->magic)
      return ncclInternalError;
    sock->state = mpibSocketStateReady;
  }
  return ncclSuccess;
}

ncclResult_t mpibSocketReady(struct mpibSocket *sock, int *running) {
  if (sock == nullptr || running == nullptr)
    return ncclInvalidArgument;
  *running = 0;
  if (sock->state == mpibSocketStateAccepting) {
    if (sock->acceptSocketDescriptor < 0)
      return ncclInvalidArgument;
    socklen_t socklen = sizeof(union mpibSocketAddress);
    int fd = accept(sock->acceptSocketDescriptor,
                    (struct sockaddr *)&sock->addr, &socklen);
    if (fd < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
        // Still waiting for an incoming connection.
        return ncclSuccess;
      }
      return ncclSystemError;
    }
    sock->socketDescriptor = fd;
    sock->salen = socklen;
    sock->finalizeCounter = 0;
    sock->state = mpibSocketStateAccepted;
    NCCLCHECK(mpibSocketSetFlags(sock));
  }
  if (sock->state == mpibSocketStateConnectPolling) {
    struct pollfd pfd;
    memset(&pfd, 0, sizeof(pfd));
    pfd.fd = sock->socketDescriptor;
    pfd.events = POLLOUT;
    int ret = poll(&pfd, 1, 0);
    if (ret > 0) {
      int err = 0;
      socklen_t len = sizeof(err);
      getsockopt(sock->socketDescriptor, SOL_SOCKET, SO_ERROR, (void *)&err,
                 &len);
      NCCLCHECK(mpibSocketConnectCheck(sock, err));
    }
  }
  if (sock->state == mpibSocketStateConnected) {
    NCCLCHECK(mpibSocketFinalizeConnect(sock));
  }
  if (sock->state == mpibSocketStateAccepted) {
    NCCLCHECK(mpibSocketFinalizeAccept(sock));
  }
  if (sock->state == mpibSocketStateReady)
    *running = 1;
  return ncclSuccess;
}

ncclResult_t mpibSocketAccept(struct mpibSocket *sock,
                              struct mpibSocket *listenSock) {
  if (sock == nullptr || listenSock == nullptr)
    return ncclInvalidArgument;
  if (listenSock->acceptSocketDescriptor < 0)
    return ncclInvalidArgument;
  sock->acceptSocketDescriptor = listenSock->acceptSocketDescriptor;
  socklen_t socklen = sizeof(union mpibSocketAddress);
  sock->socketDescriptor = accept(listenSock->acceptSocketDescriptor,
                                  (struct sockaddr *)&sock->addr, &socklen);
  if (sock->socketDescriptor < 0) {
    if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
      sock->state = mpibSocketStateAccepting;
      return ncclSuccess;
    }
    return ncclSystemError;
  }
  sock->salen = socklen;
  sock->finalizeCounter = 0;
  sock->state = mpibSocketStateAccepted;
  return mpibSocketSetFlags(sock);
}

ncclResult_t mpibSocketProgress(int op, struct mpibSocket *sock, void *ptr,
                                int size, int *offset, int *closed) {
  if (closed)
    *closed = 0;
  if (sock == nullptr || ptr == nullptr || offset == nullptr)
    return ncclInvalidArgument;
  char *data = (char *)ptr;
  int bytes = 0;
  if (op == MPIB_SOCKET_RECV) {
    bytes = recv(sock->socketDescriptor, data + (*offset), size - (*offset),
                 MSG_DONTWAIT);
    if (bytes == 0) {
      if (closed) {
        *closed = 1;
        return ncclSuccess;
      }
      WARN("mpibSocketProgress: connection closed by peer");
      return ncclRemoteError;
    }
  } else {
    bytes = send(sock->socketDescriptor, data + (*offset), size - (*offset),
                 MSG_NOSIGNAL | MSG_DONTWAIT);
  }
  if (bytes < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
      return ncclSuccess;
    return ncclSystemError;
  }
  *offset += bytes;
  return ncclSuccess;
}

ncclResult_t mpibSocketShutdown(struct mpibSocket *sock, int how) {
  if (sock && sock->socketDescriptor >= 0) {
    shutdown(sock->socketDescriptor, how);
    sock->state = mpibSocketStateTerminating;
  }
  return ncclSuccess;
}

ncclResult_t mpibSocketClose(struct mpibSocket *sock) {
  if (sock && sock->socketDescriptor >= 0) {
    close(sock->socketDescriptor);
    sock->socketDescriptor = -1;
    sock->state = mpibSocketStateClosed;
  }
  return ncclSuccess;
}

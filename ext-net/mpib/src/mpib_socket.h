#pragma once

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include "mpib_compat.h"

#define MPIB_MAX_IF_NAME_SIZE 16
#define MPIB_SOCKET_NAME_MAXLEN (NI_MAXHOST+NI_MAXSERV)
#define MPIB_SOCKET_MAGIC 0x564ab9f2fc4b9d6cULL

union mpibSocketAddress {
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
};

enum mpibSocketState {
  mpibSocketStateNone = 0,
  mpibSocketStateInitialized = 1,
  mpibSocketStateAccepting = 2,
  mpibSocketStateAccepted = 3,
  mpibSocketStateConnecting = 4,
  mpibSocketStateConnectPolling = 5,
  mpibSocketStateConnected = 6,
  mpibSocketStateReady = 7,
  mpibSocketStateTerminating = 8,
  mpibSocketStateClosed = 9,
  mpibSocketStateError = 10
};

enum mpibSocketType {
  mpibSocketTypeUnknown = 0,
  mpibSocketTypeNetIb = 1
};

struct mpibSocket {
  int socketDescriptor;
  int acceptSocketDescriptor;
  int errorRetries;
  union mpibSocketAddress addr;
  volatile uint32_t* abortFlag;
  int asyncFlag;
  enum mpibSocketState state;
  int salen;
  uint64_t magic;
  enum mpibSocketType type;
  int customRetry;
  int finalizeCounter;
  char finalizeBuffer[sizeof(uint64_t)];
};

#define MPIB_SOCKET_SEND 0
#define MPIB_SOCKET_RECV 1

const char* mpibSocketToString(const union mpibSocketAddress* addr, char* buf, int numericHostForm = 1);

ncclResult_t mpibSocketInit(struct mpibSocket* sock, const union mpibSocketAddress* addr = NULL, uint64_t magic = MPIB_SOCKET_MAGIC, enum mpibSocketType type = mpibSocketTypeUnknown, volatile uint32_t* abortFlag = NULL, int asyncFlag = 0, int customRetry = 0);

ncclResult_t mpibSocketListen(struct mpibSocket* sock);

ncclResult_t mpibSocketGetAddr(struct mpibSocket* sock, union mpibSocketAddress* addr);

ncclResult_t mpibSocketConnect(struct mpibSocket* sock);

ncclResult_t mpibSocketReady(struct mpibSocket* sock, int* running);

ncclResult_t mpibSocketAccept(struct mpibSocket* sock, struct mpibSocket* listenSock);

ncclResult_t mpibSocketProgress(int op, struct mpibSocket* sock, void* ptr, int size, int* offset, int* closed = NULL);

ncclResult_t mpibSocketShutdown(struct mpibSocket* sock, int how);

ncclResult_t mpibSocketClose(struct mpibSocket* sock);

uint16_t mpibSocketToPort(union mpibSocketAddress* addr);

#include "Endpoint.h"
#include "common.h"
#include "config.h"
#include "utils.h"
#include <arpa/inet.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <mutex>
#include <netinet/in.h>
#include <pthread.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace gnic {

int Endpoint::getNumDevices() { return nr_devices_; }
int Endpoint::initDevices() {
  int nr_devices = 0;
  struct ibv_device **devices;
  char *ib_hca = getenv("NCCL_IB_HCA");
  char *if_name = getenv("NCCL_SOCKET_IFNAME");

  struct netIf user_ifs[MAX_IB_DEVS];
  bool searchNot = ib_hca && ib_hca[0] == '^';
  if (searchNot)
    ib_hca++;

  bool searchExact = ib_hca && ib_hca[0] == '=';
  if (searchExact)
    ib_hca++;

  int num_ifs = parseStringList(ib_hca, user_ifs, MAX_IB_DEVS);
  devices = ibv_get_device_list(&nr_devices);
  if (devices == nullptr || nr_devices == 0) {
    perror("ibv_get_device_list");
    goto error;
  }

  for (int d = 0; d < nr_devices && nr_global_devices < MAX_IB_DEVS; d++) {
    struct ibv_device *device = devices[d];
    const char *name = ibv_get_device_name(device);

    struct ibv_context *context = ibv_open_device(device);
    if (context == nullptr) {
      WARN("NET/IB : Unable to open device %s", name);
      continue;
    }
    struct ibv_device_attr dev_attr;
    memset(&dev_attr, 0, sizeof(dev_attr));
    if (ibv_query_device(context, &dev_attr)) {
      ibv_close_device(context);
      continue;
    }
    for (int port_num = 1; port_num <= dev_attr.phys_port_cnt; port_num++) {
      struct ibv_port_attr port_attr;
      if (ibv_query_port(context, port_num, &port_attr)) {
        WARN("NET/IB : Unable to query port %d on device %s", port_num, name);
        continue;
      }
      if (port_attr.state != IBV_PORT_ACTIVE) {
        WARN("NET/IB : Port %d on device %s is not active", port_num, name);
        continue;
      }
      // check against user specified HCAs/ports
      if (!(matchIfList(devices[d]->name, port_num, user_ifs, num_ifs,
                        searchExact) ^
            searchNot)) {
        continue;
      }
      RNICInfo *rnic = new RNICInfo();
      strncpy(rnic->ib_name, name, sizeof(rnic->ib_name));
      rnic->device = d;
      rnic->context = context;
      rnic->guid = dev_attr.sys_image_guid;
      rnic->dev_attr = dev_attr;
      rnic->portNum = port_num;
      rnic->link_bw = ncclIbSpeed(port_attr.active_speed) *
                      ncclIbWidth(port_attr.active_width);

      // Allocate a PD for this device
      rnic->pd = ibv_alloc_pd(context);
      if (rnic->pd == NULL) {
        ERROR("ibv_alloc_pd failed for device %s port %d", name, port_num);
        ibv_close_device(context);
        continue;
      }

      // Detect DMA-BUF support
      {
        struct ibv_pd *pd = ibv_alloc_pd(context);
        // Test kernel DMA-BUF support with a dummy call (fd=-1)
        (void)ibv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/,
                                0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/);
        rnic->dmaBufSupported =
            !((errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT));
        ibv_dealloc_pd(pd);
        INFO(NCCL_INIT, "DMA-BUF support: %d", rnic->dmaBufSupported);
      }

      devices_mu_.lock();
      devices_.push_back(rnic);
      devices_mu_.unlock();

      INFO(NCCL_INIT,
           "NET/init: Initialized device %s port %d with GUID 0x%lx, link "
           "speed %d Gbps.",
           name, port_num, rnic->guid, rnic->link_bw);
    }
  }

  ibv_free_device_list(devices);
  return nr_devices;

error:
  ERROR("Failed to open RDMA devices. Please check RDMA setup.");
  return -1;
}

Endpoint::Endpoint() {
  static std::once_flag flag_once;
  std::call_once(flag_once, [&]() { nr_devices_ = initDevices(); });

  if (nr_devices_ < 0) {
    ERROR("Failed to initialize RDMA devices!");
    throw std::runtime_error("Failed to initialize RDMA devices");
  }
}

Endpoint::~Endpoint() {
  std::lock_guard<std::mutex> lk(devices_mu_);
  for (auto dev : devices_) {
    if (dev->pd)
      ibv_dealloc_pd(dev->pd);
    if (dev->context)
      ibv_close_device(dev->context);
    delete dev;
  }
  devices_.clear();
}

int Endpoint::rdmaConnect(int dev, int local_gpuidx, uint32_t remote_ip_u32,
                          uint16_t listen_port) {
  sockaddr_in servAddr, localAddr;
  int ret;
  int clientFd = socket(AF_INET, SOCK_STREAM, 0);
  if (clientFd < 0) {
    ERROR("Client failed to create socket: %s", strerror(errno));
    return -1;
  }
  localAddr.sin_family = AF_INET;
  localAddr.sin_addr.s_addr = htonl(INADDR_ANY);
  localAddr.sin_port = 0;
  if (bind(clientFd, (struct sockaddr *)&localAddr, sizeof(localAddr)) < 0) {
    ERROR("Client failed to bind socket: %s", strerror(errno));
    close(clientFd);
    return -1;
  }
  servAddr.sin_family = AF_INET;
  servAddr.sin_addr.s_addr = htonl(remote_ip_u32);
  servAddr.sin_port = htons(listen_port);
  INFO(NCCL_NET, "Connecting to %s:%d (GPU index: %d), sockfd: %d",
       inet_ntoa(servAddr.sin_addr), ntohs(servAddr.sin_port), local_gpuidx,
       clientFd);

  while (connect(clientFd, (struct sockaddr *)&servAddr, sizeof(servAddr)) <
         0) {
    INFO(NCCL_NET, "Connecting... Make sure the server is running.");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  INFO(NCCL_NET, "Connected to %s:%d (GPU index: %d)",
       inet_ntoa(servAddr.sin_addr), ntohs(servAddr.sin_port), local_gpuidx);
  // fcntl(clientFd, F_SETFL, O_NONBLOCK); // Set non-blocking mode

  return clientFd;
}

int Endpoint::rdmaAccept(int sockfd, uint32_t *clientAddr) {
  sockaddr_in clientAddrIn;
  socklen_t addrLen = sizeof(clientAddrIn);
  int clientFd = accept(sockfd, (struct sockaddr *)&clientAddrIn, &addrLen);
  if (clientFd < 0) {
    ERROR("Failed to accept connection: %s", strerror(errno));
    return -1;
  }
  *clientAddr = ntohl(clientAddrIn.sin_addr.s_addr);
  INFO(NCCL_NET, "Accepted connection from %s, sockfd: %d",
       inet_ntoa(clientAddrIn.sin_addr), sockfd);
  // fcntl(clientFd, F_SETFL, O_NONBLOCK); // Set non-blocking mode
  return clientFd;
}

/**
 * @brief Fill in req; and post to sending_thread, which could be another
 * function "postSend". TODO: Note that when posting, we should consider
 * batching, instead of posting immediately.
 */
int Endpoint::asyncSend(int sockfd, void *data, size_t size,
                        struct Request *req) {
  req->op = SEND;
  req->send.laddr = (uint64_t)data;
  req->send.data_len = size;
  req->sockfd = sockfd;
  /*
   * TODO: Step1: call socket API to send the data. Note the socket should be
   * nonblocking. step2: call RDMA two-sided API... Final step: post the
   * one-sided req to a ring. The ring should belong to a particular engine.
   */
  send(sockfd, data, size, 0);

  return 0;
}

int Endpoint::asyncRecv(int sockfd, void *data, size_t size,
                        struct Request *req) {
  req->op = RECV;
  // req->recv.laddr = (uint64_t)data;
  // req->recv.data_len = size;
  req->sockfd = sockfd;
  ssize_t bytes = recv(sockfd, data, size, 0);
  INFO(NCCL_NET, "Received data of size %d from sockfd %d", bytes, sockfd);
  return 0;
}

} // namespace gnic
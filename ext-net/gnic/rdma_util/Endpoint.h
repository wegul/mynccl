#pragma once

#include "Request.h"
#include "config.h"
#include <infiniband/verbs.h>
#include <mutex>
#include <string>
#include <vector>

namespace gnic {
#define MAX_IB_DEVS 32

struct alignas(64) RNICInfo {
  char ib_name[64];
  int numa_node;
  std::string local_ip_str;
  int link_bw;

  int device;
  uint64_t guid;
  uint8_t portNum;
  ibv_context *context;
  ibv_pd *pd;
  char *pciPath;
  char *virtualPciPath;
  struct ibv_port_attr portAttr;
  struct ibv_device_attr dev_attr;
  int dmaBufSupported;
};

struct Mhandle {
  struct ibv_mr *mr;
};

// Global variable to track the number of RDMA devices
static int nr_global_devices = -1;
/**
 * @brief Endpoint manages RDMA devices and connections.
 * It provides methods to create, connect for RDMA communication.
 */
class Endpoint {

private:
  // RDMA devices.
  int nr_devices_;
  std::vector<RNICInfo *> devices_;
  std::mutex devices_mu_; // Protect devices_ vector

public:
  Endpoint(/* args */);
  ~Endpoint();
  int initDevices();
  int getNumDevices();
  int rdmaConnect(int dev, int local_gpuidx, uint32_t remote_ip_u32,
                  uint16_t listen_port);
  int rdmaAccept(int sockfd, uint32_t *clientAddr);
  int asyncSend(int sockfd, void *data, size_t size, struct Request *req);
  int asyncRecv(int sockfd, void *data, size_t size, struct Request *req);

  inline struct RNICInfo *get_device(int idx) {
    std::lock_guard<std::mutex> lock(devices_mu_);
    if (idx < 0 || idx >= nr_devices_) {
      ERROR("Invalid device index: %d", idx);
      return nullptr;
    }
    return devices_[idx];
  }
};
} // namespace gnic
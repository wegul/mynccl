#include "mpib_common.h"
#include <cstring>
#include <fstream>
#include <ifaddrs.h>
#include <limits.h>
#include <net/if.h>
#include <string>

int64_t mpibParamIbGidIndex();
int64_t mpibParamIbAsyncEvents();

static std::mutex mpibMutex;
static int netRefCount;

MPIB_PARAM(IbDisable, "IB_DISABLE", 0);

static int mpibMatchVfPath(char *path1, char *path2) {
  return strncmp(path1, path2, strlen(path1) - 1) == 0;
}

static ncclResult_t mpibGetPciPath(char *devName, char **path, int *realPort) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/infiniband/%s/device", devName);
  char *p = realpath(devicePath, NULL);
  if (p == NULL) {
    WARN("Could not find real path of %s (%s)", devName, devicePath);
  } else {
    p[strlen(p) - 1] = '0';
    *realPort = 0;
    for (int d = 0; d < mpibNIbDevs; d++) {
      if (mpibMatchVfPath(p, mpibDevs[d].pciPath))
        (*realPort)++;
    }
  }
  *path = p;
  return ncclSuccess;
}

static int ibvWidths[] = {1, 4, 8, 12, 2};
static int ibvSpeeds[] = {2500,  5000,  10000,  10000, 14000,
                          25000, 50000, 100000, 200000};

static int firstBitSet(int val, int max) {
  int i = 0;
  while (i < max && ((val & (1 << i)) == 0))
    i++;
  return i;
}
static int mpibWidth(int width) {
  return ibvWidths[firstBitSet(width, sizeof(ibvWidths) / sizeof(int) - 1)];
}
static int mpibSpeed(int speed) {
  return ibvSpeeds[firstBitSet(speed, sizeof(ibvSpeeds) / sizeof(int) - 1)];
}

const char *ibProviderName[] = {"None", "Mlx5"};

static ncclResult_t mpibReadSysfsNetdev(const char *hcaName, int gidIndex,
                                        char *netdev, size_t netdevSize) {
  char path[PATH_MAX];
  snprintf(path, sizeof(path),
           "/sys/class/infiniband/%s/ports/1/gid_attrs/ndevs/%d", hcaName,
           gidIndex);
  std::ifstream file(path);
  if (!file.is_open()) {
    WARN("NET/MPIB : Unable to read sysfs netdev %s", path);
    return ncclInvalidUsage;
  }
  std::string name;
  file >> name;
  file.close();
  if (name.empty()) {
    WARN("NET/MPIB : Empty netdev name in %s", path);
    return ncclInvalidUsage;
  }
  strncpy(netdev, name.c_str(), netdevSize);
  netdev[netdevSize - 1] = '\0';
  return ncclSuccess;
}

static ncclResult_t mpibResolveIfAddr(const char *ifname,
                                      union mpibSocketAddress *out) {
  struct ifaddrs *ifaddr = NULL;
  if (getifaddrs(&ifaddr) == -1)
    return ncclSystemError;
  ncclResult_t ret = ncclInvalidUsage;
  for (struct ifaddrs *ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == NULL)
      continue;
    if (strcmp(ifa->ifa_name, ifname) != 0)
      continue;
    if (ifa->ifa_addr->sa_family == AF_INET) {
      memset(out, 0, sizeof(*out));
      out->sin = *(struct sockaddr_in *)ifa->ifa_addr;
      out->sin.sin_port = 0;
      ret = ncclSuccess;
      break;
    }
  }
  freeifaddrs(ifaddr);
  if (ret != ncclSuccess) {
    WARN("NET/MPIB : No IPv4 address found for netdev %s", ifname);
  }
  return ret;
}

static ncclResult_t mpibInitDevices(ncclDebugLogger_t logFunction,
                                    ncclProfilerCallback_t profFunction) {
  ncclResult_t ret = ncclSuccess;
  if (netRefCount++)
    return ret;
  mpibProfilerFunction = profFunction;
  mpibLogFunction = logFunction;
  if (mpibParamIbDisable())
    return ncclInternalError;

  if (wrap_ibv_symbols() != ncclSuccess)
    return ncclInternalError;

  if (mpibNIbDevs == -1) {
    std::lock_guard<std::mutex> lock(mpibMutex);
    wrap_ibv_fork_init();
    if (mpibNIbDevs == -1) {
      mpibNIbDevs = 0;
      mpibNMergedIbDevs = 0;

      int nIbDevs = 0;
      int devIndex = -1;
      int gidIndex = 0;
      char netdev[MPIB_MAX_IF_NAME_SIZE + 1];
      struct ibv_device **devices = NULL;
      struct ibv_context *context = NULL;
      struct ibv_device_attr devAttr;
      struct ibv_port_attr portAttr;
      mpibMergedDev *mDev = NULL;

      const char *userIbEnv = mpibGetEnv("NCCL_IB_HCA");
      if (userIbEnv == NULL || userIbEnv[0] == '\0') {
        WARN("NET/MPIB : NCCL_IB_HCA is required");
        ret = ncclInvalidUsage;
        goto fail;
      }

      if (ncclSuccess != wrap_ibv_get_device_list(&devices, &nIbDevs)) {
        ret = ncclInternalError;
        goto fail;
      }
      for (int d = 0; d < nIbDevs; d++) {
        if (strcmp(devices[d]->name, userIbEnv) == 0) {
          devIndex = d;
          break;
        }
      }
      if (devIndex < 0) {
        WARN("NET/MPIB : Unable to find requested device %s", userIbEnv);
        ret = ncclInvalidUsage;
        goto fail_devices;
      }

      if (ncclSuccess != wrap_ibv_open_device(&context, devices[devIndex]) ||
          context == NULL) {
        WARN("NET/MPIB : Unable to open device %s", devices[devIndex]->name);
        ret = ncclInternalError;
        goto fail_devices;
      }

      memset(&devAttr, 0, sizeof(devAttr));
      if (ncclSuccess != wrap_ibv_query_device(context, &devAttr)) {
        WARN("NET/MPIB : Unable to query device %s", devices[devIndex]->name);
        ret = ncclInternalError;
        goto fail_context;
      }

      if (ncclSuccess != wrap_ibv_query_port(context, 1, &portAttr)) {
        WARN("NET/MPIB : Unable to query port_num 1");
        ret = ncclInternalError;
        goto fail_context;
      }
      if (portAttr.state != IBV_PORT_ACTIVE) {
        WARN("NET/MPIB : Port 1 is not active for device %s",
             devices[devIndex]->name);
        ret = ncclInvalidUsage;
        goto fail_context;
      }
      if (portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) {
        WARN("NET/MPIB : Port 1 is not Ethernet (RoCE) for device %s",
             devices[devIndex]->name);
        ret = ncclInvalidUsage;
        goto fail_context;
      }

      gidIndex = (int)mpibParamIbGidIndex();
      NCCLCHECKGOTO(
          mpibReadSysfsNetdev(userIbEnv, gidIndex, netdev, sizeof(netdev)), ret,
          fail_context);
      strncpy(mpibIfName, netdev, sizeof(mpibIfName));
      mpibIfName[sizeof(mpibIfName) - 1] = '\0';
      NCCLCHECKGOTO(mpibResolveIfAddr(mpibIfName, &mpibIfAddr), ret,
                    fail_context);

      mpibNIbDevs = 0;
      mpibNMergedIbDevs = 0;
      mpibDevs[0].device = devIndex;
      mpibDevs[0].ibProvider = IB_PROVIDER_MLX5;
      mpibDevs[0].guid = devAttr.sys_image_guid;
      mpibDevs[0].portAttr = portAttr;
      mpibDevs[0].portNum = 1;
      mpibDevs[0].link = portAttr.link_layer;
      if (portAttr.active_speed_ex) {
        mpibDevs[0].speed = mpibSpeed(portAttr.active_speed_ex) *
                            mpibWidth(portAttr.active_width);
      } else {
        mpibDevs[0].speed =
            mpibSpeed(portAttr.active_speed) * mpibWidth(portAttr.active_width);
      }
      mpibDevs[0].context = context;
      mpibDevs[0].pdRefs = 0;
      mpibDevs[0].pd = NULL;
      strncpy(mpibDevs[0].devName, devices[devIndex]->name, MAXNAMESIZE);
      NCCLCHECKGOTO(mpibGetPciPath(mpibDevs[0].devName, &mpibDevs[0].pciPath,
                                   &mpibDevs[0].realPort),
                    ret, fail_context);
      mpibDevs[0].maxQp = devAttr.max_qp;
      mpibDevs[0].mrCache.capacity = 0;
      mpibDevs[0].mrCache.population = 0;
      mpibDevs[0].mrCache.slots = NULL;
      NCCLCHECK(mpibStatsInit(&mpibDevs[0].stats));

      if (mpibParamIbAsyncEvents()) {
        mpibAsyncThread = std::thread(mpibAsyncThreadMain, &mpibDevs[0]);
        mpibSetThreadName(mpibAsyncThread, "MPIB IbAsync %2d", mpibNIbDevs);
        mpibAsyncThread.detach();
      }

      mpibDevs[0].ar = 0;

      mpibNIbDevs = 1;
      mpibNMergedIbDevs = 1;
      mDev = mpibMergedDevs + 0;
      memset(mDev, 0, sizeof(*mDev));
      mDev->vProps.ndevs = 1;
      mDev->vProps.devs[0] = 0;
      strncpy(mDev->devName, mpibDevs[0].devName, MAXNAMESIZE);
      mDev->speed = mpibDevs[0].speed;

      INFO(NCCL_NET,
           "NET/MPIB: [%d] %s:%s:%d/%s provider=%s speed=%d context=%p "
           "pciPath=%s ar=%d",
           devIndex, devices[devIndex]->name, devices[devIndex]->dev_name,
           mpibDevs[0].portNum, MPIB_IB_LLSTR(portAttr.link_layer),
           ibProviderName[mpibDevs[0].ibProvider], mpibDevs[0].speed, context,
           mpibDevs[0].pciPath, mpibDevs[0].ar);

    fail_devices:
      if (nIbDevs && (ncclSuccess != wrap_ibv_free_device_list(devices))) {
        ret = ncclInternalError;
        goto fail;
      }
      if (ret != ncclSuccess)
        goto fail;
      goto done_init;

    fail_context:
      if (context && ncclSuccess != wrap_ibv_close_device(context)) {
        ret = ncclInternalError;
      }
      goto fail_devices;
    }
  done_init:
    if (mpibNIbDevs == 0) {
      INFO(NCCL_INIT | NCCL_NET, "NET/MPIB : No device found.");
    }

    char line[2048];
    line[0] = '\0';
    mpibRelaxedOrderingEnabled = 0;
    for (int d = 0; d < mpibNIbDevs; d++) {
      snprintf(line + strlen(line), sizeof(line) - strlen(line),
               " [%d]%s:%d/%s", d, mpibDevs[d].devName, mpibDevs[d].portNum,
               MPIB_IB_LLSTR(mpibDevs[d].link));
    }
    char addrline[MPIB_SOCKET_NAME_MAXLEN + 1];
    INFO(NCCL_INIT | NCCL_NET, "NET/MPIB : Using%s; OOB %s:%s", line,
         mpibIfName, mpibSocketToString(&mpibIfAddr, addrline));
  }

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t mpibFinalizeDevices(void) {
  netRefCount--;
  return ncclSuccess;
}

__hidden ncclResult_t mpibInit(void **ctx, uint64_t commId,
                               ncclNetCommConfig_t *config,
                               ncclDebugLogger_t logFunction,
                               ncclProfilerCallback_t profFunction) {
  (void)commId;
  ncclResult_t ret = ncclSuccess;
  ncclNetCommConfig_t *netCommConfig = nullptr;
  NCCLCHECK(mpibInitDevices(logFunction, profFunction));
  NCCLCHECK(mpibCalloc(&netCommConfig, 1));
  netCommConfig->trafficClass =
      config ? config->trafficClass : NCCL_NET_TRAFFIC_CLASS_UNDEF;
  *ctx = (void *)netCommConfig;
  return ret;
}

__hidden ncclResult_t mpibDevices(int *ndev) {
  *ndev = mpibNIbDevs;
  return ncclSuccess;
}

static ncclResult_t mpibGetPhysProperties(int dev, ncclNetProperties_t *props) {
  struct mpibDev *ibDev = mpibDevs + dev;
  std::lock_guard<std::mutex> lock(ibDev->mutex);
  props->name = (char *)"mpib";
  props->speed = ibDev->speed;
  props->pciPath = ibDev->pciPath;
  props->guid = ibDev->guid;
  props->ptrSupport = NCCL_PTR_HOST;
  props->regIsGlobal = 1;
  props->forceFlush = 0;
  props->latency = 0;
  props->port = ibDev->portNum + ibDev->realPort;
  props->maxComms = 1024;
  props->maxRecvs = MPIB_NET_IB_MAX_RECVS;
  props->netDeviceType = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxCollBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxMultiRequestSize = 1;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  return ncclSuccess;
}

__hidden ncclResult_t mpibGetProperties(int dev, ncclNetProperties_t *props) {
  if (dev >= mpibNIbDevs) {
    WARN("NET/MPIB : Requested properties for dev %d, only %d devs have been "
         "created",
         dev, mpibNIbDevs);
    return ncclInvalidUsage;
  }
  NCCLCHECK(mpibGetPhysProperties(dev, props));
  return ncclSuccess;
}

__hidden ncclResult_t mpibMakeVDevice(int *d, ncclNetVDeviceProps_t *props) {
  (void)d;
  (void)props;
  return ncclInvalidUsage;
}

__hidden ncclResult_t mpibFinalize(void *ctx) {
  free(ctx);
  return mpibFinalizeDevices();
}

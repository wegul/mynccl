#include "mpib_agent_client.h"
#include "mpib_common.h"
#include <cstring>
#include <ifaddrs.h>
#include <limits.h>
#include <net/if.h>

int64_t mpibParamIbGidIndex();

static std::mutex mpibMutex;
static int netRefCount;

MPIB_PARAM(IbDisable, "IB_DISABLE", 0);

static ncclResult_t mpibGetPciPath(char *devName, char **path, int *realPort) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/infiniband/%s/device", devName);
  char *p = realpath(devicePath, NULL);
  if (p == NULL) {
    WARN("Could not find real path of %s (%s)", devName, devicePath);
    return ncclSystemError;
  }
  p[strlen(p) - 1] = '0';
  *realPort = 0;
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
  return ret;
}

// Helper to initialize a single mpibDev entry
static ncclResult_t mpibInitSingleDev(struct ibv_device **devices, int nIbDevs,
                                      const char *hcaName, int devSlot) {
  int devIndex = -1;
  for (int d = 0; d < nIbDevs; d++) {
    if (strcmp(devices[d]->name, hcaName) == 0) {
      devIndex = d;
      break;
    }
  }
  if (devIndex < 0) {
    WARN("NET/MPIB : Unable to find device %s", hcaName);
    return ncclInvalidUsage;
  }

  struct ibv_context *context = NULL;
  NCCLCHECK(wrap_ibv_open_device(&context, devices[devIndex]));

  struct ibv_device_attr devAttr;
  memset(&devAttr, 0, sizeof(devAttr));
  NCCLCHECK(wrap_ibv_query_device(context, &devAttr));

  struct ibv_port_attr portAttr;
  NCCLCHECK(wrap_ibv_query_port(context, 1, &portAttr));
  if (portAttr.state != IBV_PORT_ACTIVE) {
    WARN("NET/MPIB : Port 1 is not active for device %s", hcaName);
    wrap_ibv_close_device(context);
    return ncclInvalidUsage;
  }
  if (portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) {
    WARN("NET/MPIB : Port 1 is not Ethernet (RoCE) for device %s", hcaName);
    wrap_ibv_close_device(context);
    return ncclInvalidUsage;
  }

  mpibDev *dev = &mpibDevs[devSlot];
  dev->device = devIndex;
  dev->ibProvider = IB_PROVIDER_MLX5;
  dev->guid = devAttr.sys_image_guid;
  dev->portAttr = portAttr;
  dev->portNum = 1;
  dev->link = portAttr.link_layer;
  if (portAttr.active_speed_ex) {
    dev->speed =
        mpibSpeed(portAttr.active_speed_ex) * mpibWidth(portAttr.active_width);
  } else {
    dev->speed =
        mpibSpeed(portAttr.active_speed) * mpibWidth(portAttr.active_width);
  }
  dev->context = context;
  dev->pdRefs = 0;
  dev->pd = NULL;
  strncpy(dev->devName, devices[devIndex]->name, MAXNAMESIZE);
  NCCLCHECK(mpibGetPciPath(dev->devName, &dev->pciPath, &dev->realPort));
  dev->maxQp = devAttr.max_qp;
  dev->mrCache.capacity = 0;
  dev->mrCache.population = 0;
  dev->mrCache.slots = NULL;
  NCCLCHECK(mpibStatsInit(&dev->stats));
  dev->ar = 0;

  INFO(NCCL_NET, "NET/MPIB: [%d] %s:%d/%s speed=%d context=%p pciPath=%s",
       devSlot, dev->devName, dev->portNum, MPIB_IB_LLSTR(dev->link),
       dev->speed, context, dev->pciPath);

  return ncclSuccess;
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

      // Read HCA names from environment
      const char *hcaSout = mpibGetEnv("MPIB_HCA_SOUT");
      const char *hcaSup = mpibGetEnv("MPIB_HCA_SUP");
      if (!hcaSout || hcaSout[0] == '\0') {
        WARN("NET/MPIB : MPIB_HCA_SOUT is required");
        return ncclInvalidUsage;
      }
      if (!hcaSup || hcaSup[0] == '\0') {
        WARN("NET/MPIB : MPIB_HCA_SUP is required");
        return ncclInvalidUsage;
      }

      int nIbDevs = 0;
      struct ibv_device **devices = NULL;
      NCCLCHECKGOTO(wrap_ibv_get_device_list(&devices, &nIbDevs), ret, fail);

      // Initialize SOUT NIC (index 0)
      NCCLCHECKGOTO(mpibInitSingleDev(devices, nIbDevs, hcaSout, 0), ret,
                    fail_devices);
      // Initialize SUP NIC (index 1)
      NCCLCHECKGOTO(mpibInitSingleDev(devices, nIbDevs, hcaSup, 1), ret,
                    fail_devices);
      mpibNIbDevs = 2;

      wrap_ibv_free_device_list(devices);

      // Resolve OOB interface (independent of RDMA NICs)
      const char *oobIf = mpibGetEnv("MPIB_OOB_IF");
      if (!oobIf || oobIf[0] == '\0') {
        WARN("NET/MPIB : MPIB_OOB_IF is required");
        ret = ncclInvalidUsage;
        goto fail_devices;
      }
      strncpy(mpibIfName, oobIf, sizeof(mpibIfName));
      mpibIfName[sizeof(mpibIfName) - 1] = '\0';
      NCCLCHECKGOTO(mpibResolveIfAddr(mpibIfName, &mpibIfAddr), ret,
                    fail_devices);

      // Initialize the single merged device (dev=0) spanning both NICs
      mpibMergedDevs[0].vProps.ndevs = 2;
      mpibMergedDevs[0].vProps.devs[0] = 0; // SOUT
      mpibMergedDevs[0].vProps.devs[1] = 1; // SUP
      mpibMergedDevs[0].speed = mpibDevs[0].speed + mpibDevs[1].speed;
      mpibNMergedIbDevs = 1;

      char addrline[MPIB_SOCKET_NAME_MAXLEN + 1];
      INFO(NCCL_INIT | NCCL_NET,
           "NET/MPIB : Using [0]%s:SOUT [1]%s:SUP speed=%d; OOB %s:%s",
           mpibDevs[0].devName, mpibDevs[1].devName,
           mpibDevs[0].speed + mpibDevs[1].speed, mpibIfName,
           mpibSocketToString(&mpibIfAddr, addrline));
    }
  }
  return ncclSuccess;

fail_devices:
  // Cleanup any opened devices
  for (int d = 0; d < mpibNIbDevs; d++) {
    if (mpibDevs[d].context)
      wrap_ibv_close_device(mpibDevs[d].context);
  }
fail:
  mpibNIbDevs = 0;
  return ret;
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

  /* Initialize agent client (SHM mapping) */
  NCCLCHECK(mpibAgentClientInit());
  NCCLCHECK(mpibCalloc(&netCommConfig, 1));
  netCommConfig->trafficClass =
      config ? config->trafficClass : NCCL_NET_TRAFFIC_CLASS_UNDEF;
  *ctx = (void *)netCommConfig;
  return ret;
}

// MPIB exposes a single logical device (dev=0) that spans both physical NICs
__hidden ncclResult_t mpibDevices(int *ndev) {
  *ndev = 1; // Single fused device
  return ncclSuccess;
}

__hidden ncclResult_t mpibGetProperties(int dev, ncclNetProperties_t *props) {
  if (dev != 0 || mpibNIbDevs != 2) {
    WARN("NET/MPIB : Invalid dev %d or mpibNIbDevs=%d", dev, mpibNIbDevs);
    return ncclInvalidUsage;
  }
  // Use SOUT (dev 0) for base properties
  struct mpibDev *ibDev = &mpibDevs[0];
  std::lock_guard<std::mutex> lock(ibDev->mutex);
  props->name = (char *)"mpib";
  props->speed = mpibDevs[0].speed + mpibDevs[1].speed; // Aggregated
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
  // Expose both physical devices in vProps
  props->vProps.ndevs = 2;
  props->vProps.devs[0] = 0; // SOUT
  props->vProps.devs[1] = 1; // SUP
  return ncclSuccess;
}

__hidden ncclResult_t mpibMakeVDevice(int *d, ncclNetVDeviceProps_t *props) {
  (void)d;
  (void)props;
  return ncclInvalidUsage;
}

__hidden ncclResult_t mpibFinalize(void *ctx) {
  free(ctx);
  mpibAgentClientDestroy();
  return mpibFinalizeDevices();
}

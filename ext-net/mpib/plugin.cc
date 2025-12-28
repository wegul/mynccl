/*************************************************************************
 * Copyright (c) 2015-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "net.h"

#define __hidden __attribute__((visibility("hidden")))
#define NCCL_PLUGIN_MAX_RECVS 1

// Maximum number of in-flight requests this plugin supports.
int max_requests = NCCL_NET_MAX_REQUESTS;

// Basic v11-only skeleton for mpib (multipath-ib) net plugin.
// All functions currently return success/defaults or ncclInternalError as
// placeholders for you to implement.

__hidden ncclResult_t pluginInit(void **ctx, uint64_t commId,
                                 ncclNetCommConfig_t *config,
                                 ncclDebugLogger_t logFunction,
                                 ncclProfilerCallback_t profFunction) {
  // TODO: allocate and initialize a plugin context object if needed and
  //       store it in *ctx. You may also want to save logFunction/profFunction.
  (void)commId;
  (void)config;
  (void)logFunction;
  (void)profFunction;
  *ctx = nullptr;
  return ncclSuccess;
}

__hidden ncclResult_t pluginDevices(int *ndev) {
  // TODO: detect and return the number of mpib devices.
  *ndev = 0;
  return ncclSuccess;
}

__hidden ncclResult_t pluginGetProperties(int dev, ncclNetProperties_t *props) {
  // Below are default values, if unsure don't change.

  (void)dev;
  props->name = 0;
  // Fill for proper topology detection, e.g.
  // /sys/devices/pci0000:00/0000:00:10.0/0000:0b:00.0
  props->pciPath = NULL;
  // Only used to detect NICs with multiple PCI attachments.
  props->guid = 0;
  // Add NCCL_PTR_CUDA if GPU Direct RDMA is supported and regMr can take CUDA
  // pointers.
  props->ptrSupport = NCCL_PTR_HOST;
  // If you regMr has a fast registration cache, set to 1. If set to 0, user
  // buffer registration may be disabled.
  props->regIsGlobal = 0;
  // Force flush after receive. Needed if the control path and data path use a
  // different path to the GPU
  props->forceFlush = 0;
  // Speed in *Mbps*. 100000 means 100G
  props->speed = 100000;
  // Port number, used in conjunction with guid
  props->port = 0;
  // Custom latency (used to help tuning if latency is high. If set to 0, use
  // default NCCL values.
  props->latency = 0;
  // Maximum number of comm objects we can create.
  props->maxComms = 1024 * 1024;
  // Maximum number of receive operations taken by irecv().
  props->maxRecvs = NCCL_PLUGIN_MAX_RECVS;
  // Coupling with NCCL network device-side code.
  props->netDeviceType = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  // Used to tell NCCL core whether this is a virtual device fusing multiple
  // physical devices.
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  // maximum transfer sizes the plugin can handle
  props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxCollBytes = NCCL_MAX_NET_SIZE_BYTES;
  // v11: maximum number of requests that can be grouped in a
  // multi-request operation.
  props->maxMultiRequestSize = max_requests;
  return ncclSuccess;
}
__hidden ncclResult_t pluginListen(void *ctx, int dev, void *handle,
                                   void **listenComm) {
  (void)ctx;
  (void)dev;
  (void)handle;
  (void)listenComm;
  return ncclInternalError;
}

__hidden ncclResult_t pluginConnect(void *ctx, int dev, void *handle,
                                    void **sendComm,
                                    ncclNetDeviceHandle_t **sendDevComm) {
  (void)ctx;
  (void)dev;
  (void)handle;
  (void)sendComm;
  (void)sendDevComm;
  return ncclInternalError;
}

__hidden ncclResult_t pluginAccept(void *listenComm, void **recvComm,
                                   ncclNetDeviceHandle_t **recvDevComm) {
  (void)listenComm;
  (void)recvComm;
  (void)recvDevComm;
  return ncclInternalError;
}

__hidden ncclResult_t pluginRegMr(void *collComm, void *data, size_t size,
                                  int type, void **mhandle) {
  (void)collComm;
  (void)data;
  (void)size;
  (void)type;
  (void)mhandle;
  return ncclInternalError;
}

__hidden ncclResult_t pluginRegMrDmaBuf(void *collComm, void *data, size_t size,
                                        int type, uint64_t offset, int fd,
                                        void **mhandle) {
  (void)collComm;
  (void)data;
  (void)size;
  (void)type;
  (void)offset;
  (void)fd;
  (void)mhandle;
  return ncclInternalError;
}

__hidden ncclResult_t pluginDeregMr(void *collComm, void *mhandle) {
  (void)collComm;
  (void)mhandle;
  return ncclInternalError;
}

__hidden ncclResult_t pluginIsend(void *sendComm, void *data, size_t size,
                                  int tag, void *mhandle, void *phandle,
                                  void **request) {
  (void)sendComm;
  (void)data;
  (void)size;
  (void)tag;
  (void)mhandle;
  (void)phandle;
  (void)request;
  return ncclInternalError;
}

__hidden ncclResult_t pluginIrecv(void *recvComm, int n, void **data,
                                  size_t *sizes, int *tags, void **mhandles,
                                  void **phandles, void **request) {
  (void)recvComm;
  (void)n;
  (void)data;
  (void)sizes;
  (void)tags;
  (void)mhandles;
  (void)phandles;
  (void)request;
  return ncclInternalError;
}

__hidden ncclResult_t pluginIflush(void *recvComm, int n, void **data,
                                   int *sizes, void **mhandles,
                                   void **request) {
  (void)recvComm;
  (void)n;
  (void)data;
  (void)sizes;
  (void)mhandles;
  (void)request;
  return ncclInternalError;
}

__hidden ncclResult_t pluginTest(void *request, int *done, int *size) {
  (void)request;
  (void)done;
  (void)size;
  return ncclInternalError;
}

__hidden ncclResult_t pluginCloseSend(void *sendComm) {
  (void)sendComm;
  return ncclInternalError;
}

__hidden ncclResult_t pluginCloseRecv(void *recvComm) {
  (void)recvComm;
  return ncclInternalError;
}

__hidden ncclResult_t pluginCloseListen(void *listenComm) {
  (void)listenComm;
  return ncclInternalError;
}

__hidden ncclResult_t pluginIrecvConsumed(void *recvComm, int n,
                                          void *request) {
  (void)recvComm;
  (void)n;
  (void)request;
  return ncclInternalError;
}

__hidden ncclResult_t pluginGetDeviceMr(void *comm, void *mhandle,
                                        void **dptr_mhandle) {
  (void)comm;
  (void)mhandle;
  (void)dptr_mhandle;
  return ncclInternalError;
}

__hidden ncclResult_t pluginMakeVDevice(int *d, ncclNetVDeviceProps_t *props) {
  (void)d;
  (void)props;
  return ncclInternalError;
}

__hidden ncclResult_t pluginFinalize(void *ctx) {
  (void)ctx;
  return ncclSuccess;
}

#define PLUGIN_NAME "Plugin"

const ncclNet_v11_t ncclNetPlugin_v11 = {
    .name = PLUGIN_NAME,
    .init = pluginInit,
    .devices = pluginDevices,
    .getProperties = pluginGetProperties,
    .listen = pluginListen,
    .connect = pluginConnect,
    .accept = pluginAccept,
    .regMr = pluginRegMr,
    .regMrDmaBuf = pluginRegMrDmaBuf,
    .deregMr = pluginDeregMr,
    .isend = pluginIsend,
    .irecv = pluginIrecv,
    .iflush = pluginIflush,
    .test = pluginTest,
    .closeSend = pluginCloseSend,
    .closeRecv = pluginCloseRecv,
    .closeListen = pluginCloseListen,
    .getDeviceMr = pluginGetDeviceMr,
    .irecvConsumed = pluginIrecvConsumed,
    .makeVDevice = pluginMakeVDevice,
    .finalize = pluginFinalize,
};

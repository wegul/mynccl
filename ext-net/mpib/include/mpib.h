#pragma once

#include "net.h"

ncclResult_t mpibInit(void **ctx, uint64_t commId, ncclNetCommConfig_t *config,
                      ncclDebugLogger_t logFunction,
                      ncclProfilerCallback_t profFunction);

ncclResult_t mpibDevices(int *ndev);

ncclResult_t mpibGetProperties(int dev, ncclNetProperties_t *props);

ncclResult_t mpibListen(void *ctx, int dev, void *handle, void **listenComm);

ncclResult_t mpibConnect(void *ctx, int dev, void *handle, void **sendComm,
                         ncclNetDeviceHandle_t **sendDevComm);

ncclResult_t mpibAccept(void *listenComm, void **recvComm,
                        ncclNetDeviceHandle_t **recvDevComm);

ncclResult_t mpibRegMr(void *collComm, void *data, size_t size, int type,
                       void **mhandle);

ncclResult_t mpibRegMrDmaBuf(void *collComm, void *data, size_t size, int type,
                             uint64_t offset, int fd, void **mhandle);

ncclResult_t mpibDeregMr(void *collComm, void *mhandle);

ncclResult_t mpibIsend(void *sendComm, void *data, size_t size, int tag,
                       void *mhandle, void *phandle, void **request);

ncclResult_t mpibIrecv(void *recvComm, int n, void **data, size_t *sizes,
                       int *tags, void **mhandles, void **phandles,
                       void **request);

ncclResult_t mpibIflush(void *recvComm, int n, void **data, int *sizes,
                        void **mhandles, void **request);

ncclResult_t mpibTest(void *request, int *done, int *size);

ncclResult_t mpibCloseSend(void *sendComm);

ncclResult_t mpibCloseRecv(void *recvComm);

ncclResult_t mpibCloseListen(void *listenComm);

ncclResult_t mpibGetDeviceMr(void *comm, void *mhandle, void **dptr_mhandle);

ncclResult_t mpibIrecvConsumed(void *recvComm, int n, void *request);

ncclResult_t mpibMakeVDevice(int *d, ncclNetVDeviceProps_t *props);

ncclResult_t mpibFinalize(void *ctx);

/*************************************************************************
 * mpib net plugin wrapper (C++ TU, C-exported entrypoint)
 ************************************************************************/

#include "mpib.h"
#include "mpib_compat.h"

#define PLUGIN_NAME "mpib"

// NCCL discovers the plugin by dlsym("ncclNetPlugin_v11").
// This symbol must be globally visible and have C linkage (unmangled).
extern "C" __public const ncclNet_v11_t ncclNetPlugin_v11 = {
    .name = PLUGIN_NAME,
    .init = mpibInit,
    .devices = mpibDevices,
    .getProperties = mpibGetProperties,
    .listen = mpibListen,
    .connect = mpibConnect,
    .accept = mpibAccept,
    .regMr = mpibRegMr,
    .regMrDmaBuf = mpibRegMrDmaBuf,
    .deregMr = mpibDeregMr,
    .isend = mpibIsend,
    .irecv = mpibIrecv,
    .iflush = mpibIflush,
    .test = mpibTest,
    .closeSend = mpibCloseSend,
    .closeRecv = mpibCloseRecv,
    .closeListen = mpibCloseListen,
    .getDeviceMr = NULL,
    .irecvConsumed = NULL,
    .makeVDevice = mpibMakeVDevice,
    .finalize = mpibFinalize,
};

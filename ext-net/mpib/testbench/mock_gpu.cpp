// mpib testbench: mock GPU endpoint
// This file is a skeleton modeled after ext-net/gnic/testbench/mock_gpu.cpp.
// You will fill in the actual implementation.

#include "net.h" // from ext-net/mpib/nccl or the appropriate include path

// TODO: include any logging / debug helpers you want
// #include "debug_util.h"

// TODO: choose the correct plugin symbol once your mpib plugin is implemented.
// For example:
//   extern const ncclNet_v11_t ncclNetPlugin_v11;
//   static const ncclNet_v11_t* plugin = &ncclNetPlugin_v11;

int main(int argc, char **argv) {
  // TODO: parse optional command-line arguments (e.g., server IP/port, device
  // index).

  // TODO: initialize plugin (v11):
  //   void* ctx = nullptr;
  //   uint64_t commId = 0; // or derive from args/env
  //   ncclNetCommConfig_t config = {};
  //   ncclResult_t res = plugin->init(&ctx, commId, &config, /*log*/nullptr,
  //   /*prof*/nullptr);

  // TODO: query devices and properties via plugin->devices /
  // plugin->getProperties.

  // TODO: call plugin->listen(ctx, dev, handle, &listenComm) to obtain a listen
  // handle.

  // TODO: exchange handles with the control server (see ctrl_server.cpp
  // skeleton)
  //       and obtain the peer's handle.

  // TODO: spawn threads (or use non-blocking loops) that call
  //       plugin->accept and plugin->connect(ctx, dev, handle, &sendComm,...)
  //       until both sides establish comms.

  // TODO: allocate test buffers, issue plugin->isend / plugin->irecv, and
  //       drive completion via plugin->test on the returned request objects.

  // TODO: print basic success/failure information and clean up all comms
  //       using plugin->closeSend / closeRecv / closeListen and
  //       plugin->finalize(ctx).

  return 0;
}

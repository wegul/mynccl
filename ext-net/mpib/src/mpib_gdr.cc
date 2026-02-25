#include "mpib_common.h"

// Detect whether GDR can work on a given NIC with the current CUDA device
// Returns :
// ncclSuccess : GDR works (nv_peermem module loaded)
// ncclSystemError : no module loaded
#define KNL_MODULE_LOADED(a) ((access(a, F_OK) == -1) ? 0 : 1)
static int mpibGdrModuleLoaded = 0;
static void mpibGdrSupportInitOnce() {
  mpibGdrModuleLoaded =
      KNL_MODULE_LOADED("/sys/kernel/mm/memory_peers/nv_mem/version") ||
      KNL_MODULE_LOADED("/sys/kernel/mm/memory_peers/nv_mem_nc/version") ||
      KNL_MODULE_LOADED("/sys/module/nvidia_peermem/version");
}
ncclResult_t mpibGdrSupport() {
  static std::once_flag once;
  std::call_once(once, mpibGdrSupportInitOnce);
  if (!mpibGdrModuleLoaded)
    return ncclSystemError;
  return ncclSuccess;
}

ncclResult_t mpibDmaBufSupport(int dev) {
  (void)dev;
  return ncclInvalidUsage;
}

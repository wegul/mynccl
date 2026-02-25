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

static thread_local int mpibDmaBufSupportInitDev;
static std::once_flag mpibDmaBufOnces[MAX_IB_DEVS];

static void mpibDmaBufSupportInitOnce() {
  int dev = mpibDmaBufSupportInitDev;
  mpibDevs[dev].dmaBufSupported = -1;
  // Allocate a temporary PD for the probe.
  struct ibv_pd *pd = ibv_alloc_pd(mpibDevs[dev].context);
  if (pd == NULL)
    goto out;
  // fd=-1 probes whether ibv_reg_dmabuf_mr verb exists in the kernel.
  // EBADF = verb exists (fd is just invalid) → supported.
  // EOPNOTSUPP / EPROTONOSUPPORT = verb not available → not supported.
  {
    struct ibv_mr *mr =
        wrap_direct_ibv_reg_dmabuf_mr(pd, 0ULL, 0ULL, 0ULL, -1, 0);
    if (mr) {
      ibv_dereg_mr(mr);
      mpibDevs[dev].dmaBufSupported = 1;
    } else if (errno != EOPNOTSUPP && errno != EPROTONOSUPPORT) {
      mpibDevs[dev].dmaBufSupported = 1;
    }
  }
  ibv_dealloc_pd(pd);
out:
  INFO(NCCL_INIT | NCCL_NET, "NET/MPIB : DMA-BUF dev %d (%s): %s", dev,
       mpibDevs[dev].devName,
       mpibDevs[dev].dmaBufSupported == 1 ? "supported" : "not supported");
}

ncclResult_t mpibDmaBufSupport(int dev) {
  if (dev < 0 || dev >= mpibNIbDevs)
    return ncclInvalidUsage;
  mpibDmaBufSupportInitDev = dev;
  std::call_once(mpibDmaBufOnces[dev], mpibDmaBufSupportInitOnce);
  return mpibDevs[dev].dmaBufSupported == 1 ? ncclSuccess : ncclSystemError;
}

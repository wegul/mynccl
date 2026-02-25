#include "mpib_common.h"

static ncclResult_t mpibRegMrDmaBufInternal2(mpibNetCommDevBase *base,
                                             void *data, size_t size, int type,
                                             uint64_t offset, int fd,
                                             uint64_t mrFlags,
                                             ibv_mr **mhandle) {
  (void)type;
  (void)offset;
  static thread_local uintptr_t pageSize = 0;
  if (pageSize == 0)
    pageSize = sysconf(_SC_PAGESIZE);
  struct mpibMrCache *cache = &mpibDevs[base->ibDevN].mrCache;
  uintptr_t addr = (uintptr_t)data & -pageSize;
  size_t pages = ((uintptr_t)data + size - addr + pageSize - 1) / pageSize;
  std::lock_guard<std::mutex> lock(mpibDevs[base->ibDevN].mutex);
  for (int slot = 0;; slot++) {
    if (slot == cache->population || addr < cache->slots[slot].addr) {
      if (cache->population == cache->capacity) {
        cache->capacity = cache->capacity < 32 ? 32 : 2 * cache->capacity;
        NCCLCHECK(
            mpibRealloc(&cache->slots, cache->population, cache->capacity));
      }
      if (fd != -1)
        return ncclInvalidUsage;
      struct ibv_mr *mr;
      unsigned int flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
      bool relaxedOrdering = mpibRelaxedOrderingEnabled &&
                             (mrFlags & NCCL_NET_MR_FLAG_FORCE_SO) == 0;
      if (relaxedOrdering)
        flags |= IBV_ACCESS_RELAXED_ORDERING;
      if (relaxedOrdering) {
        NCCLCHECK(wrap_ibv_reg_mr_iova2(&mr, base->pd, (void *)addr,
                                        pages * pageSize, addr, flags));
      } else {
        NCCLCHECK(wrap_ibv_reg_mr(&mr, base->pd, (void *)addr, pages * pageSize,
                                  flags));
      }
      TRACE(NCCL_INIT | NCCL_NET,
            "regAddr=0x%lx size=%lld rkey=0x%x lkey=0x%x fd=%d",
            (unsigned long)addr, (long long)pages * pageSize, mr->rkey,
            mr->lkey, fd);
      if (slot != cache->population)
        memmove(cache->slots + slot + 1, cache->slots + slot,
                (cache->population - slot) * sizeof(struct mpibMr));
      cache->slots[slot].addr = addr;
      cache->slots[slot].pages = pages;
      cache->slots[slot].refs = 1;
      cache->slots[slot].mr = mr;
      cache->population += 1;
      *mhandle = mr;
      return ncclSuccess;
    } else if ((addr >= cache->slots[slot].addr) &&
               ((addr - cache->slots[slot].addr) / pageSize + pages) <=
                   cache->slots[slot].pages) {
      cache->slots[slot].refs += 1;
      *mhandle = cache->slots[slot].mr;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

static ncclResult_t mpibRegMrDmaBufInternal(void *comm, void *data, size_t size,
                                            int type, uint64_t offset, int fd,
                                            uint64_t mrFlags, void **mhandle) {
  ncclResult_t ret = ncclSuccess;
  assert(size > 0);
  struct mpibNetCommBase *base = (struct mpibNetCommBase *)comm;
  struct mpibMrHandle *mhandleWrapper =
      (struct mpibMrHandle *)malloc(sizeof(struct mpibMrHandle));
  for (int i = 0; i < base->vProps.ndevs; i++) {
    struct mpibNetCommDevBase *devComm = mpibGetNetCommDevBase(base, i);
    NCCLCHECKGOTO(mpibRegMrDmaBufInternal2(devComm, data, size, type, offset,
                                           fd, mrFlags,
                                           mhandleWrapper->mrs + i),
                  ret, fail);
  }
  *mhandle = (void *)mhandleWrapper;
exit:
  return ret;
fail:
  free(mhandleWrapper);
  goto exit;
}

__hidden ncclResult_t mpibRegMrDmaBuf(void *comm, void *data, size_t size,
                                      int type, uint64_t offset, int fd,
                                      void **mhandle) {
  (void)comm;
  (void)data;
  (void)size;
  (void)type;
  (void)offset;
  (void)fd;
  (void)mhandle;
  return ncclInvalidUsage;
}

__hidden ncclResult_t mpibRegMr(void *comm, void *data, size_t size, int type,
                                void **mhandle) {
  return mpibRegMrDmaBufInternal(comm, data, size, type, 0ULL, -1, 0, mhandle);
}

static ncclResult_t mpibDeregMrInternal(mpibNetCommDevBase *base,
                                        ibv_mr *mhandle) {
  struct mpibMrCache *cache = &mpibDevs[base->ibDevN].mrCache;
  std::lock_guard<std::mutex> lock(mpibDevs[base->ibDevN].mutex);
  for (int i = 0; i < cache->population; i++) {
    if (mhandle == cache->slots[i].mr) {
      if (0 == --cache->slots[i].refs) {
        memmove(&cache->slots[i], &cache->slots[--cache->population],
                sizeof(struct mpibMr));
        if (cache->population == 0) {
          free(cache->slots);
          cache->slots = NULL;
          cache->capacity = 0;
        }
        NCCLCHECK(wrap_ibv_dereg_mr(mhandle));
      }
      return ncclSuccess;
    }
  }
  WARN("NET/MPIB: could not find mr %p inside cache of %d entries", mhandle,
       cache->population);
  return ncclInternalError;
}

__hidden ncclResult_t mpibDeregMr(void *comm, void *mhandle) {
  if (mhandle == NULL)
    return ncclSuccess;

  struct mpibMrHandle *mhandleWrapper = (struct mpibMrHandle *)mhandle;
  struct mpibNetCommBase *base = (struct mpibNetCommBase *)comm;
  for (int i = 0; i < base->vProps.ndevs; i++) {
    struct mpibNetCommDevBase *devComm = mpibGetNetCommDevBase(base, i);
    NCCLCHECK(mpibDeregMrInternal(devComm, mhandleWrapper->mrs[i]));
  }
  free(mhandleWrapper);
  return ncclSuccess;
}

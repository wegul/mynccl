#pragma once

#include "mpib_common.h"

// Select CTS QP: always use the first QP on the selected device.
// Device alternates by fifoHead % ndevs; within each device, CTS is pinned
// to QP0 (not round-robin). This matches net_ib's design and ensures the
// simple signaling rule (slot == devIndex) works correctly.
//
// Data QPs use round-robin within each device (see mpibCommBaseGetQpForRequest).
static inline ncclResult_t
mpibRecvCommGetQpForCts(struct mpibRecvComm *recvComm, uint32_t id,
                        mpibQp **qp) {
  uint32_t ndevs = recvComm->base.vProps.ndevs;
  uint32_t dev = id % ndevs;
  // CTS always uses the first QP on the selected device
  uint32_t idx = (dev == 0) ? 0 : recvComm->base.nqpsSout;
  *qp = &recvComm->base.qps[idx];
  assert(*qp != NULL);
  return ncclSuccess;
}

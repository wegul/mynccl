#pragma once

#include "mpib_common.h"

static inline ncclResult_t
mpibRecvCommGetQpForCts(struct mpibRecvComm *recvComm, uint32_t id,
                        mpibQp **qp) {
  int devIndex = id % recvComm->base.vProps.ndevs;
  int qpIndex = 0;
  mpibCommBaseGetQpByIndex(&recvComm->base, devIndex, qpIndex, qp);
  assert(*qp != NULL);
  return ncclSuccess;
}

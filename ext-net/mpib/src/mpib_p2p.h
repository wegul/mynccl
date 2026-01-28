#pragma once

#include "mpib_common.h"

// Select CTS QP: always use SOUT (device 0) QP.
// CTS payload carries rkeys for both rails, so data path can still use
// both SOUT and SUP. Pinning CTS to one rail avoids unnecessary traffic
// on the second device.
//
// Data QPs use round-robin within each device (see
// mpibCommBaseGetQpForRequest).
static inline ncclResult_t
mpibRecvCommGetQpForCts(struct mpibRecvComm *recvComm, uint32_t id,
                        mpibQp **qp) {
  (void)id; // Unused: CTS always on SOUT
  *qp = &recvComm->base.qps[0];
  assert(*qp != NULL);
  return ncclSuccess;
}

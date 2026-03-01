#pragma once

#include "mpib_common.h"

// Select CTS QP: route CTS on the same rail as data traffic.
//   - SOUT-only connections (pathSupBw == 0): use first SOUT QP (qps[0]).
//   - SUP-only connections (pathSupBw == UINT32_MAX): use first SUP QP
//     (qps[nqpsSout]).
// This keeps all RDMA traffic (data + CTS) on the designated rail.
//
// Data QPs use round-robin within each device (see
// mpibCommBaseGetQpForRequest).
static inline ncclResult_t
mpibRecvCommGetQpForCts(struct mpibRecvComm *recvComm, uint32_t id,
                        mpibQp **qp) {
  (void)id;
  if (recvComm->base.pathSupBw == UINT32_MAX) {
    // SUP-only: use first SUP QP
    *qp = &recvComm->base.qps[recvComm->base.nqpsSout];
  } else {
    // SOUT-only (or mixed): use first SOUT QP
    *qp = &recvComm->base.qps[0];
  }
  assert(*qp != NULL);
  return ncclSuccess;
}

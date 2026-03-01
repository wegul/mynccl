#pragma once

#include "mpib_common.h"

// Select CTS QP: route CTS on the correct rail based on pathClass and mode.
//   - Vanilla mode (MPIB_MODE=0): strict path isolation.
//     INTRA_ISLAND → SUP QP (qps[nqpsSout]).
//     INTER_ISLAND → SOUT QP (qps[0]).
//   - Advanced mode (MPIB_MODE=1): CTS always on SOUT (agent doesn't
//     control CTS path).
//
// Data QPs use round-robin within each device (see
// mpibCommBaseGetQpForRequest).
static inline ncclResult_t
mpibRecvCommGetQpForCts(struct mpibRecvComm *recvComm, uint32_t id,
                        mpibQp **qp) {
  (void)id;
  const int mode = recvComm->base.mode;

  if (mode == 0) {
    // Vanilla: CTS follows strict path isolation
    if (recvComm->base.pathClass == MPIB_PATH_INTRA_ISLAND) {
      *qp = &recvComm->base.qps[recvComm->base.nqpsSout]; // SUP
    } else {
      *qp = &recvComm->base.qps[0]; // SOUT
    }
  } else {
    // Advanced: CTS always on SOUT (agent doesn't control CTS path)
    *qp = &recvComm->base.qps[0];
  }
  assert(*qp != NULL);
  return ncclSuccess;
}

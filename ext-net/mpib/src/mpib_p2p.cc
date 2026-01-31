#include "mpib_p2p.h"
#include "common.h"
#include "mpib_agent_client.h"
#include "mpib_common.h"
#include "mpib_compat.h"
#include <cassert>
#include <cstdint>

const char *mpibReqTypeStr[] = {"Unused", "Send", "Recv", "Flush", "IPut"};

// ===========================================================================
// SRQ refill helper
// Posts generic receive WRs to the SRQ when below low water mark.
// Called from mpibIrecv and mpibTest for safety.
// ===========================================================================
static ncclResult_t mpibSrqCheckAndRefill(struct mpibRecvComm *comm) {
  for (int devIndex = 0; devIndex < comm->base.vProps.ndevs; devIndex++) {
    struct mpibNetCommDevBase *devBase = &comm->devs[devIndex].base;
    if (devBase->srq == NULL)
      continue;
    if (devBase->srqPosted >= MPIB_SRQ_LOW_WATER)
      continue;

    // Refill to high water mark
    struct ibv_recv_wr wr;
    struct ibv_recv_wr *bad_wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = 0;      // Generic WR; wr_id is unused with SRQ
    wr.sg_list = NULL; // 0 SGE - receiver gets data via RDMA WRITE
    wr.num_sge = 0;
    wr.next = NULL;

    while (devBase->srqPosted < MPIB_SRQ_HIGH_WATER) {
      ncclResult_t ret = wrap_ibv_post_srq_recv(devBase->srq, &wr, &bad_wr);
      if (ret != ncclSuccess)
        return ret;
      devBase->srqPosted++;
    }
  }
  return ncclSuccess;
}

ncclResult_t mpibGetRequest(struct mpibNetCommBase *base,
                            struct mpibRequest **req) {
  for (int i = 0; i < NET_IB_MAX_REQUESTS; i++) {
    struct mpibRequest *r = base->reqs + i;
    if (r->type == MPIB_NET_IB_REQ_UNUSED) {
      r->base = base;
      r->sock = NULL;
      memset(r->devBases, 0, sizeof(r->devBases));
      memset(r->events, 0, sizeof(r->events));
      // Clear SRQ tracking fields
      r->slot = 0;
      r->expected_mask = 0;
      r->seen_mask = 0;
      *req = r;
      return ncclSuccess;
    }
  }
  WARN("NET/MPIB : unable to allocate requests");
  *req = NULL;
  return ncclInternalError;
}

ncclResult_t mpibFreeRequest(struct mpibRequest *r) {
  r->type = MPIB_NET_IB_REQ_UNUSED;
  return ncclSuccess;
}

void mpibAddEvent(struct mpibRequest *req, int devIndex) {
  struct mpibNetCommDevBase *base = mpibGetNetCommDevBase(req->base, devIndex);
  req->events[devIndex]++;
  req->devBases[devIndex] = base;
}

__hidden ncclResult_t mpibIsend(void *sendComm, void *data, size_t size,
                                int tag, void *mhandle, void *phandle,
                                void **request) {
  struct mpibSendComm *comm = (struct mpibSendComm *)sendComm;
  if (comm->base.ready == 0) {
    WARN("NET/MPIB: mpibIsend() called when comm->base.ready == 0");
    *request = NULL;
    return ncclInternalError;
  }
  NCCLCHECK(mpibStatsCheckFatalCount(&comm->base.stats, __func__));

  struct mpibMrHandle *mhandleWrapper = (struct mpibMrHandle *)mhandle;

  volatile struct mpibSendFifo *slots;

  const int slot = comm->base.fifoHead % NET_IB_MAX_REQUESTS;
  struct mpibRequest **reqs = comm->fifoReqs[slot];
  slots = comm->ctsFifo[slot];
  uint64_t idx = comm->base.fifoHead + 1;
  if (slots[0].idx != idx) {
    *request = NULL;
    return ncclSuccess;
  }
  const uint32_t nreqs = slots[0].nreqs;
  if (nreqs == 0 || nreqs > MPIB_NET_IB_MAX_RECVS) {
    *request = NULL;
    return ncclInternalError;
  }
  for (uint32_t r = 1; r < nreqs; r++) {
    if (slots[r].idx != idx) {
      *request = NULL;
      return ncclSuccess;
    }
  }
  std::atomic_thread_fence(std::memory_order_seq_cst);
  for (uint32_t r = 0; r < nreqs; r++) {
    if (reqs[r] != NULL || slots[r].tag != tag)
      continue;

    if (size > slots[r].size)
      size = slots[r].size;
    if (slots[r].addr == 0 || slots[r].rkeys[0] == 0) {
      char line[MPIB_SOCKET_NAME_MAXLEN + 1];
      union mpibSocketAddress addr;
      mpibSocketGetAddr(&comm->base.sock, &addr);
      WARN("NET/MPIB: peer=%s Incorrect fifo setup recvSize=%ld addr=%lx "
           "rkey=%x",
           mpibSocketToString(&addr, line), slots[r].size, slots[r].addr,
           slots[r].rkeys[0]);
      return ncclInternalError;
    }

    struct mpibRequest *req;
    NCCLCHECK(mpibGetRequest(&comm->base, &req));
    req->type = MPIB_NET_IB_REQ_SEND;
    req->sock = &comm->base.sock;
    req->base = &comm->base;
    req->nreqs = nreqs;
    req->send.size = size;
    req->send.data = data;
    req->send.offset = 0;

    for (int i = 0; i < comm->base.vProps.ndevs; i++)
      req->send.lkeys[i] = mhandleWrapper ? mhandleWrapper->mrs[i]->lkey : 0;

    *request = reqs[r] = req;

    for (uint32_t r2 = 0; r2 < nreqs; r2++)
      if (reqs[r2] == NULL)
        return ncclSuccess;

    TIME_START(0);

    // =========================================================================
    // WR Construction and Posting (SRQ-aware)
    //
    // Data is split between SOUT (dev0) and SUP (dev1). Each device uses
    // exactly one QP per request (no intra-device striping).
    //
    // SRQ Protocol:
    //   - Compute active_mask from split (bit0=SOUT, bit1=SUP)
    //   - Only post to active rails (skip inactive rails entirely)
    //   - Encode imm_data = (slot | mask << 8 | size_q << 10)
    //   - Leader rail (lowest bit in active_mask) writes cmplsRecords
    //
    // WR chain structure (per active QP):
    //   wrs[0..nreqs-2]: IBV_WR_RDMA_WRITE (data only, not signaled)
    //   wrs[nreqs-1]:    IBV_WR_RDMA_WRITE_WITH_IMM (data + completion signal)
    // =========================================================================
    {
      // Build wr_id from packed request indices
      uint64_t wr_id = 0ULL;
      for (uint32_t i = 0; i < nreqs; i++)
        wr_id |= (uint64_t)(reqs[i] - comm->base.reqs) << (i * 8);

      // 128B alignment for device-level split (LL/LL128 protocol compatibility)
      const size_t align = 128;

      // Per-request sizes for each device
      size_t sizeSout[MPIB_NET_IB_MAX_RECVS];
      size_t sizeSup[MPIB_NET_IB_MAX_RECVS];

      // Read hint from agent and compute split ratio
      const uint32_t sup_bw = mpibAgentReadHint(comm->hint_slot);
      const float sup_ratio =
          (sup_bw == 0) ? 0.0f : ((float)sup_bw / (1.0f + (float)sup_bw));

      for (uint32_t i = 0; i < nreqs; i++) {
        const size_t reqSize = reqs[i]->send.size;
        size_t sup_raw = (size_t)(reqSize * sup_ratio);
        size_t sup = (sup_raw / align) * align;
        if (sup > reqSize)
          sup = reqSize;
        sizeSup[i] = sup;
        sizeSout[i] = reqSize - sizeSup[i];
      }

      // Compute active_mask: OR of rails that have > 0 bytes across all
      // requests bit0 = SOUT active, bit1 = SUP active
      uint8_t active_mask = 0;
      for (uint32_t i = 0; i < nreqs; i++) {
        if (sizeSout[i] > 0)
          active_mask |= 0x1;
        if (sizeSup[i] > 0)
          active_mask |= 0x2;
      }
      if (active_mask == 0)
        active_mask = 0x1;

      // Leader rail selection: lowest bit in active_mask
      // If SOUT active (bit0) -> leader = 0 (SOUT)
      // Else (only SUP active) -> leader = 1 (SUP)
      const int leaderDev = (active_mask & 0x1) ? 0 : 1;

      // Record sizes in cmplsRecords buffer (for leader to write)
      int *sizesRecord = comm->remCmplsRecords.elems[slot];
      for (uint32_t i = 0; i < nreqs; i++)
        sizesRecord[i] = (int)reqs[i]->send.size;

      // Compute size_q for IMM encoding
      // For first implementation: always use sentinel (consult cmplsRecords)
      const uint32_t size_q = MPIB_IMM_SIZEQ_SENTINEL;

      // Encode IMM data: (slot | mask << 8 | size_q << 10)
      const uint32_t immData =
          mpibImmEncode((uint8_t)slot, active_mask, size_q);

      // Prepare data WRs: all are RDMA_WRITE with next pointing to next WR
      for (uint32_t r = 0; r < nreqs; r++) {
        struct ibv_send_wr *wr = comm->wrs + r;
        memset(wr, 0, sizeof(struct ibv_send_wr));

        struct ibv_sge *sge = comm->sges + r;
        sge->addr = (uintptr_t)reqs[r]->send.data;
        wr->opcode = IBV_WR_RDMA_WRITE;
        wr->send_flags = 0;
        wr->wr.rdma.remote_addr = slots[r].addr;
        wr->next = wr + 1;
      }

      // Set up lastWr for completion signaling
      // For first impl, always use a separate WR for signaling to avoid SGE
      // conflicts wrs[0..nreqs-1] = data WRs, wrs[nreqs] = signaling WR
      struct ibv_send_wr *lastWr = comm->wrs + nreqs;
      memset(lastWr, 0, sizeof(struct ibv_send_wr));
      lastWr->wr.rdma.remote_addr =
          comm->remCmplsRecords.addr + (uint64_t)slot *
                                           (uint64_t)MPIB_NET_IB_MAX_RECVS *
                                           (uint64_t)sizeof(int);
      // Link the last data WR to the signaling WR
      comm->wrs[nreqs - 1].next = lastWr;
      lastWr->wr_id = wr_id;
      lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
      lastWr->imm_data = htobe32(immData);
      lastWr->next = NULL;
      lastWr->send_flags = IBV_SEND_SIGNALED;

      // Post WRs only to active rails
      const int nqps = mpibCommBaseGetNqpsPerRequest(&comm->base);
      for (int i = 0; i < nqps; i++) {
        mpibQp *qpPtr;
        uint32_t qpIdx;
        NCCLCHECK(mpibCommBaseGetQpForRequest(&comm->base, comm->base.fifoHead,
                                              i, &qpPtr, &qpIdx));
        const int devIndex = qpPtr->devIndex; // 0 = SOUT, 1 = SUP
        const uint8_t devBit = (uint8_t)(1 << devIndex);

        // Skip inactive rails entirely
        if ((active_mask & devBit) == 0)
          continue;

        // Add event for this active device
        mpibAddEvent(req, devIndex);

        // Set up all data WRs (wrs[0..nreqs-1]) for this device
        for (uint32_t j = 0; j < nreqs; j++) {
          comm->wrs[j].wr.rdma.rkey = slots[j].rkeys[qpPtr->remDevIdx];

          const size_t devBaseOffset = (devIndex == 0) ? 0 : sizeSout[j];
          const size_t length = (devIndex == 0) ? sizeSout[j] : sizeSup[j];

          if (length <= 0) {
            comm->wrs[j].sg_list = NULL;
            comm->wrs[j].num_sge = 0;
          } else {
            comm->sges[j].lkey = reqs[j]->send.lkeys[devIndex];
            comm->sges[j].length = length;
            comm->sges[j].addr = (uintptr_t)reqs[j]->send.data + devBaseOffset;
            comm->wrs[j].wr.rdma.remote_addr = slots[j].addr + devBaseOffset;
            comm->wrs[j].sg_list = comm->sges + j;
            comm->wrs[j].num_sge = 1;
          }
        }

        // Leader rail writes cmplsRecords; non-leader sends IMM-only
        if (devIndex == leaderDev) {
          // Leader: attach SGE to write sizes to remote cmplsRecords
          lastWr->sg_list = &(comm->devs[devIndex].sge);
          lastWr->sg_list[0].addr = (uint64_t)sizesRecord;
          lastWr->sg_list[0].length = nreqs * sizeof(int);
          lastWr->num_sge = 1;
          lastWr->wr.rdma.rkey = comm->remCmplsRecords.rkeys[qpPtr->remDevIdx];
        } else {
          // Non-leader: IMM-only (no SGE for cmplsRecords)
          lastWr->sg_list = NULL;
          lastWr->num_sge = 0;
          // Still need rkey for the RDMA_WRITE_WITH_IMM (even if 0 bytes)
          lastWr->wr.rdma.rkey = comm->remCmplsRecords.rkeys[qpPtr->remDevIdx];
        }

        struct ibv_send_wr *bad_wr;
        NCCLCHECK(wrap_ibv_post_send(qpPtr->qp, comm->wrs, &bad_wr));
      }
    }
    // =========================================================================

    memset((void *)slots, 0, MPIB_NET_IB_MAX_RECVS * sizeof(*slots));
    memset((void *)reqs, 0,
           MPIB_NET_IB_MAX_RECVS * sizeof(struct mpibRequest *));
    comm->base.fifoHead++;
    TIME_STOP(0);
    return ncclSuccess;
  }

  *request = NULL;
  return ncclSuccess;
}

static ncclResult_t mpibPostFifo(struct mpibRecvComm *comm, int n, void **data,
                                 size_t *sizes, int *tags, void **mhandles,
                                 struct mpibRequest *req) {
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));

  uint32_t slot = comm->base.fifoHead % NET_IB_MAX_REQUESTS;
  req->recv.sizes = comm->cmplsRecords[slot];
  for (int i = 0; i < n; i++)
    req->recv.sizes[i] = 0;
  struct mpibSendFifo *localElem = comm->remCtsFifo.elems[slot];

  mpibQp *ctsQp = NULL;
  NCCLCHECK(mpibRecvCommGetQpForCts(comm, comm->base.fifoHead, &ctsQp));

  for (int i = 0; i < n; i++) {
    localElem[i].addr = (uint64_t)data[i];
    struct mpibMrHandle *mhandleWrapper = (struct mpibMrHandle *)mhandles[i];

    for (int j = 0; j < comm->base.vProps.ndevs; j++)
      localElem[i].rkeys[j] = mhandleWrapper ? mhandleWrapper->mrs[j]->rkey : 0;

    localElem[i].nreqs = n;
    localElem[i].size = sizes[i];
    localElem[i].tag = tags[i];
    localElem[i].idx = comm->base.fifoHead + 1;
  }
  wr.wr.rdma.remote_addr =
      comm->remCtsFifo.addr +
      slot * MPIB_NET_IB_MAX_RECVS * sizeof(struct mpibSendFifo);
  wr.wr.rdma.rkey = comm->base.remDevs[ctsQp->remDevIdx].rkey;
  wr.sg_list = &(comm->devs[ctsQp->devIndex].sge);
  wr.sg_list[0].addr = (uint64_t)localElem;
  wr.sg_list[0].length = n * sizeof(struct mpibSendFifo);
  wr.num_sge = 1;

  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = comm->remCtsFifo.flags;

  // Signal every MPIB_CTS_SIGNAL_INTERVAL slots to drain the single CTS QP.
  // CTS is pinned to SOUT (device 0), so all 256 slots go through one QP.
  if ((slot % MPIB_CTS_SIGNAL_INTERVAL) == 0) {
    wr.send_flags |= IBV_SEND_SIGNALED;
    wr.wr_id = req - comm->base.reqs;
    mpibAddEvent(req, ctsQp->devIndex);
  }
  struct ibv_send_wr *bad_wr;
  NCCLCHECK(wrap_ibv_post_send(ctsQp->qp, &wr, &bad_wr));
  comm->base.fifoHead++;

  return ncclSuccess;
}

__hidden ncclResult_t mpibIrecv(void *recvComm, int n, void **data,
                                size_t *sizes, int *tags, void **mhandles,
                                void **phandles, void **request) {
  struct mpibRecvComm *comm = (struct mpibRecvComm *)recvComm;
  if (comm->base.ready == 0) {
    WARN("NET/MPIB: mpibIrecv() called when comm->base.ready == 0");
    *request = NULL;
    return ncclInternalError;
  }
  if (n > MPIB_NET_IB_MAX_RECVS)
    return ncclInternalError;
  NCCLCHECK(mpibStatsCheckFatalCount(&comm->base.stats, __func__));

  struct mpibRequest *req;
  NCCLCHECK(mpibGetRequest(&comm->base, &req));
  req->type = MPIB_NET_IB_REQ_RECV;
  req->sock = &comm->base.sock;
  req->nreqs = n;

  // =========================================================================
  // SRQ-based receive: compute slot and record slot→request mapping
  // =========================================================================
  const uint8_t slot = (uint8_t)(comm->base.fifoHead % NET_IB_MAX_REQUESTS);

  // ASSERTION: slot must not already be in use (detect slot reuse while
  // outstanding)
  assert(comm->base.slotReq[slot] == NULL &&
         "SRQ protocol error: slot reuse while request outstanding");

  // Initialize SRQ tracking state
  req->slot = slot;
  req->expected_mask = 0; // Learned from first arriving IMM
  req->seen_mask = 0;
  comm->base.slotReq[slot] = req;

  // Store devBases for completion polling (needed for CQ access)
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  TIME_START(1);
  // SRQ: refill if needed (replaces per-QP ibv_post_recv)
  NCCLCHECK(mpibSrqCheckAndRefill(comm));
  TIME_STOP(1);

  TIME_START(2);
  NCCLCHECK(mpibPostFifo(comm, n, data, sizes, tags, mhandles, req));
  TIME_STOP(2);

  *request = req;
  return ncclSuccess;
}

__hidden ncclResult_t mpibIflush(void *recvComm, int n, void **data, int *sizes,
                                 void **mhandles, void **request) {
  (void)recvComm;
  (void)n;
  (void)data;
  (void)sizes;
  (void)mhandles;
  if (request)
    *request = NULL;
  return ncclSuccess;
}

static inline bool mpibRequestIsComplete(struct mpibRequest *request) {
  if (request->type == MPIB_NET_IB_REQ_RECV) {
    // SRQ-based RECV: complete when seen_mask == expected_mask (and mask is
    // set)
    return (request->expected_mask != 0 &&
            request->seen_mask == request->expected_mask);
  }
  // SEND: use events-based completion (unchanged)
  return (request->events[0] == 0 && request->events[1] == 0);
}

static inline ncclResult_t mpibRequestComplete(struct mpibRequest *r, int *done,
                                               int *sizes) {
  TRACE(NCCL_NET, "r=%p done type=%d slot=%d", r, r->type, r->slot);
  *done = 1;
  if (sizes && r->type == MPIB_NET_IB_REQ_RECV) {
    for (uint32_t i = 0; i < r->nreqs; i++)
      sizes[i] = r->recv.sizes[i];
    // Clear slot->request mapping
    r->base->slotReq[r->slot] = NULL;
  }
  if (sizes && r->type == MPIB_NET_IB_REQ_SEND) {
    sizes[0] = r->send.size;
  }
  NCCLCHECK(mpibFreeRequest(r));
  return ncclSuccess;
}

// ===========================================================================
// Completion Event Processing
//
// SEND completions: use wr_id-based tracking (unchanged)
// RECV completions (SRQ): decode imm_data to find slot/mask, use mask learning
// ===========================================================================
static inline ncclResult_t
mpibCompletionEventProcess(struct mpibNetCommBase *commBase, struct ibv_wc *wc,
                           int devIndex, struct mpibNetCommDevBase *devBase) {
  // RECV completion with SRQ: IBV_WC_RECV_RDMA_WITH_IMM
  if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
    // Decode IMM data: (slot | mask << 8 | size_q << 10)
    const uint32_t imm = be32toh(wc->imm_data);
    uint8_t slot, active_mask;
    uint32_t size_q;
    mpibImmDecode(imm, &slot, &active_mask, &size_q);

    // ASSERTION: active_mask must be valid (1, 2, or 3)
    assert(active_mask != 0 && active_mask <= 3 &&
           "SRQ protocol error: invalid active_mask in IMM");

    // Lookup request from slot
    struct mpibRequest *req = commBase->slotReq[slot];

    // ASSERTION: request must exist for this slot
    assert(req != NULL && "SRQ protocol error: IMM for slot with no request");
    assert(req->type == MPIB_NET_IB_REQ_RECV &&
           "SRQ protocol error: IMM for non-RECV request");

    // Mask learning: set expected_mask on first IMM arrival
    if (req->expected_mask == 0) {
      req->expected_mask = active_mask;
    }

    // Mark this rail as seen
    req->seen_mask |= (uint8_t)(1 << devIndex);

    // Decrement SRQ posted count (each CQE consumes one SRQ WQE)
    if (devBase->srq != NULL) {
      devBase->srqPosted--;
    }

    // Size handling: read from cmplsRecords (always, for first implementation)
    // Note: req->recv.sizes was set up in mpibPostFifo to point to
    // cmplsRecords[slot] The sender writes sizes there via leader rail, so
    // sizes are already in place. For size_q != SENTINEL optimization (future):
    // could decode size from imm.

    return ncclSuccess;
  }

  // SEND completion: use wr_id-based tracking (unchanged from original)
  const uint64_t wr_id = wc->wr_id;

  const uint32_t reqIndex0 = (uint32_t)(wr_id & 0xff);
  if (reqIndex0 >= NET_IB_MAX_REQUESTS)
    return ncclInternalError;
  struct mpibRequest *req0 = commBase->reqs + reqIndex0;

  // Packed SEND completion: decrement each referenced request once for this dev
  if (req0->type == MPIB_NET_IB_REQ_SEND && req0->nreqs > 1) {
    if (req0->nreqs == 0 || req0->nreqs > MPIB_NET_IB_MAX_RECVS)
      return ncclInternalError;
    for (uint32_t j = 0; j < req0->nreqs; j++) {
      const uint32_t reqIndex = (uint32_t)((wr_id >> (j * 8)) & 0xff);
      if (reqIndex >= NET_IB_MAX_REQUESTS)
        return ncclInternalError;
      struct mpibRequest *sendReq = commBase->reqs + reqIndex;
      if (sendReq->events[devIndex] <= 0)
        return ncclInternalError;
      sendReq->events[devIndex]--;
    }
    return ncclSuccess;
  }

  // Single SEND completion (nreqs == 1)
  if (req0->events[devIndex] <= 0)
    return ncclInternalError;
  req0->events[devIndex]--;
  return ncclSuccess;
}

__hidden ncclResult_t mpibTest(void *request, int *done, int *sizes) {
  struct mpibRequest *r = (struct mpibRequest *)request;
  *done = 0;
  int totalWrDone = 0;
  int wrDone = 0;
  struct ibv_wc wcs[4];
  do {
    NCCLCHECK(mpibStatsCheckFatalCount(&r->base->stats, __func__));
    if (mpibRequestIsComplete(r))
      return mpibRequestComplete(r, done, sizes);

    totalWrDone = 0;
    for (int i = 0; i < MPIB_MAX_DEVS; i++) {
      if (r->devBases[i] == NULL)
        continue;

      // For RECV with SRQ: always poll (don't check events[] since we use
      // mask-based completion) For SEND: only poll if events[i] > 0
      if (r->type != MPIB_NET_IB_REQ_RECV && r->events[i] == 0)
        continue;

      wrDone = 0;
      NCCLCHECK(wrap_ibv_poll_cq(r->devBases[i]->cq, 4, wcs, &wrDone));
      if (wrDone > 0) {
        totalWrDone += wrDone;
        for (int j = 0; j < wrDone; j++) {
          if (wcs[j].status != IBV_WC_SUCCESS) {
            WARN("NET/MPIB: CQ error status=%d opcode=%d", wcs[j].status,
                 wcs[j].opcode);
            return ncclSystemError;
          }
          NCCLCHECK(
              mpibCompletionEventProcess(r->base, wcs + j, i, r->devBases[i]));
        }
      }
    }

    // Safety: refill SRQ for RECV requests during long polling
    if (r->type == MPIB_NET_IB_REQ_RECV) {
      // Cast to mpibRecvComm to call refill helper
      // r->base is mpibNetCommBase, which is the first member of mpibRecvComm
      struct mpibRecvComm *rComm =
          (struct mpibRecvComm *)((char *)r->base -
                                  offsetof(struct mpibRecvComm, base));
      NCCLCHECK(mpibSrqCheckAndRefill(rComm));
    }
  } while (totalWrDone > 0);

  return ncclSuccess;
}

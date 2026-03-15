#include "mpib_p2p.h"
#include "common.h"
#include "mpib_agent_client.h"
#include "mpib_common.h"
#include "mpib_compat.h"
#include <cassert>
#include <cstdint>

const char *mpibReqTypeStr[] = {"Unused", "Send", "Recv", "Flush", "IPut"};

ncclResult_t mpibGetRequest(struct mpibNetCommBase *base,
                            struct mpibRequest **req) {
  for (int i = 0; i < NET_IB_MAX_REQUESTS; i++) {
    struct mpibRequest *r = base->reqs + i;
    if (r->type == MPIB_NET_IB_REQ_UNUSED) {
      r->base = base;
      r->sock = NULL;
      memset(r->events, 0, sizeof(r->events));
      memset((void *)r->devBases, 0, sizeof(r->devBases));
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

// ===========================================================================
// Weighted rail/QP selection (recv side)
//
// Selects one QP for a message based on supbwHint:
//   UINT32_MAX → SUP-only (intra-island vanilla)
//   0          → SOUT-only (inter-island vanilla)
//   1..1022    → weighted round-robin (advanced mode)
//
// Returns flat qps[] index. Sets *outDevIndex.
// ===========================================================================
static int mpibWeightedSelectQp(struct mpibRecvComm *comm, uint32_t supbwHint,
                                int *outDevIndex) {
  struct mpibNetCommBase *base = &comm->base;

  // Guard: if no SUP QPs exist, always use SOUT
  if (base->nqpsSup == 0 || supbwHint == 0) {
    *outDevIndex = 0;
    return (int)(base->qpCursorSout++ % base->nqpsSout);
  }

  if (supbwHint >= 1024) {
    // SUP-only
    *outDevIndex = 1;
    return (int)(base->nqpsSout + (base->qpCursorSup++ % base->nqpsSup));
  }

  // Weighted split: the fraction (totalCursor % 1024) cycles 0..1023.
  // supbwHint is parts-per-1024 of SUP share.
  // Each message independently lands on the correct rail at the current ratio.
  if ((base->totalCursor++ % 1024) < supbwHint) {
    *outDevIndex = 1;
    return (int)(base->nqpsSout + (base->qpCursorSup++ % base->nqpsSup));
  }
  *outDevIndex = 0;
  return (int)(base->qpCursorSout++ % base->nqpsSout);
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
    // WR Construction and Posting (single QP, mirrors net-ib)
    //
    // Receiver chose the rail/QP; sender reads selection from CTS metadata.
    // One QP per message — no split, no active_mask.
    //
    // WR chain structure:
    //   wrs[0..nreqs-2]: IBV_WR_RDMA_WRITE (data only, not signaled)
    //   wrs[nreqs-1]:    IBV_WR_RDMA_WRITE_WITH_IMM when nreqs == 1
    //                    or IBV_WR_RDMA_WRITE when nreqs > 1 (extra WR follows)
    //   wrs[nreqs]:      IBV_WR_RDMA_WRITE_WITH_IMM (cmplsRecords + IMM)
    //                    only when nreqs > 1
    // =========================================================================
    {
      // Build wr_id from packed request indices (matches net-ib)
      uint64_t wr_id = 0ULL;
      for (uint32_t i = 0; i < nreqs; i++)
        wr_id |= (uint64_t)(reqs[i] - comm->base.reqs) << (i * 8);

      // Read receiver's QP selection from CTS
      const uint8_t selDevIdx = slots[0].selectedDevIndex;
      const uint8_t selQpIdx = slots[0].selectedQpIndex;
      struct mpibQp *selectedQp = &comm->base.qps[selQpIdx];

      // IMM carries size (net-ib convention: nreqs==1 uses it, nreqs>1 ignores)
      const uint32_t immData = (uint32_t)reqs[0]->send.size;

      // Prepare data WRs
      for (uint32_t r = 0; r < nreqs; r++) {
        struct ibv_send_wr *wr = comm->wrs + r;
        memset(wr, 0, sizeof(struct ibv_send_wr));
        wr->wr.rdma.rkey = slots[r].rkeys[selectedQp->remDevIdx];
        wr->wr.rdma.remote_addr = slots[r].addr;
        wr->opcode = IBV_WR_RDMA_WRITE;
        wr->send_flags = 0;
        wr->next = wr + 1;

        comm->sges[r].addr = (uintptr_t)reqs[r]->send.data;
        comm->sges[r].length = reqs[r]->send.size;
        comm->sges[r].lkey = reqs[r]->send.lkeys[selDevIdx];
        wr->sg_list = comm->sges + r;
        wr->num_sge = 1;
      }

      struct ibv_send_wr *lastWr = comm->wrs + nreqs - 1;
      if (nreqs > 1) {
        // Multi-recv: use a separate signaling WR that writes cmplsRecords
        int *sizesRecord = comm->remCmplsRecords.elems[slot];
        for (uint32_t r = 0; r < nreqs; r++)
          sizesRecord[r] = (int)reqs[r]->send.size;

        lastWr = comm->wrs + nreqs;
        memset(lastWr, 0, sizeof(struct ibv_send_wr));
        lastWr->wr.rdma.remote_addr =
            comm->remCmplsRecords.addr +
            (uint64_t)slot * MPIB_NET_IB_MAX_RECVS * sizeof(int);
        lastWr->sg_list = &(comm->devs[selDevIdx].sge);
        lastWr->sg_list[0].addr = (uint64_t)sizesRecord;
        lastWr->sg_list[0].length = nreqs * sizeof(int);
        lastWr->num_sge = 1;
        lastWr->wr.rdma.rkey = comm->remCmplsRecords.rkeys[selDevIdx];
        comm->wrs[nreqs - 1].next = lastWr;
      }
      lastWr->wr_id = wr_id;
      lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
      lastWr->imm_data = htobe32(immData);
      lastWr->next = NULL;
      lastWr->send_flags = IBV_SEND_SIGNALED;

      // Post once on the receiver-chosen QP
      mpibAddEvent(req, selDevIdx);
      struct ibv_send_wr *bad_wr;
      NCCLCHECK(wrap_ibv_post_send(selectedQp->qp, comm->wrs, &bad_wr));
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
                                 struct mpibRequest *req,
                                 uint8_t selectedDevIndex,
                                 uint8_t selectedQpIndex) {
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
    localElem[i].selectedDevIndex = selectedDevIndex;
    localElem[i].selectedQpIndex = selectedQpIndex;
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

  // Select one rail/QP via weighted round-robin
  const uint32_t supbwHint = mpibGetSupBw(comm, 0);
  int selectedDevIndex;
  const int selectedQpIdx =
      mpibWeightedSelectQp(comm, supbwHint, &selectedDevIndex);
  struct mpibQp *selectedQp = &comm->base.qps[selectedQpIdx];
  struct ibv_recv_wr rwr;
  memset(&rwr, 0, sizeof(rwr));
  rwr.wr_id = req - comm->base.reqs;
  rwr.sg_list = NULL;
  rwr.num_sge = 0;

  TIME_START(1);
  struct ibv_recv_wr *bad_wr;
  NCCLCHECK(wrap_ibv_post_recv(selectedQp->qp, &rwr, &bad_wr));
  mpibAddEvent(req, selectedDevIndex);
  TIME_STOP(1);

  // Post CTS to notify sender (carries selectedDevIndex/selectedQpIndex)
  TIME_START(2);
  NCCLCHECK(mpibPostFifo(comm, n, data, sizes, tags, mhandles, req,
                         (uint8_t)selectedDevIndex, (uint8_t)selectedQpIdx));
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
  return (request->events[0] == 0 && request->events[1] == 0);
}

static inline ncclResult_t mpibRequestComplete(struct mpibRequest *r, int *done,
                                               int *sizes) {
  TRACE(NCCL_NET, "r=%p done type=%d", r, r->type);
  *done = 1;
  if (sizes && r->type == MPIB_NET_IB_REQ_RECV) {
    for (uint32_t i = 0; i < r->nreqs; i++)
      sizes[i] = r->recv.sizes[i];
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
// RECV: wr_id-based request lookup (mirrors net-ib)
// SEND: wr_id-based packed request decrement
// ===========================================================================
static inline ncclResult_t
mpibCompletionEventProcess(struct mpibNetCommBase *commBase, struct ibv_wc *wc,
                           int devIndex) {
  // RECV completion: IBV_WC_RECV_RDMA_WITH_IMM
  if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
    const uint32_t reqIndex = (uint32_t)(wc->wr_id & 0xff);
    if (reqIndex >= NET_IB_MAX_REQUESTS)
      return ncclInternalError;
    struct mpibRequest *req = commBase->reqs + reqIndex;
    assert(req->type == MPIB_NET_IB_REQ_RECV &&
           "RECV completion for non-RECV request");

    // nreqs == 1: size carried in IMM (net-ib convention)
    if (req->nreqs == 1) {
      req->recv.sizes[0] = be32toh(wc->imm_data);
    }
    // nreqs > 1: sizes already in cmplsRecords, written by sender's lastWr

    req->events[devIndex]--;
    return ncclSuccess;
  }

  // SEND completion: wr_id-based packed request decrement
  const uint64_t wr_id = wc->wr_id;
  const uint32_t reqIndex0 = (uint32_t)(wr_id & 0xff);
  if (reqIndex0 >= NET_IB_MAX_REQUESTS)
    return ncclInternalError;
  struct mpibRequest *req0 = commBase->reqs + reqIndex0;

  if (req0->type == MPIB_NET_IB_REQ_SEND && req0->nreqs > 1) {
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

  // Single SEND or CTS completion
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
      if (r->devBases[i] == NULL || r->events[i] == 0)
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
          NCCLCHECK(mpibCompletionEventProcess(r->base, wcs + j, i));
        }
      }
    }
  } while (totalWrDone > 0);

  return ncclSuccess;
}

#include "mpib_p2p.h"
#include "mpib_common.h"
#include "mpib_compat.h"
#include <cstdint>

const char *mpibReqTypeStr[] = {"Unused", "Send", "Recv", "Flush", "IPut"};

ncclResult_t mpibGetRequest(struct mpibNetCommBase *base,
                            struct mpibRequest **req) {
  for (int i = 0; i < NET_IB_MAX_REQUESTS; i++) {
    struct mpibRequest *r = base->reqs + i;
    if (r->type == MPIB_NET_IB_REQ_UNUSED) {
      r->base = base;
      r->sock = NULL;
      memset(r->devBases, 0, sizeof(r->devBases));
      memset(r->events, 0, sizeof(r->events));
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

    // Populate events per QP (one QP per device)
    int nqps = mpibCommBaseGetNqpsPerRequest(&comm->base);
    uint32_t qpIndex = 0;
    mpibQp *qp = NULL;
    for (int i = 0; i < nqps; i++) {
      NCCLCHECK(mpibCommBaseGetQpForRequest(&comm->base, comm->base.fifoHead, i,
                                            &qp, &qpIndex));
      mpibAddEvent(req, qp->devIndex);
    }

    for (int i = 0; i < comm->base.vProps.ndevs; i++)
      req->send.lkeys[i] = mhandleWrapper ? mhandleWrapper->mrs[i]->lkey : 0;

    *request = reqs[r] = req;

    for (uint32_t r2 = 0; r2 < nreqs; r2++)
      if (reqs[r2] == NULL)
        return ncclSuccess;

    TIME_START(0);

    // =========================================================================
    // WR Construction and Posting
    //
    // Data is split between SOUT (dev0) and SUP (dev1). Each device uses
    // exactly one QP per request (no intra-device striping).
    //
    // WR chain structure (per QP):
    //   wrs[0..nreqs-2]: IBV_WR_RDMA_WRITE (data only, not signaled)
    //   wrs[nreqs-1]:    IBV_WR_RDMA_WRITE_WITH_IMM (data + completion signal)
    //
    // For nreqs > 1, the last WR writes completion sizes to remCmplsRecords.
    // For nreqs == 1, the single WR carries data directly with IMM.
    // =========================================================================
    {
      // Build wr_id from packed request indices
      uint64_t wr_id = 0ULL;
      for (uint32_t i = 0; i < nreqs; i++)
        wr_id |= (uint64_t)(reqs[i] - comm->base.reqs) << (i * 8);

      // Record sizes for multi-recv
      uint32_t immData = reqs[0]->send.size;
      if (nreqs > 1) {
        int *sizes = comm->remCmplsRecords.elems[slot];
        for (uint32_t i = 0; i < nreqs; i++)
          sizes[i] = (int)reqs[i]->send.size;
      }

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
      // For nreqs > 1, advance lastWr to wrs[nreqs] to preserve all data WRs
      struct ibv_send_wr *lastWr = comm->wrs + nreqs - 1;
      if (nreqs > 1) {
        lastWr++;
        memset(lastWr, 0, sizeof(struct ibv_send_wr));
        // Write remote sizes Fifo
        lastWr->wr.rdma.remote_addr =
            comm->remCmplsRecords.addr + (uint64_t)slot *
                                             (uint64_t)MPIB_NET_IB_MAX_RECVS *
                                             (uint64_t)sizeof(int);
        // num_sge will be set per-QP below
      }
      lastWr->wr_id = wr_id;
      lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
      lastWr->imm_data = htobe32(immData);
      lastWr->next = NULL;
      lastWr->send_flags = IBV_SEND_SIGNALED;

      // 128B alignment for device-level split (LL/LL128 protocol compatibility)
      const size_t align = 128;

      // Per-request sizes for each device
      size_t sizeSout[MPIB_NET_IB_MAX_RECVS];
      size_t sizeSup[MPIB_NET_IB_MAX_RECVS];

      for (uint32_t i = 0; i < nreqs; i++) {
        const size_t reqSize = reqs[i]->send.size;
        // Device split: SOUT=0%, SUP=100%
        // Keep SOUT aligned to 128B.
        size_t sout = 0;
        if (sout < reqSize) {
          sout = (sout / align) * align;
        }
        sizeSout[i] = sout;
        sizeSup[i] = reqSize - sizeSout[i];
      }

      // Post WRs: one QP per device (MPIB device-based split, not QP striping)
      for (uint32_t i = 0; i < nqps; i++) {
        mpibQp *qpPtr;
        uint32_t qpIdx;
        NCCLCHECK(mpibCommBaseGetQpForRequest(&comm->base, comm->base.fifoHead,
                                              i, &qpPtr, &qpIdx));
        const int devIndex = qpPtr->devIndex; // 0 = SOUT, 1 = SUP

        // Set up all data WRs (wrs[0..nreqs-1])
        for (uint32_t j = 0; j < nreqs; j++) {
          // Select proper rkey (needed even for 0-size send)
          comm->wrs[j].wr.rdma.rkey = slots[j].rkeys[qpPtr->remDevIdx];

          // This device's portion: offset and length
          const size_t devBaseOffset = (devIndex == 0) ? 0 : sizeSout[j];
          const size_t length = (devIndex == 0) ? sizeSout[j] : sizeSup[j];

          if (length <= 0) {
            comm->wrs[j].sg_list = NULL;
            comm->wrs[j].num_sge = 0;
          } else {
            // Select proper lkey and set up sge
            comm->sges[j].lkey = reqs[j]->send.lkeys[devIndex];
            comm->sges[j].length = length;
            comm->sges[j].addr = (uintptr_t)reqs[j]->send.data + devBaseOffset;
            comm->wrs[j].wr.rdma.remote_addr = slots[j].addr + devBaseOffset;
            comm->wrs[j].sg_list = comm->sges + j;
            comm->wrs[j].num_sge = 1;
          }
        }

        if (nreqs > 1) {
          // Populating the correct gather information based on the device and
          // slot used. (Following net_ib pattern)
          lastWr->sg_list = &(comm->devs[devIndex].sge);
          lastWr->sg_list[0].addr =
              (uint64_t)(comm->remCmplsRecords.elems[slot]);
          lastWr->sg_list[0].length = nreqs * sizeof(int);
          lastWr->num_sge = 1;
          // Populate the correct RKey based on the device used
          lastWr->wr.rdma.rkey = comm->remCmplsRecords.rkeys[devIndex];
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

  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = req - comm->base.reqs;
  wr.sg_list = NULL;
  wr.num_sge = 0;

  TIME_START(1);
  const int nqps = mpibCommBaseGetNqpsPerRequest(&comm->base);
  uint32_t qpIndex = 0;
  mpibQp *qp = NULL;
  struct ibv_recv_wr *bad_wr;
  for (int i = 0; i < nqps; i++) {
    NCCLCHECK(mpibCommBaseGetQpForRequest(&comm->base, comm->base.fifoHead, i,
                                          &qp, &qpIndex));
    mpibAddEvent(req, qp->devIndex);
    NCCLCHECK(wrap_ibv_post_recv(qp->qp, &wr, &bad_wr));
  }
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
  return (request->events[0] == 0 && request->events[1] == 0);
}

static inline ncclResult_t mpibRequestComplete(struct mpibRequest *r, int *done,
                                               int *sizes) {
  TRACE(NCCL_NET, "r=%p done", r);
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

static inline ncclResult_t
mpibCompletionEventProcess(struct mpibNetCommBase *commBase, struct ibv_wc *wc,
                           int devIndex) {
  // Note: for SEND, we pack up to MPIB_NET_IB_MAX_RECVS request indices into
  // wc->wr_id (one byte per request). For RECV/other, wr_id is a single index.
  const uint64_t wr_id = wc->wr_id;

  const uint32_t reqIndex0 = (uint32_t)(wr_id & 0xff);
  if (reqIndex0 >= NET_IB_MAX_REQUESTS)
    return ncclInternalError;
  struct mpibRequest *req0 = commBase->reqs + reqIndex0;

  // Packed SEND completion: decrement each referenced request once for this
  // dev.
  if (req0->type == MPIB_NET_IB_REQ_SEND && req0->nreqs > 1) {
    if (req0->nreqs == 0 || req0->nreqs > MPIB_NET_IB_MAX_RECVS)
      return ncclInternalError;
    for (int j = 0; j < req0->nreqs; j++) {
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

  // Single-request completion (always valid for RECV; also used for SEND with
  // nreqs==1). For RDMA_WRITE_WITH_IMM, the receiver gets
  // IBV_WC_RECV_RDMA_WITH_IMM.
  if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
    if (req0->type != MPIB_NET_IB_REQ_RECV)
      return ncclInternalError;
    if (req0->nreqs == 1)
      req0->recv.sizes[0] = (int)be32toh(wc->imm_data);
  }

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
      if (r->events[i] == 0)
        continue;

      wrDone = 0;
      NCCLCHECK(wrap_ibv_poll_cq(r->devBases[i]->cq, 4, wcs, &wrDone));
      if (wrDone > 0) {
        totalWrDone += wrDone;
        for (int j = 0; j < wrDone; j++) {
          if (wcs[j].status != IBV_WC_SUCCESS)
            return ncclSystemError;
          NCCLCHECK(mpibCompletionEventProcess(r->base, wcs + j, i));
        }
      }
    }
  } while (totalWrDone > 0);

  return ncclSuccess;
}

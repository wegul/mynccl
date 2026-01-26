#include "mpib_p2p.h"
#include "mpib_common.h"
#include <algorithm>

MPIB_PARAM(IbArThreshold, "IB_AR_THRESHOLD", 8192);

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

ncclResult_t mpibMultiSend(struct mpibSendComm *comm, int slot) {
  struct mpibRequest **reqs = comm->fifoReqs[slot];
  volatile struct mpibSendFifo *slots = comm->ctsFifo[slot];
  int nreqs = slots[0].nreqs;
  if (nreqs > MPIB_NET_IB_MAX_RECVS)
    return ncclInternalError;

  uint64_t wr_id = 0ULL;
  for (int r = 0; r < nreqs; r++) {
    struct ibv_send_wr *wr = comm->wrs + r;
    memset(wr, 0, sizeof(struct ibv_send_wr));

    struct ibv_sge *sge = comm->sges + r;
    sge->addr = (uintptr_t)reqs[r]->send.data;
    wr->opcode = IBV_WR_RDMA_WRITE;
    wr->send_flags = 0;
    wr->wr.rdma.remote_addr = slots[r].addr;
    wr->next = wr + 1;
    wr_id += (reqs[r] - comm->base.reqs) << (r * 8);
  }

  uint32_t immData = reqs[0]->send.size;
  if (nreqs > 1) {
    int *sizes = comm->remCmplsRecords.elems[slot];
    for (int r = 0; r < nreqs; r++)
      sizes[r] = reqs[r]->send.size;
  }

  struct ibv_send_wr *lastWr = comm->wrs + nreqs - 1;
  if (nreqs > 1 ||
      (comm->ar && reqs[0]->send.size > mpibParamIbArThreshold())) {
    lastWr++;
    memset(lastWr, 0, sizeof(struct ibv_send_wr));
    if (nreqs > 1) {
      lastWr->wr.rdma.remote_addr = comm->remCmplsRecords.addr +
                                    slot * MPIB_NET_IB_MAX_RECVS * sizeof(int);
      lastWr->num_sge = 1;
    }
  }
  lastWr->wr_id = wr_id;
  lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  lastWr->imm_data = htobe32(immData);
  lastWr->next = NULL;
  lastWr->send_flags = IBV_SEND_SIGNALED;

  const int align = 128;
  int nqps = mpibCommBaseGetNqpsPerRequest(&comm->base);
  int qpIndex = -1;
  mpibQp *qp = NULL;
  for (int i = 0; i < nqps; i++) {
    NCCLCHECK(mpibCommBaseGetQpForRequest(&comm->base, comm->base.fifoHead, i,
                                          &qp, &qpIndex));
    int devIndex = qp->devIndex;
    for (int r = 0; r < nreqs; r++) {
      comm->wrs[r].wr.rdma.rkey = slots[r].rkeys[qp->remDevIdx];
      int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, nqps), align) * align;
      int length =
          std::min(reqs[r]->send.size - reqs[r]->send.offset, chunkSize);
      if (length <= 0) {
        comm->wrs[r].sg_list = NULL;
        comm->wrs[r].num_sge = 0;
      } else {
        comm->sges[r].lkey = reqs[r]->send.lkeys[devIndex];
        comm->sges[r].length = length;
        comm->wrs[r].sg_list = comm->sges + r;
        comm->wrs[r].num_sge = 1;
      }
    }

    if (nreqs > 1) {
      lastWr->sg_list = &(comm->devs[devIndex].sge);
      lastWr->sg_list[0].addr = (uint64_t)(comm->remCmplsRecords.elems[slot]);
      lastWr->sg_list[0].length = nreqs * sizeof(int);
      lastWr->wr.rdma.rkey = comm->remCmplsRecords.rkeys[devIndex];
    }

    struct ibv_send_wr *bad_wr;
    NCCLCHECK(wrap_ibv_post_send(qp->qp, comm->wrs, &bad_wr));

    for (int r = 0; r < nreqs; r++) {
      int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, nqps), align) * align;
      reqs[r]->send.offset += chunkSize;
      comm->sges[r].addr += chunkSize;
      comm->wrs[r].wr.rdma.remote_addr += chunkSize;
    }
  }

  return ncclSuccess;
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

  int nreqs = 0;
  volatile struct mpibSendFifo *slots;

  int slot = comm->base.fifoHead % NET_IB_MAX_REQUESTS;
  struct mpibRequest **reqs = comm->fifoReqs[slot];
  slots = comm->ctsFifo[slot];
  uint64_t idx = comm->base.fifoHead + 1;
  if (slots[0].idx != idx) {
    *request = NULL;
    return ncclSuccess;
  }
  nreqs = slots[0].nreqs;
  for (int r = 1; r < nreqs; r++) {
    if (slots[r].idx != idx) {
      *request = NULL;
      return ncclSuccess;
    }
  }
  std::atomic_thread_fence(std::memory_order_seq_cst);
  for (int r = 0; r < nreqs; r++) {
    if (reqs[r] != NULL || slots[r].tag != tag)
      continue;

    if (size > slots[r].size)
      size = slots[r].size;
    if (slots[r].size < 0 || slots[r].addr == 0 || slots[r].rkeys[0] == 0) {
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

    int nqps = mpibCommBaseGetNqpsPerRequest(&comm->base);
    int qpIndex = -1;
    mpibQp *qp = NULL;
    for (int i = 0; i < nqps; i++) {
      NCCLCHECK(mpibCommBaseGetQpForRequest(&comm->base, comm->base.fifoHead, i,
                                            &qp, &qpIndex));
      mpibAddEvent(req, qp->devIndex);
    }

    for (int i = 0; i < comm->base.vProps.ndevs; i++)
      req->send.lkeys[i] = mhandleWrapper ? mhandleWrapper->mrs[i]->lkey : 0;

    *request = reqs[r] = req;

    for (int r2 = 0; r2 < nreqs; r2++)
      if (reqs[r2] == NULL)
        return ncclSuccess;

    TIME_START(0);
    NCCLCHECK(mpibMultiSend(comm, slot));

    memset((void *)slots, 0, sizeof(struct mpibSendFifo));
    memset(reqs, 0, MPIB_NET_IB_MAX_RECVS * sizeof(struct mpibRequest *));
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

  int slot = comm->base.fifoHead % NET_IB_MAX_REQUESTS;
  req->recv.sizes = comm->cmplsRecords[slot];
  for (int i = 0; i < n; i++)
    req->recv.sizes[i] = 0;
  struct mpibSendFifo *localElem = comm->remCtsFifo.elems[slot];

  mpibQp *ctsQp = NULL;
  ;
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

  if (slot == ctsQp->devIndex) {
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

  struct ibv_recv_wr *bad_wr;
  int qpIndex = -1;
  mpibQp *qp = NULL;
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
  return (request->events[0] == 0 && request->events[1] == 0 &&
          request->events[2] == 0 && request->events[3] == 0);
}

static inline ncclResult_t mpibRequestComplete(struct mpibRequest *r, int *done,
                                               int *sizes) {
  TRACE(NCCL_NET, "r=%p done", r);
  *done = 1;
  if (sizes && r->type == MPIB_NET_IB_REQ_RECV) {
    for (int i = 0; i < r->nreqs; i++)
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
    if (req0->nreqs <= 0 || req0->nreqs > MPIB_NET_IB_MAX_RECVS)
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
    for (int i = 0; i < MPIB_IB_MAX_DEVS_PER_NIC; i++) {
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

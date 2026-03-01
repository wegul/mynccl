#include "mpib_agent_client.h"
#include "mpib_common.h"
#include <algorithm>
#include <cstdint>
#include <limits.h>

static uint32_t mpibGidToIpv4(const union ibv_gid *gid, int *isV4);

MPIB_PARAM(IbGidIndex, "IB_GID_INDEX", 3);
MPIB_PARAM(IbTimeout, "IB_TIMEOUT", 20);
MPIB_PARAM(IbRetryCnt, "IB_RETRY_CNT", 7);
MPIB_PARAM(IbPkey, "IB_PKEY", 0);
MPIB_PARAM(IbUseInline, "IB_USE_INLINE", 0);
MPIB_PARAM(IbSl, "IB_SL", -1);
MPIB_PARAM(IbTc, "IB_TC", -1);
MPIB_PARAM(IbFifoTc, "IB_FIFO_TC", -1);
MPIB_PARAM(IbEceEnable, "IB_ECE_ENABLE", 1);
MPIB_PARAM(SoutQp, "SOUT_QP", 2);
MPIB_PARAM(SupQp, "SUP_QP", 4);
MPIB_PARAM(IslandPrefixLen, "ISLAND_PREFIX_LEN", 24);

// Returns 1 if src and dst are in the same island (same SOUT subnet)
static int mpibIsSameIsland(uint32_t sout_src_ip, uint32_t sout_dst_ip) {
  int prefixLen = (int)mpibParamIslandPrefixLen();
  if (prefixLen <= 0 || prefixLen > 32) {
    prefixLen = 24;
    WARN("NET/MPIB : Invalid ISLAND_PREFIX_LEN=%d, using default 24",
         prefixLen);
  }
  uint32_t mask =
      (prefixLen == 32) ? UINT32_MAX : ~((1u << (32 - prefixLen)) - 1);
  return (sout_src_ip & mask) == (sout_dst_ip & mask);
}

/* Agent registration conn_id counter (thread-safe) */
static std::atomic<uint16_t> g_mpib_conn_counter{0};

/* Include shared interface for MPIB_MAKE_CONN_ID */
#include "../include/mpib_agent_iface.h"

struct mpibQpInfo {
  uint32_t qpn;
  struct ibv_ece ece;
  int ece_supported;
  int devIndex;
};

struct mpibConnectionMetadata {
  struct mpibQpInfo qpInfo[MPIB_IB_MAX_QPS];
  struct mpibDevInfo devs[MPIB_MAX_DEVS];
  char devName[MAX_MERGED_DEV_NAME];
  uint64_t addr;
  int ndevs;
  uint32_t nqpsSout;
  uint32_t nqpsSup;
  int tc;
  int sl;
};

enum mpibCommState {
  mpibCommStateStart = 0,
  mpibCommStateConnect = 1,
  mpibCommStateAccept = 3,
  mpibCommStateSend = 4,
  mpibCommStateRecv = 5,
  mpibCommStateConnecting = 6,
  mpibCommStateConnected = 7,
  mpibCommStatePendingReady = 8,
  mpibCommStateSendDevList = 9,
  mpibCommStateRecvDevList = 10,
};

static const char *mpibCommStateStr(enum mpibCommState s) {
  switch (s) {
  case mpibCommStateStart:
    return "Start";
  case mpibCommStateConnect:
    return "Connect";
  case mpibCommStateAccept:
    return "Accept";
  case mpibCommStateSend:
    return "SendMeta";
  case mpibCommStateRecv:
    return "RecvMeta";
  case mpibCommStateConnecting:
    return "Connecting";
  case mpibCommStateConnected:
    return "Connected";
  case mpibCommStatePendingReady:
    return "PendingReady";
  case mpibCommStateSendDevList:
    return "SendDevList";
  case mpibCommStateRecvDevList:
    return "RecvDevList";
  default:
    return "Unknown";
  }
}

struct mpibCommStage {
  enum mpibCommState state;
  int offset;
  void *buffer;
  void *comm;
};

struct mpibHandle {
  union mpibSocketAddress connectAddr;
  uint64_t magic;
  struct mpibCommStage stage;
};

ncclResult_t mpibInitCommDevBase(int ibDevN, struct mpibNetCommDevBase *base,
                                 void *cq_context) {
  base->ibDevN = ibDevN;
  // Initialize SRQ fields (will be created later for recv comms)
  base->srq = NULL;
  base->srqPosted = 0;

  mpibDev *ibDev = mpibDevs + ibDevN;
  {
    std::lock_guard<std::mutex> lock(ibDev->mutex);
    if (0 == ibDev->pdRefs++) {
      NCCLCHECK(wrap_ibv_alloc_pd(&ibDev->pd, ibDev->context));
    }
    base->pd = ibDev->pd;
  }

  NCCLCHECK(wrap_ibv_create_cq(&base->cq, ibDev->context,
                               2 * NET_IB_MAX_REQUESTS *
                                   (mpibParamSoutQp() + mpibParamSupQp()),
                               cq_context, NULL, 0));

  return ncclSuccess;
}

ncclResult_t mpibDestroyBase(struct mpibNetCommDevBase *base) {
  // Destroy SRQ first (if present)
  if (base->srq != NULL) {
    NCCLCHECK(wrap_ibv_destroy_srq(base->srq));
    base->srq = NULL;
  }
  NCCLCHECK(wrap_ibv_destroy_cq(base->cq));
  std::lock_guard<std::mutex> lock(mpibDevs[base->ibDevN].mutex);
  if (0 == --mpibDevs[base->ibDevN].pdRefs) {
    NCCLCHECK(wrap_ibv_dealloc_pd(mpibDevs[base->ibDevN].pd));
    mpibDevs[base->ibDevN].pd = NULL;
  }
  return ncclSuccess;
}

// Create SRQ for recv comms. Must be called after mpibInitCommDevBase.
static ncclResult_t mpibCreateSrqForRecvBase(struct mpibNetCommDevBase *base) {
  struct ibv_srq_init_attr srqAttr;
  memset(&srqAttr, 0, sizeof(srqAttr));
  srqAttr.attr.max_wr = MPIB_SRQ_HIGH_WATER;
  srqAttr.attr.max_sge = 1; // Even though we post num_sge=0
  NCCLCHECK(wrap_ibv_create_srq(&base->srq, base->pd, &srqAttr));
  base->srqPosted = 0;
  INFO(NCCL_NET, "NET/MPIB : Created SRQ %p for dev %d (highWater=%d)",
       base->srq, base->ibDevN, MPIB_SRQ_HIGH_WATER);
  return ncclSuccess;
}

ncclResult_t mpibGetGidIndex(struct ibv_context *context, uint8_t portNum,
                             struct ibv_port_attr *portAttr, int *gidIndex) {
  (void)context;
  (void)portNum;
  const int gidTblLen = portAttr->gid_tbl_len;
  const int idx = (int)mpibParamIbGidIndex();
  if (idx < 0 || idx >= gidTblLen) {
    WARN("NET/MPIB : Invalid IB_GID_INDEX=%d (gid_tbl_len=%d)", idx, gidTblLen);
    return ncclInvalidUsage;
  }
  *gidIndex = idx;
  return ncclSuccess;
}

ncclResult_t mpibCreateQp(uint8_t ib_port, struct mpibNetCommDevBase *base,
                          int access_flags, void *qp_context,
                          struct mpibQp *qp) {
  struct ibv_qp_init_attr qpInitAttr;
  memset(&qpInitAttr, 0, sizeof(qpInitAttr));
  qpInitAttr.qp_context = qp_context;
  qpInitAttr.send_cq = base->cq;
  qpInitAttr.recv_cq = base->cq;
  qpInitAttr.qp_type = IBV_QPT_RC;
  qpInitAttr.cap.max_send_wr = 2 * NET_IB_MAX_REQUESTS;
  // SRQ: if SRQ is present, use it and set max_recv_wr to 0
  if (base->srq != NULL) {
    qpInitAttr.srq = base->srq;
    qpInitAttr.cap.max_recv_wr = 0;
  } else {
    qpInitAttr.cap.max_recv_wr = NET_IB_MAX_REQUESTS;
  }
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.cap.max_inline_data =
      mpibParamIbUseInline() ? sizeof(struct mpibSendFifo) : 0;
  NCCLCHECK(wrap_ibv_create_qp(&qp->qp, base->pd, &qpInitAttr));

  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(qpAttr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = mpibParamIbPkey();
  qpAttr.port_num = ib_port;
  qpAttr.qp_access_flags = access_flags;
  NCCLCHECK(wrap_ibv_modify_qp(qp->qp, &qpAttr,
                               IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                                   IBV_QP_ACCESS_FLAGS));

  TRACE(NCCL_NET,
        "NET/MPIB : mpibCreateQp port=%d dev=%d devName=%s qpn=%u pkey=%u "
        "pd=%p srq=%p",
        ib_port, base->ibDevN, mpibDevs[base->ibDevN].devName, qp->qp->qp_num,
        qpAttr.pkey_index, base->pd, base->srq);
  return ncclSuccess;
}

ncclResult_t mpibRtrQp(struct ibv_qp *qp, struct mpibGidInfo *sGidInfo,
                       uint32_t dest_qp_num, struct mpibDevInfo *info,
                       bool fifoTc, int tc, int sl) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(qpAttr));
  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = info->mtu;
  qpAttr.dest_qp_num = dest_qp_num;
  qpAttr.rq_psn = 0;
  qpAttr.max_dest_rd_atomic = 1;
  qpAttr.min_rnr_timer = 12;
  if (info->link_layer == IBV_LINK_LAYER_ETHERNET) {
    qpAttr.ah_attr.is_global = 1;
    qpAttr.ah_attr.grh.hop_limit = 255;
    qpAttr.ah_attr.grh.dgid = info->gid;
    qpAttr.ah_attr.grh.sgid_index = sGidInfo->localGidIndex;
    if (fifoTc) {
      qpAttr.ah_attr.grh.traffic_class = tc;
    }
  } else {
    qpAttr.ah_attr.is_global = 0;
    qpAttr.ah_attr.dlid = info->lid;
  }
  qpAttr.ah_attr.sl = sl;
  qpAttr.ah_attr.src_path_bits = 0;
  qpAttr.ah_attr.port_num = info->ib_port;
  NCCLCHECK(wrap_ibv_modify_qp(
      qp, &qpAttr,
      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
          IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER));
  return ncclSuccess;
}

ncclResult_t mpibRtsQp(struct ibv_qp *qp) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(qpAttr));
  qpAttr.qp_state = IBV_QPS_RTS;
  qpAttr.timeout = mpibParamIbTimeout();
  qpAttr.retry_cnt = mpibParamIbRetryCnt();
  qpAttr.rnr_retry = 7;
  qpAttr.sq_psn = 0;
  qpAttr.max_rd_atomic = 1;
  NCCLCHECK(wrap_ibv_modify_qp(qp, &qpAttr,
                               IBV_QP_STATE | IBV_QP_TIMEOUT |
                                   IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                   IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC));
  return ncclSuccess;
}

__hidden ncclResult_t mpibListen(void *ctx, int dev, void *opaqueHandle,
                                 void **listenComm) {
  ncclResult_t ret = ncclSuccess;
  struct mpibListenComm *comm;
  NCCLCHECK(mpibCalloc(&comm, 1));
  struct mpibHandle *handle = (struct mpibHandle *)opaqueHandle;
  union mpibSocketAddress listenAddr;
  static_assert(sizeof(struct mpibHandle) < NCCL_NET_HANDLE_MAXSIZE,
                "mpibHandle size too large");
  memset(handle, 0, sizeof(struct mpibHandle));
  comm->dev = dev;
  handle->magic = MPIB_SOCKET_MAGIC;
  NCCLCHECKGOTO(mpibSocketInit(&comm->sock, &mpibIfAddr, handle->magic,
                               mpibSocketTypeNetIb, NULL, 1),
                ret, fail);
  NCCLCHECKGOTO(mpibSocketListen(&comm->sock), ret, fail);
  NCCLCHECKGOTO(mpibSocketGetAddr(&comm->sock, &listenAddr), ret, fail);
  memcpy(&handle->connectAddr, &listenAddr, sizeof(handle->connectAddr));

  *listenComm = comm;
exit:
  return ret;
fail:
  mpibSocketClose(&comm->sock);
  free(comm);
  goto exit;
}

#define MPIB_IB_SL_DEFAULT 0
#define MPIB_IB_TC_DEFAULT 0

static ncclResult_t mpibSenderQpsCreate(mpibSendComm *comm,
                                        struct mpibConnectionMetadata *meta) {
  const uint32_t nqps = comm->base.nqps;
  const uint32_t nqpsSout = comm->base.nqpsSout;
  for (uint32_t qpIndex = 0; qpIndex < nqps; qpIndex++) {
    // Contiguous layout: [SOUT_0..SOUT_{n-1}, SUP_0..SUP_{m-1}]
    int devIndex = (qpIndex < nqpsSout) ? 0 : 1;
    mpibQp *qp = comm->base.qps + qpIndex;
    qp->devIndex = devIndex;
    NCCLCHECK(mpibCreateQp(mpibDevs[comm->base.vProps.devs[devIndex]].portNum,
                           &comm->devs[devIndex].base, IBV_ACCESS_REMOTE_WRITE,
                           &comm->base.stats, qp));
    mpibQpInfo *qpInfo = meta->qpInfo + qpIndex;
    qpInfo->qpn = qp->qp->qp_num;
    qpInfo->devIndex = qp->devIndex;
    if (mpibParamIbEceEnable()) {
      qpInfo->ece_supported = 0;
      (void)wrap_ibv_query_ece(qp->qp, &qpInfo->ece, &qpInfo->ece_supported);
    }
  }
  return ncclSuccess;
}

static ncclResult_t mpibSenderQpsToRts(mpibSendComm *comm, int dev,
                                       struct mpibConnectionMetadata *remMeta) {
  const uint32_t nqps = comm->base.nqps;
  for (uint32_t qpIndex = 0; qpIndex < nqps; qpIndex++) {
    mpibQp *qp = comm->base.qps + qpIndex;
    int devIndex = qp->devIndex;
    int remDevIndex = remMeta->qpInfo[qpIndex].devIndex;
    qp->remDevIdx = remDevIndex; // MPIB-CUSTOM: Set remDevIdx for sender QPs
    struct mpibDevInfo *devInfo = comm->base.remDevs + remDevIndex;
    if (remMeta->qpInfo[qpIndex].ece_supported && mpibParamIbEceEnable()) {
      int supported = 0;
      NCCLCHECK(
          wrap_ibv_set_ece(qp->qp, &remMeta->qpInfo[qpIndex].ece, &supported));
    }
    NCCLCHECK(mpibRtrQp(qp->qp, &comm->devs[devIndex].base.gidInfo,
                        remMeta->qpInfo[qpIndex].qpn, devInfo, 0, remMeta->tc,
                        remMeta->sl));
    NCCLCHECK(mpibRtsQp(qp->qp));
  }
  return ncclSuccess;
}

__hidden ncclResult_t mpibConnect(void *ctx, int dev, void *opaqueHandle,
                                  void **sendComm,
                                  ncclNetDeviceHandle_t **sendDevComm) {
  (void)sendDevComm;
  ncclResult_t ret = ncclSuccess;
  struct mpibHandle *handle = (struct mpibHandle *)opaqueHandle;
  struct mpibCommStage *stage = &handle->stage;
  struct mpibSendComm *comm = (struct mpibSendComm *)stage->comm;
  int ready;
  int link_layer = IBV_LINK_LAYER_UNSPECIFIED;
  *sendComm = NULL;

  if (stage->state == mpibCommStateConnect)
    goto ib_connect_check;
  if (stage->state == mpibCommStateSendDevList)
    goto ib_send_dev_list;
  if (stage->state == mpibCommStateRecvDevList)
    goto ib_recv_dev_list;
  if (stage->state == mpibCommStateSend)
    goto ib_send;
  if (stage->state == mpibCommStateConnecting)
    goto ib_connect;
  if (stage->state == mpibCommStateConnected)
    goto ib_send_ready;
  if (stage->state != mpibCommStateStart) {
    return ncclInternalError;
  }

  stage->buffer = NULL;
  if (stage->state == mpibCommStateStart) {
    NCCLCHECK(mpibMalloc((void **)&comm, sizeof(struct mpibSendComm)));
    NCCLCHECKGOTO(mpibStatsInit(&comm->base.stats), ret, fail);
    NCCLCHECKGOTO(mpibSocketInit(&comm->base.sock, &handle->connectAddr,
                                 handle->magic, mpibSocketTypeNetIb, NULL, 1),
                  ret, fail);
    stage->comm = comm;
    stage->state = mpibCommStateConnect;
    NCCLCHECKGOTO(mpibSocketConnect(&comm->base.sock), ret, fail);
  }

ib_connect_check:
  NCCLCHECKGOTO(mpibSocketReady(&comm->base.sock, &ready), ret, fail);
  if (!ready) {
    *sendComm = NULL;
    return ncclSuccess;
  }

  struct mpibMergedDev *mergedDev;
  if (dev >= mpibNMergedIbDevs) {
    ret = ncclInvalidUsage;
    goto fail;
  }

  mergedDev = mpibMergedDevs + dev;
  comm->base.vProps = mergedDev->vProps;
  comm->base.isSend = true;
  stage->state = mpibCommStateSendDevList;
  stage->offset = 0;
  struct mpibConnectionMetadata meta;
  NCCLCHECKGOTO(mpibMalloc((void **)&stage->buffer, sizeof(meta)), ret, fail);
  memcpy(stage->buffer, &mergedDev->vProps, sizeof(ncclNetVDeviceProps_t));

ib_send_dev_list:
  NCCLCHECK(mpibSocketProgress(MPIB_SOCKET_SEND, &comm->base.sock,
                               stage->buffer, sizeof(ncclNetVDeviceProps_t),
                               &stage->offset));
  if (stage->offset != (int)sizeof(ncclNetVDeviceProps_t)) {
    *sendComm = NULL;
    return ncclSuccess;
  }

  stage->state = mpibCommStateRecvDevList;
  stage->offset = 0;

ib_recv_dev_list:
  NCCLCHECK(mpibSocketProgress(MPIB_SOCKET_RECV, &comm->base.sock,
                               stage->buffer, sizeof(ncclNetVDeviceProps_t),
                               &stage->offset));
  if (stage->offset != (int)sizeof(ncclNetVDeviceProps_t)) {
    *sendComm = NULL;
    return ncclSuccess;
  }
  stage->offset = 0;
  ncclNetVDeviceProps_t remoteVProps;
  ncclNetCommConfig_t *config;
  memcpy(&remoteVProps, stage->buffer, sizeof(ncclNetVDeviceProps_t));
  mergedDev = mpibMergedDevs + dev;
  comm->base.vProps = mergedDev->vProps;

  // Set nqpsSout/nqpsSup from params directly (contiguous layout)
  comm->base.nqpsSout = (uint32_t)mpibParamSoutQp();
  comm->base.nqpsSup = (uint32_t)mpibParamSupQp();
  comm->base.nqps = comm->base.nqpsSout + comm->base.nqpsSup;

  // Init PD, Ctx for each IB device
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    int ibDevN = comm->base.vProps.devs[i];
    NCCLCHECKGOTO(
        mpibInitCommDevBase(ibDevN, &comm->devs[i].base, &comm->base.stats),
        ret, fail);
  }

  memset(&meta, 0, sizeof(meta));
  meta.ndevs = comm->base.vProps.ndevs;

  NCCLCHECKGOTO(mpibSenderQpsCreate(comm, &meta), ret, fail);

  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    mpibSendCommDev *commDev = comm->devs + i;
    mpibDev *ibDev = mpibDevs + commDev->base.ibDevN;
    struct mpibDevInfo *devInfo = meta.devs + i;

    devInfo->ib_port = ibDev->portNum;
    devInfo->mtu = ibDev->portAttr.active_mtu;
    devInfo->lid = ibDev->portAttr.lid;

    NCCLCHECKGOTO(wrap_ibv_reg_mr(&commDev->putSignalScratchpadMr,
                                  commDev->base.pd, &comm->putSignalScratchpad,
                                  sizeof(comm->putSignalScratchpad),
                                  IBV_ACCESS_LOCAL_WRITE),
                  ret, fail);

    NCCLCHECKGOTO(wrap_ibv_reg_mr(&commDev->ctsFifoMr, commDev->base.pd,
                                  comm->ctsFifo, sizeof(comm->ctsFifo),
                                  IBV_ACCESS_LOCAL_WRITE |
                                      IBV_ACCESS_REMOTE_WRITE |
                                      IBV_ACCESS_REMOTE_READ),
                  ret, fail);
    devInfo->rkey = commDev->ctsFifoMr->rkey;

    devInfo->link_layer = commDev->base.gidInfo.link_layer =
        ibDev->portAttr.link_layer;
    NCCLCHECKGOTO(mpibGetGidIndex(ibDev->context, ibDev->portNum,
                                  &ibDev->portAttr,
                                  &commDev->base.gidInfo.localGidIndex),
                  ret, fail);
    NCCLCHECKGOTO(wrap_ibv_query_gid(ibDev->context, ibDev->portNum,
                                     commDev->base.gidInfo.localGidIndex,
                                     &commDev->base.gidInfo.localGid),
                  ret, fail);
    devInfo->gid.global.subnet_prefix =
        commDev->base.gidInfo.localGid.global.subnet_prefix;
    devInfo->gid.global.interface_id =
        commDev->base.gidInfo.localGid.global.interface_id;

    if (link_layer == IBV_LINK_LAYER_UNSPECIFIED)
      link_layer = devInfo->link_layer;
    if (link_layer != devInfo->link_layer) {
      int ibDev0 = comm->devs[0].base.ibDevN;
      WARN("NET/MPIB : Attempted to connect incompatible devices: [%d]%s:%d/%s "
           "and [%d]%s:%d/%s. Try selecting NICs of only one link type using "
           "NCCL_IB_HCA",
           commDev->base.ibDevN, ibDev->devName, ibDev->portNum,
           MPIB_IB_LLSTR(ibDev->portAttr.link_layer), ibDev0,
           mpibDevs[ibDev0].devName, mpibDevs[ibDev0].portNum,
           MPIB_IB_LLSTR(link_layer));
      return ncclInternalError;
    }
  }
  config = (ncclNetCommConfig_t *)ctx;
  meta.addr = (uint64_t)comm->ctsFifo;
  meta.nqpsSout = comm->base.nqpsSout;
  meta.nqpsSup = comm->base.nqpsSup;
  meta.sl = (mpibParamIbSl() != -1) ? mpibParamIbSl()
            : (config && config->trafficClass != NCCL_NET_TRAFFIC_CLASS_UNDEF)
                ? config->trafficClass
                : MPIB_IB_SL_DEFAULT;
  meta.tc = (mpibParamIbTc() != -1) ? mpibParamIbTc()
            : (config && config->trafficClass != NCCL_NET_TRAFFIC_CLASS_UNDEF)
                ? config->trafficClass
                : MPIB_IB_TC_DEFAULT;
  strncpy(meta.devName, mergedDev->devName, MAX_MERGED_DEV_NAME);

  {
    char line[MPIB_SOCKET_NAME_MAXLEN + 1];
    union mpibSocketAddress addr;
    mpibSocketGetAddr(&comm->base.sock, &addr);
    INFO(NCCL_NET,
         "NET/MPIB: mpibConnect advertise CTS peer=%s ctsFifoAddr=0x%llx "
         "ndevs=%d rkey[0]=0x%x",
         mpibSocketToString(&addr, line), (unsigned long long)meta.addr,
         meta.ndevs, (unsigned)meta.devs[0].rkey);
  }

  stage->state = mpibCommStateSend;
  stage->offset = 0;
  memcpy(stage->buffer, &meta, sizeof(meta));

ib_send:
  NCCLCHECKGOTO(mpibSocketProgress(MPIB_SOCKET_SEND, &comm->base.sock,
                                   stage->buffer, sizeof(meta), &stage->offset),
                ret, fail);
  if (stage->offset != (int)sizeof(meta)) {
    *sendComm = NULL;
    return ncclSuccess;
  }

  stage->state = mpibCommStateConnecting;
  stage->offset = 0;
  memset(stage->buffer, 0, sizeof(meta));

ib_connect:
  struct mpibConnectionMetadata remMeta;
  NCCLCHECKGOTO(
      mpibSocketProgress(MPIB_SOCKET_RECV, &comm->base.sock, stage->buffer,
                         sizeof(mpibConnectionMetadata), &stage->offset),
      ret, fail);
  if (stage->offset != (int)sizeof(remMeta)) {
    *sendComm = NULL;
    return ncclSuccess;
  }

  memcpy(&remMeta, stage->buffer, sizeof(mpibConnectionMetadata));

  // Validate peer QP counts match
  if (remMeta.nqpsSout != comm->base.nqpsSout ||
      remMeta.nqpsSup != comm->base.nqpsSup) {
    WARN("NET/MPIB : QP count mismatch: local nqpsSout=%u nqpsSup=%u, "
         "remote nqpsSout=%u nqpsSup=%u",
         comm->base.nqpsSout, comm->base.nqpsSup, remMeta.nqpsSout,
         remMeta.nqpsSup);
    ret = ncclInvalidUsage;
    goto fail;
  }

  comm->base.nRemDevs = remMeta.ndevs;

  {
    char line[MPIB_SOCKET_NAME_MAXLEN + 1];
    union mpibSocketAddress addr;
    mpibSocketGetAddr(&comm->base.sock, &addr);
    INFO(NCCL_NET,
         "NET/MPIB: mpibConnect got remote meta peer=%s remoteAddr=0x%llx "
         "remoteNdevs=%d remoteRkey[0]=0x%x",
         mpibSocketToString(&addr, line), (unsigned long long)remMeta.addr,
         remMeta.ndevs, (unsigned)remMeta.devs[0].rkey);
  }

  if (comm->base.vProps.ndevs > 0) {
    int ibDev0 = comm->devs[0].base.ibDevN;
    link_layer = mpibDevs[ibDev0].portAttr.link_layer;
    for (int i = 0; i < remMeta.ndevs; i++) {
      if (remMeta.devs[i].link_layer != link_layer) {
        WARN("NET/MPIB : Remote %s device is incompatible with the local "
             "[%d]%s:%d/%s. Try selecting NICs of only one link type using "
             "NCCL_IB_HCA",
             MPIB_IB_LLSTR(remMeta.devs[i].link_layer), ibDev0,
             mpibDevs[ibDev0].devName, mpibDevs[ibDev0].portNum,
             MPIB_IB_LLSTR(link_layer));
        return ncclInternalError;
      }
    }
  }

  for (int i = 0; i < remMeta.ndevs; i++) {
    comm->base.remDevs[i] = remMeta.devs[i];
    comm->base.remDevs[i].remoteGid.global.interface_id =
        comm->base.remDevs[i].gid.global.interface_id;
    comm->base.remDevs[i].remoteGid.global.subnet_prefix =
        comm->base.remDevs[i].gid.global.subnet_prefix;
  }

  comm->remCmplsRecords.addr = remMeta.addr;
  for (int i = 0; i < remMeta.ndevs; i++)
    comm->remCmplsRecords.rkeys[i] = remMeta.devs[i].rkey;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    mpibSendCommDev *commDev = comm->devs + i;
    NCCLCHECKGOTO(
        wrap_ibv_reg_mr(&commDev->cmplsRecordsMr, comm->devs[i].base.pd,
                        &comm->remCmplsRecords.elems,
                        sizeof(comm->remCmplsRecords.elems),
                        IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
                            IBV_ACCESS_REMOTE_READ),
        ret, fail);
    comm->devs[i].sge.lkey = comm->devs[i].cmplsRecordsMr->lkey;
  }

  NCCLCHECKGOTO(mpibSenderQpsToRts(comm, dev, &remMeta), ret, fail);

  comm->base.ready = 1;
  stage->state = mpibCommStateConnected;

  stage->offset = 0;

ib_send_ready:
  NCCLCHECKGOTO(mpibSocketProgress(MPIB_SOCKET_SEND, &comm->base.sock,
                                   &comm->base.ready, sizeof(int),
                                   &stage->offset),
                ret, fail);
  if (stage->offset != (int)sizeof(int)) {
    *sendComm = NULL;
    return ncclSuccess;
  }
  stage->state = mpibCommStateConnected;
  {
    char line[MPIB_SOCKET_NAME_MAXLEN + 1];
    union mpibSocketAddress addr;
    mpibSocketGetAddr(&comm->base.sock, &addr);
    INFO(NCCL_NET, "NET/MPIB: mpibConnect DONE peer=%s",
         mpibSocketToString(&addr, line));
  }

  /* Register connection with agent */
  {
    uint16_t counter =
        g_mpib_conn_counter.fetch_add(1, std::memory_order_relaxed);
    comm->conn_id = MPIB_MAKE_CONN_ID(getpid(), counter);

    /* Extract IPv4 from GID (IPv4-mapped IPv6: ::ffff:a.b.c.d) */
    /* SOUT = dev[0], SUP = dev[1] */
    int sout_src_is_v4 = 0, sout_dst_is_v4 = 0;
    int sup_src_is_v4 = 0, sup_dst_is_v4 = 0;
    uint32_t sout_src_ip =
        mpibGidToIpv4(&comm->devs[0].base.gidInfo.localGid, &sout_src_is_v4);
    uint32_t sout_dst_ip =
        mpibGidToIpv4(&comm->base.remDevs[0].gid, &sout_dst_is_v4);
    uint32_t sup_src_ip =
        mpibGidToIpv4(&comm->devs[1].base.gidInfo.localGid, &sup_src_is_v4);
    uint32_t sup_dst_ip =
        mpibGidToIpv4(&comm->base.remDevs[1].gid, &sup_dst_is_v4);
    if (!sout_src_is_v4 || !sout_dst_is_v4 || !sup_src_is_v4 ||
        !sup_dst_is_v4) {
      WARN("NET/MPIB : Non-IPv4 GID detected for agent registration: "
           "sout_src_v4=%d sout_dst_v4=%d sup_src_v4=%d sup_dst_v4=%d",
           sout_src_is_v4, sout_dst_is_v4, sup_src_is_v4, sup_dst_is_v4);
    }

    /* Classify connection path based on SOUT subnet */
    int sameIsland = mpibIsSameIsland(sout_src_ip, sout_dst_ip);
    comm->base.pathSupBw = sameIsland ? UINT32_MAX : 0;
    comm->base.hintActive = sameIsland ? 1 : 0;
    INFO(NCCL_NET,
         "NET/MPIB : Path classification: %s hintActive=%d "
         "(sout_src=0x%08x sout_dst=0x%08x)",
         sameIsland ? "SUP (intra-island)" : "SOUT (inter-island)",
         comm->base.hintActive, sout_src_ip, sout_dst_ip);

    ncclResult_t regRet =
        mpibAgentRegister(comm->conn_id, sout_src_ip, sout_dst_ip, sup_src_ip,
                          sup_dst_ip, &comm->hint_slot);
    if (regRet != ncclSuccess) {
      WARN("NET/MPIB : Agent registration failed for conn_id=0x%08x",
           comm->conn_id);
      ret = regRet;
      goto fail;
    }
    INFO(NCCL_NET, "NET/MPIB : Registered conn_id=0x%08x hint_slot=%u",
         comm->conn_id, comm->hint_slot);
  }

  *sendComm = comm;
  return ncclSuccess;

fail:
  if (stage->buffer)
    free(stage->buffer);
  if (comm) {
    mpibSocketClose(&comm->base.sock);
    free(comm);
  }
  return ret;
}

MPIB_PARAM(IbWarnRailLocal, "IB_WARN_RAIL_LOCAL", 0);

static uint32_t mpibGidToIpv4(const union ibv_gid *gid, int *isV4) {
  const uint8_t *b = gid->raw;
  int v4 = 1;
  for (int i = 0; i < 10; i++) {
    if (b[i] != 0) {
      v4 = 0;
      break;
    }
  }
  if (v4 && (b[10] != 0xff || b[11] != 0xff))
    v4 = 0;
  if (isV4)
    *isV4 = v4;
  if (!v4)
    return 0;
  return ((uint32_t)b[12] << 24) | ((uint32_t)b[13] << 16) |
         ((uint32_t)b[14] << 8) | (uint32_t)b[15];
}

static ncclResult_t mpibCheckVProps(ncclNetVDeviceProps_t *vProps1,
                                    ncclNetVDeviceProps_t *vProps2) {
  if (vProps1->ndevs != vProps2->ndevs)
    return ncclInvalidUsage;
  return ncclSuccess;
}

static ncclResult_t
mpibReceiverQpsCreateToRts(mpibRecvComm *rComm,
                           struct mpibConnectionMetadata *remMeta,
                           struct mpibConnectionMetadata *meta) {
  const uint32_t nqps = rComm->base.nqps;
  const uint32_t nqpsSout = rComm->base.nqpsSout;
  for (uint32_t qpIndex = 0; qpIndex < nqps; qpIndex++) {
    // Contiguous layout: [SOUT_0..SOUT_{n-1}, SUP_0..SUP_{m-1}]
    int devIndex = (qpIndex < nqpsSout) ? 0 : 1;
    mpibRecvCommDev *rCommDev = &rComm->devs[devIndex];
    mpibDev *ibDev = &mpibDevs[rCommDev->base.ibDevN];
    mpibQpInfo *remQpInfo = &remMeta->qpInfo[qpIndex];
    mpibQpInfo *localQpInfo = &meta->qpInfo[qpIndex];
    int remDevIndex = remQpInfo->devIndex;
    mpibDevInfo *remDevInfo = &remMeta->devs[remDevIndex];
    mpibQp *localQp = &rComm->base.qps[qpIndex];

    localQp->remDevIdx = remDevIndex;
    localQp->devIndex = devIndex;

    NCCLCHECK(mpibCreateQp(ibDev->portNum, &rCommDev->base,
                           IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC,
                           &rComm->base.stats, localQp));

    localQpInfo->qpn = localQp->qp->qp_num;
    localQpInfo->devIndex = localQp->devIndex;

    if (remQpInfo->ece_supported) {
      int supported = 0;
      NCCLCHECK(wrap_ibv_set_ece(localQp->qp, &remQpInfo->ece, &supported));
      localQpInfo->ece_supported = supported;
    } else {
      localQpInfo->ece_supported = 0;
    }

    ibDev->portAttr.active_mtu =
        std::min(ibDev->portAttr.active_mtu, remDevInfo->mtu);

    NCCLCHECK(mpibRtrQp(localQp->qp, &rCommDev->base.gidInfo, remQpInfo->qpn,
                        remDevInfo, true, remMeta->tc, remMeta->sl));
    NCCLCHECK(mpibRtsQp(localQp->qp));

    if (remQpInfo->ece_supported && localQpInfo->ece_supported) {
      int supported = 0;
      NCCLCHECK(wrap_ibv_query_ece(localQp->qp, &localQpInfo->ece, &supported));
      localQpInfo->ece_supported = supported;
    }
  }
  return ncclSuccess;
}

__hidden ncclResult_t mpibAccept(void *listenComm, void **recvComm,
                                 ncclNetDeviceHandle_t **recvDevComm) {
  (void)recvDevComm;
  ncclResult_t ret = ncclSuccess;
  struct mpibListenComm *lComm = (struct mpibListenComm *)listenComm;
  struct mpibCommStage *stage = lComm->stage;
  if (stage == NULL) {
    stage = (struct mpibCommStage *)calloc(1, sizeof(*stage));
    lComm->stage = stage;
  }
  struct mpibRecvComm *rComm = (struct mpibRecvComm *)stage->comm;
  int ready;
  int link_layer = IBV_LINK_LAYER_UNSPECIFIED;
  struct mpibMergedDev *mergedDev = NULL;
  *recvComm = NULL;

  if (stage->state == mpibCommStateAccept)
    goto ib_accept_check;
  if (stage->state == mpibCommStateRecvDevList)
    goto ib_recv_dev_list;
  if (stage->state == mpibCommStateSendDevList)
    goto ib_send_dev_list;
  if (stage->state == mpibCommStateRecv)
    goto ib_recv;
  if (stage->state == mpibCommStateSend)
    goto ib_send;
  if (stage->state == mpibCommStatePendingReady)
    goto ib_recv_ready;
  if (stage->state != mpibCommStateStart) {
    return ncclInternalError;
  }

  NCCLCHECK(mpibMalloc((void **)&rComm, sizeof(struct mpibRecvComm)));
  NCCLCHECKGOTO(mpibStatsInit(&rComm->base.stats), ret, fail);
  stage->comm = rComm;
  stage->state = mpibCommStateAccept;
  NCCLCHECKGOTO(mpibSocketInit(&rComm->base.sock), ret, fail);
  NCCLCHECKGOTO(mpibSocketAccept(&rComm->base.sock, &lComm->sock), ret, fail);

  struct mpibConnectionMetadata remMeta;
  stage->offset = 0;
  NCCLCHECK(mpibMalloc((void **)&stage->buffer, sizeof(remMeta)));

ib_accept_check:
  NCCLCHECKGOTO(mpibSocketReady(&rComm->base.sock, &ready), ret, fail);
  if (!ready) {
    return ncclSuccess;
  }
  stage->state = mpibCommStateRecvDevList;
  stage->offset = 0;

ib_recv_dev_list:
  NCCLCHECK(mpibSocketProgress(MPIB_SOCKET_RECV, &rComm->base.sock,
                               stage->buffer, sizeof(ncclNetVDeviceProps_t),
                               &stage->offset));
  if (stage->offset != (int)sizeof(ncclNetVDeviceProps_t)) {
    return ncclSuccess;
  }
  ncclNetVDeviceProps_t remoteVProps;
  memcpy(&remoteVProps, stage->buffer, sizeof(ncclNetVDeviceProps_t));
  if (lComm->dev >= mpibNMergedIbDevs)
    return ncclInternalError;

  mergedDev = mpibMergedDevs + lComm->dev;
  NCCLCHECK(mpibCheckVProps(&mergedDev->vProps, &remoteVProps));
  rComm->base.vProps = mergedDev->vProps;
  memcpy(stage->buffer, &rComm->base.vProps, sizeof(ncclNetVDeviceProps_t));
  rComm->base.isSend = false;

  // Set nqpsSout/nqpsSup from params directly (contiguous layout)
  rComm->base.nqpsSout = (uint32_t)mpibParamSoutQp();
  rComm->base.nqpsSup = (uint32_t)mpibParamSupQp();
  rComm->base.nqps = rComm->base.nqpsSout + rComm->base.nqpsSup;

  stage->offset = 0;
  stage->state = mpibCommStateSendDevList;

ib_send_dev_list:
  NCCLCHECKGOTO(mpibSocketProgress(MPIB_SOCKET_SEND, &rComm->base.sock,
                                   stage->buffer, sizeof(ncclNetVDeviceProps_t),
                                   &stage->offset),
                ret, fail);
  if (stage->offset != (int)sizeof(ncclNetVDeviceProps_t)) {
    return ncclSuccess;
  }

  stage->offset = 0;
  stage->state = mpibCommStateRecv;

ib_recv:
  NCCLCHECKGOTO(mpibSocketProgress(MPIB_SOCKET_RECV, &rComm->base.sock,
                                   stage->buffer, sizeof(remMeta),
                                   &stage->offset),
                ret, fail);
  if (stage->offset != (int)sizeof(remMeta)) {
    return ncclSuccess;
  }

  memcpy(&remMeta, stage->buffer, sizeof(remMeta));

  // Validate peer QP counts match
  if (remMeta.nqpsSout != rComm->base.nqpsSout ||
      remMeta.nqpsSup != rComm->base.nqpsSup) {
    WARN("NET/MPIB : QP count mismatch: local nqpsSout=%u nqpsSup=%u, "
         "remote nqpsSout=%u nqpsSup=%u",
         rComm->base.nqpsSout, rComm->base.nqpsSup, remMeta.nqpsSout,
         remMeta.nqpsSup);
    ret = ncclInvalidUsage;
    goto fail;
  }

  struct mpibDev *ibDev;
  int ibDevN;
  struct mpibRecvCommDev *rCommDev;

  mergedDev = mpibMergedDevs + lComm->dev;
  rComm->base.nRemDevs = remMeta.ndevs;

  struct mpibConnectionMetadata meta;
  memset(&meta, 0, sizeof(meta));
  for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
    rCommDev = rComm->devs + i;
    ibDevN = rComm->base.vProps.devs[i];
    NCCLCHECKGOTO(
        mpibInitCommDevBase(ibDevN, &rCommDev->base, &rComm->base.stats), ret,
        fail);
    // Create SRQ for this recv comm device (mandatory for recv comms)
    NCCLCHECKGOTO(mpibCreateSrqForRecvBase(&rCommDev->base), ret, fail);
    ibDev = mpibDevs + ibDevN;
    NCCLCHECKGOTO(mpibGetGidIndex(ibDev->context, ibDev->portNum,
                                  &ibDev->portAttr,
                                  &rCommDev->base.gidInfo.localGidIndex),
                  ret, fail);
    NCCLCHECKGOTO(wrap_ibv_query_gid(ibDev->context, ibDev->portNum,
                                     rCommDev->base.gidInfo.localGidIndex,
                                     &rCommDev->base.gidInfo.localGid),
                  ret, fail);
    if (link_layer == IBV_LINK_LAYER_UNSPECIFIED)
      link_layer = ibDev->portAttr.link_layer;
    if (link_layer != ibDev->portAttr.link_layer)
      return ncclInternalError;
  }

  // Initialize slotReq map for SRQ-based completion
  memset(rComm->base.slotReq, 0, sizeof(rComm->base.slotReq));

  for (int i = 0; i < remMeta.ndevs; i++) {
    rComm->base.remDevs[i] = remMeta.devs[i];
    rComm->base.remDevs[i].remoteGid.global.interface_id =
        rComm->base.remDevs[i].gid.global.interface_id;
    rComm->base.remDevs[i].remoteGid.global.subnet_prefix =
        rComm->base.remDevs[i].gid.global.subnet_prefix;
    if (remMeta.devs[i].link_layer != link_layer)
      return ncclInternalError;
  }

  NCCLCHECKGOTO(mpibReceiverQpsCreateToRts(rComm, &remMeta, &meta), ret, fail);

  rComm->remCtsFifo.addr = remMeta.addr;
  for (int i = 0; i < remMeta.ndevs; i++)
    rComm->remCtsFifo.rkeys[i] = remMeta.devs[i].rkey;

  for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
    rCommDev = rComm->devs + i;
    NCCLCHECKGOTO(wrap_ibv_reg_mr(
                      &rCommDev->ctsFifoMr, rCommDev->base.pd,
                      &rComm->remCtsFifo.elems, sizeof(rComm->remCtsFifo.elems),
                      IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
                          IBV_ACCESS_REMOTE_READ),
                  ret, fail);
    rCommDev->sge.lkey = rCommDev->ctsFifoMr->lkey;
    NCCLCHECKGOTO(
        wrap_ibv_reg_mr(&rCommDev->cmplsRecordsMr, rCommDev->base.pd,
                        &rComm->cmplsRecords, sizeof(rComm->cmplsRecords),
                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                            IBV_ACCESS_REMOTE_READ),
        ret, fail);
    meta.devs[i].rkey = rCommDev->cmplsRecordsMr->rkey;
  }
  if (mpibParamIbUseInline())
    rComm->remCtsFifo.flags = IBV_SEND_INLINE;

  for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
    rCommDev = rComm->devs + i;
    ibDev = mpibDevs + rCommDev->base.ibDevN;

    meta.devs[i].lid = ibDev->portAttr.lid;
    meta.devs[i].link_layer = rCommDev->base.gidInfo.link_layer =
        ibDev->portAttr.link_layer;
    meta.devs[i].ib_port = ibDev->portNum;
    meta.devs[i].gid.global.subnet_prefix =
        rCommDev->base.gidInfo.localGid.global.subnet_prefix;
    meta.devs[i].gid.global.interface_id =
        rCommDev->base.gidInfo.localGid.global.interface_id;
    meta.devs[i].mtu = ibDev->portAttr.active_mtu;
  }
  meta.addr = (uint64_t)rComm->cmplsRecords;
  meta.nqpsSout = rComm->base.nqpsSout;
  meta.nqpsSup = rComm->base.nqpsSup;
  meta.sl = remMeta.sl;
  meta.tc = remMeta.tc;

  meta.ndevs = rComm->base.vProps.ndevs;
  strncpy(meta.devName, mergedDev->devName, MAX_MERGED_DEV_NAME);

  stage->state = mpibCommStateSend;
  stage->offset = 0;
  if (stage->buffer) {
    free(stage->buffer);
    stage->buffer = NULL;
  }
  NCCLCHECKGOTO(mpibMalloc((void **)&stage->buffer,
                           sizeof(struct mpibConnectionMetadata)),
                ret, fail);
  memcpy(stage->buffer, &meta, sizeof(struct mpibConnectionMetadata));

ib_send:
  NCCLCHECKGOTO(
      mpibSocketProgress(MPIB_SOCKET_SEND, &rComm->base.sock, stage->buffer,
                         sizeof(struct mpibConnectionMetadata), &stage->offset),
      ret, fail);
  if (stage->offset < (int)sizeof(struct mpibConnectionMetadata)) {
    return ncclSuccess;
  }

  stage->offset = 0;
  stage->state = mpibCommStatePendingReady;

ib_recv_ready:
  NCCLCHECKGOTO(mpibSocketProgress(MPIB_SOCKET_RECV, &rComm->base.sock,
                                   &rComm->base.ready, sizeof(int),
                                   &stage->offset),
                ret, fail);
  if (stage->offset != (int)sizeof(int)) {
    return ncclSuccess;
  }

  /* Register connection with agent */
  {
    uint16_t counter =
        g_mpib_conn_counter.fetch_add(1, std::memory_order_relaxed);
    rComm->conn_id = MPIB_MAKE_CONN_ID(getpid(), counter);

    /* Extract IPv4 from GID (IPv4-mapped IPv6: ::ffff:a.b.c.d) */
    /* SOUT = dev[0], SUP = dev[1] */
    int sout_src_is_v4 = 0, sout_dst_is_v4 = 0;
    int sup_src_is_v4 = 0, sup_dst_is_v4 = 0;
    uint32_t sout_src_ip =
        mpibGidToIpv4(&rComm->devs[0].base.gidInfo.localGid, &sout_src_is_v4);
    uint32_t sout_dst_ip =
        mpibGidToIpv4(&rComm->base.remDevs[0].gid, &sout_dst_is_v4);
    uint32_t sup_src_ip =
        mpibGidToIpv4(&rComm->devs[1].base.gidInfo.localGid, &sup_src_is_v4);
    uint32_t sup_dst_ip =
        mpibGidToIpv4(&rComm->base.remDevs[1].gid, &sup_dst_is_v4);
    if (!sout_src_is_v4 || !sout_dst_is_v4 || !sup_src_is_v4 ||
        !sup_dst_is_v4) {
      WARN("NET/MPIB : Non-IPv4 GID detected for agent registration: "
           "sout_src_v4=%d sout_dst_v4=%d sup_src_v4=%d sup_dst_v4=%d",
           sout_src_is_v4, sout_dst_is_v4, sup_src_is_v4, sup_dst_is_v4);
    }

    /* Classify connection path based on SOUT subnet */
    int sameIsland = mpibIsSameIsland(sout_src_ip, sout_dst_ip);
    rComm->base.pathSupBw = sameIsland ? UINT32_MAX : 0;
    rComm->base.hintActive = sameIsland ? 1 : 0;
    INFO(NCCL_NET,
         "NET/MPIB : Path classification: %s hintActive=%d "
         "(sout_src=0x%08x sout_dst=0x%08x)",
         sameIsland ? "SUP (intra-island)" : "SOUT (inter-island)",
         rComm->base.hintActive, sout_src_ip, sout_dst_ip);

    ncclResult_t regRet =
        mpibAgentRegister(rComm->conn_id, sout_src_ip, sout_dst_ip, sup_src_ip,
                          sup_dst_ip, &rComm->hint_slot);
    if (regRet != ncclSuccess) {
      WARN("NET/MPIB : Agent registration failed for conn_id=0x%08x",
           rComm->conn_id);
      ret = regRet;
      goto fail;
    }
    INFO(NCCL_NET, "NET/MPIB : Registered conn_id=0x%08x hint_slot=%u",
         rComm->conn_id, rComm->hint_slot);
  }

  *recvComm = rComm;
  {
    char line[MPIB_SOCKET_NAME_MAXLEN + 1];
    union mpibSocketAddress addr;
    mpibSocketGetAddr(&rComm->base.sock, &addr);
    INFO(NCCL_NET, "NET/MPIB: mpibAccept DONE peer=%s",
         mpibSocketToString(&addr, line));
  }
exit:
  if (stage->buffer)
    free(stage->buffer);
  free(stage);
  lComm->stage = NULL;
  return ret;
fail:
  free(rComm);
  goto exit;
}

__hidden ncclResult_t mpibCloseSend(void *sendComm) {
  struct mpibSendComm *comm = (struct mpibSendComm *)sendComm;
  if (comm) {
    /* Deregister from agent (best-effort) */
    if (comm->conn_id != 0) {
      mpibAgentDeregister(comm->conn_id);
    }

    NCCLCHECK(mpibSocketClose(&comm->base.sock));
    for (int q = 0; q < comm->base.nqps; q++)
      if (comm->base.qps[q].qp != NULL)
        NCCLCHECK(wrap_ibv_destroy_qp(comm->base.qps[q].qp));

    for (int i = 0; i < comm->base.vProps.ndevs; i++) {
      struct mpibSendCommDev *commDev = comm->devs + i;
      if (commDev->ctsFifoMr != NULL)
        NCCLCHECK(wrap_ibv_dereg_mr(commDev->ctsFifoMr));
      if (commDev->cmplsRecordsMr != NULL)
        NCCLCHECK(wrap_ibv_dereg_mr(commDev->cmplsRecordsMr));
      if (commDev->putSignalScratchpadMr != NULL)
        NCCLCHECK(wrap_ibv_dereg_mr(commDev->putSignalScratchpadMr));
      NCCLCHECK(mpibDestroyBase(&commDev->base));
    }

    free(comm);
  }
  return ncclSuccess;
}

__hidden ncclResult_t mpibCloseRecv(void *recvComm) {
  struct mpibRecvComm *comm = (struct mpibRecvComm *)recvComm;
  if (comm) {
    /* Deregister from agent (best-effort) */
    if (comm->conn_id != 0) {
      mpibAgentDeregister(comm->conn_id);
    }

    NCCLCHECK(mpibSocketClose(&comm->base.sock));

    for (int q = 0; q < comm->base.nqps; q++)
      if (comm->base.qps[q].qp != NULL)
        NCCLCHECK(wrap_ibv_destroy_qp(comm->base.qps[q].qp));

    for (int i = 0; i < comm->base.vProps.ndevs; i++) {
      struct mpibRecvCommDev *commDev = comm->devs + i;
      if (commDev->ctsFifoMr != NULL)
        NCCLCHECK(wrap_ibv_dereg_mr(commDev->ctsFifoMr));
      if (commDev->cmplsRecordsMr != NULL)
        NCCLCHECK(wrap_ibv_dereg_mr(commDev->cmplsRecordsMr));
      NCCLCHECK(mpibDestroyBase(&commDev->base));
    }

    free(comm);
  }
  return ncclSuccess;
}

__hidden ncclResult_t mpibCloseListen(void *listenComm) {
  struct mpibListenComm *comm = (struct mpibListenComm *)listenComm;
  if (comm) {
    mpibSocketClose(&comm->sock);
    if (comm->stage)
      free(comm->stage);
    free(comm);
  }
  return ncclSuccess;
}

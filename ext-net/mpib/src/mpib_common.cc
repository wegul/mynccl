#include "mpib_common.h"

ncclDebugLogger_t mpibLogFunction = nullptr;

char mpibIfName[MPIB_MAX_IF_NAME_SIZE + 1];
union mpibSocketAddress mpibIfAddr;

int mpibNMergedIbDevs = -1;
int mpibNIbDevs = -1;
struct mpibMergedDev mpibMergedDevs[MAX_IB_VDEVS];
struct mpibDev mpibDevs[MAX_IB_DEVS];
int mpibRelaxedOrderingEnabled = 0;

ncclProfilerCallback_t mpibProfilerFunction = nullptr;

MPIB_PARAM(IbAsyncEvents, "IB_RETURN_ASYNC_EVENTS", 1);

ncclResult_t mpibStatsCheckFatalCount(struct mpibStats *stat,
                                      const char *funcName) {
  if (mpibParamIbAsyncEvents() &&
      COMPILER_ATOMIC_LOAD(&stat->fatalErrorCount, std::memory_order_relaxed)) {
    WARN("communicator encountered a fatal error (detected in %s)", funcName);
    return ncclSystemError;
  }
  return ncclSuccess;
}

struct mpibNetCommDevBase *mpibGetNetCommDevBase(struct mpibNetCommBase *base,
                                                 int devIndex) {
  if (base->isSend) {
    struct mpibSendComm *sComm = (struct mpibSendComm *)base;
    return &sComm->devs[devIndex].base;
  } else {
    struct mpibRecvComm *rComm = (struct mpibRecvComm *)base;
    return &rComm->devs[devIndex].base;
  }
}

std::thread mpibAsyncThread;
void *mpibAsyncThreadMain(void *args) {
  struct mpibDev *dev = (struct mpibDev *)args;
  while (1) {
    struct ibv_async_event event;
    if (ncclSuccess != wrap_ibv_get_async_event(dev->context, &event)) {
      break;
    }
    char *str;
    struct ibv_cq *cq = event.element.cq;
    struct ibv_qp *qp = event.element.qp;
    struct ibv_srq *srq = event.element.srq;
    if (ncclSuccess != wrap_ibv_event_type_str(&str, event.event_type)) {
      break;
    }
    switch (event.event_type) {
    case IBV_EVENT_DEVICE_FATAL:
      WARN("NET/MPIB : %s:%d async fatal event: %s", dev->devName, dev->portNum,
           str);
      mpibDevFatalError(dev);
      break;
    case IBV_EVENT_CQ_ERR:
      WARN("NET/MPIB : %s:%d async fatal event on CQ (%p): %s", dev->devName,
           dev->portNum, cq, str);
      mpibCqFatalError(cq);
      break;
    case IBV_EVENT_QP_FATAL:
    case IBV_EVENT_QP_REQ_ERR:
    case IBV_EVENT_QP_ACCESS_ERR:
      WARN("NET/MPIB : %s:%d async fatal event on QP (%p): %s", dev->devName,
           dev->portNum, qp, str);
      mpibQpFatalError(qp);
      break;
    case IBV_EVENT_SRQ_ERR:
      WARN("NET/MPIB : %s:%d async fatal event on SRQ, unused for now (%p): %s",
           dev->devName, dev->portNum, srq, str);
      break;
    case IBV_EVENT_GID_CHANGE:
      WARN("NET/MPIB : %s:%d GID table changed", dev->devName, dev->portNum);
      break;
    case IBV_EVENT_PATH_MIG_ERR:
    case IBV_EVENT_PORT_ERR:
    case IBV_EVENT_PATH_MIG:
    case IBV_EVENT_PORT_ACTIVE:
    case IBV_EVENT_SQ_DRAINED:
    case IBV_EVENT_LID_CHANGE:
    case IBV_EVENT_PKEY_CHANGE:
    case IBV_EVENT_SM_CHANGE:
    case IBV_EVENT_QP_LAST_WQE_REACHED:
    case IBV_EVENT_CLIENT_REREGISTER:
    case IBV_EVENT_SRQ_LIMIT_REACHED:
      WARN("NET/MPIB : %s:%d Got non-fatal async event: %s(%d)", dev->devName,
           dev->portNum, str, event.event_type);
      break;
    case IBV_EVENT_COMM_EST:
      break;
    default:
      WARN("NET/MPIB : %s:%d unknown event type (%d)", dev->devName,
           dev->portNum, event.event_type);
      break;
    }
    if (ncclSuccess != wrap_ibv_ack_async_event(&event)) {
      break;
    }
  }
  return NULL;
}

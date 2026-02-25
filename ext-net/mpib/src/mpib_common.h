#pragma once

#include "ibvwrap.h"
#include "mpib_compat.h"
#include "mpib_param.h"
#include "mpib_socket.h"
#include "mpib_utils.h"

#include <assert.h>
#include <mutex>
#include <poll.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>

ncclResult_t mpibRtsQp(struct ibv_qp *qp);

#define MAXSUFFIXSIZE 16
#define MAXNAMESIZE (64 + MAXSUFFIXSIZE)

extern char mpibIfName[MPIB_MAX_IF_NAME_SIZE + 1];
extern union mpibSocketAddress mpibIfAddr;

struct mpibMr {
  uintptr_t addr;
  size_t pages;
  int refs;
  ibv_mr *mr;
};

struct mpibMrCache {
  struct mpibMr *slots;
  int capacity, population;
};

extern int mpibNMergedIbDevs;
#define MPIB_MAX_DEVS 2 // MPIB-CUSTOM: Exactly 2 devices (SOUT + SUP)
#define MPIB_CTS_SIGNAL_INTERVAL 128 // Signal CTS every N slots

// SRQ watermarks (hardcoded per design doc)
#define MPIB_SRQ_LOW_WATER 64
#define MPIB_SRQ_HIGH_WATER 512

// IMM data encoding for SRQ-based completion
// Layout: [7:0] slot_idx, [9:8] active_mask, [31:10] size_q
#define MPIB_IMM_SLOT_BITS 8
#define MPIB_IMM_MASK_BITS 2
#define MPIB_IMM_SIZEQ_BITS 22
#define MPIB_IMM_SIZE_GRANULARITY 128
#define MPIB_IMM_SIZEQ_SENTINEL ((1u << MPIB_IMM_SIZEQ_BITS) - 1)

// IMM data pack/unpack helpers
static inline uint32_t mpibImmEncode(uint8_t slot, uint8_t mask,
                                     uint32_t size_q) {
  return ((uint32_t)slot) | ((uint32_t)(mask & 0x3) << 8) |
         ((size_q & MPIB_IMM_SIZEQ_SENTINEL) << 10);
}
static inline void mpibImmDecode(uint32_t imm, uint8_t *slot, uint8_t *mask,
                                 uint32_t *size_q) {
  *slot = (uint8_t)(imm & 0xFF);
  *mask = (uint8_t)((imm >> 8) & 0x3);
  *size_q = (imm >> 10) & MPIB_IMM_SIZEQ_SENTINEL;
}

#define MAX_MERGED_DEV_NAME (MAXNAMESIZE * MPIB_MAX_DEVS) + MPIB_MAX_DEVS
struct alignas(64) mpibMergedDev {
  ncclNetVDeviceProps_t vProps;
  int speed;
  char devName[MAX_MERGED_DEV_NAME];
};

struct mpibStats {
  int fatalErrorCount;
};

enum mpibProvider {
  IB_PROVIDER_NONE = 0,
  IB_PROVIDER_MLX5 = 1,
  IB_PROVIDER_MAX = 2,
};

extern int mpibNIbDevs;
struct alignas(64) mpibDev {
  std::mutex mutex;
  int device;
  uint64_t guid;
  uint8_t portNum;
  uint8_t link;
  int speed;
  ibv_context *context;
  int pdRefs;
  ibv_pd *pd;
  char devName[MAXNAMESIZE];
  char *pciPath;
  char *virtualPciPath;
  int realPort;
  int maxQp;
  float latency;
  struct mpibMrCache mrCache;
  int ar;
  struct ibv_port_attr portAttr;
  struct mpibStats stats;
  enum mpibProvider ibProvider;
  int dmaBufSupported; // 1=yes, -1=no, 0=untested
};

#define MAX_IB_DEVS 32
#define MAX_IB_VDEVS MAX_IB_DEVS * 8
extern struct mpibMergedDev mpibMergedDevs[MAX_IB_VDEVS];
extern struct mpibDev mpibDevs[MAX_IB_DEVS];
extern int mpibRelaxedOrderingEnabled;

#define MPIB_IB_LLSTR(ll)                                                      \
  (((ll) == IBV_LINK_LAYER_INFINIBAND)                                         \
       ? "IB"                                                                  \
       : (((ll) == IBV_LINK_LAYER_ETHERNET) ? "RoCE" : "UNSPECIFIED"))

struct mpibDevInfo {
  uint32_t lid;
  uint8_t ib_port;
  enum ibv_mtu mtu;
  uint8_t link_layer;
  union ibv_gid gid;
  uint32_t rkey;
  union ibv_gid remoteGid;
};

struct mpibGidInfo {
  uint8_t link_layer;
  union ibv_gid localGid;
  int32_t localGidIndex;
};

#define MAX_QPS_PER_REQ 8
struct mpibProfilerInfo {
  void *qpEventHandles[MAX_QPS_PER_REQ];
  int qpIndex[MAX_QPS_PER_REQ];
  int nEventHandles;
  void *pHandle;
};

#define MPIB_NET_IB_MAX_RECVS 8
#define MPIB_IB_MAX_QPS 128

#define MPIB_NET_IB_REQ_UNUSED 0
#define MPIB_NET_IB_REQ_SEND 1
#define MPIB_NET_IB_REQ_RECV 2
#define MPIB_NET_IB_REQ_FLUSH 3
#define MPIB_NET_IB_REQ_GIN_IPUT 4
extern const char *mpibReqTypeStr[];

struct mpibRequest {
  struct mpibNetCommBase *base;
  int type;
  struct mpibSocket *sock;
  int events[MPIB_MAX_DEVS];
  struct mpibNetCommDevBase *devBases[MPIB_MAX_DEVS];
#ifdef NCCL_ENABLE_NET_PROFILING
  struct mpibProfilerInfo pInfo[MPIB_NET_IB_MAX_RECVS];
#endif
  uint32_t nreqs;
  // SRQ-based completion tracking (RECV only)
  uint8_t slot;          // Slot index (0..255)
  uint8_t expected_mask; // Active rail mask learned from first IMM (0 = unset)
  uint8_t seen_mask;     // Rails that have delivered an IMM
  uint8_t _pad;
  union {
    struct {
      size_t size;
      void *data;
      uint32_t lkeys[MPIB_MAX_DEVS];
      int offset;
    } send;
    struct {
      int *sizes;
    } recv;
    struct {
      int rank;
    } iput;
  };
};

struct mpibNetCommDevBase {
  int ibDevN;
  struct ibv_pd *pd;
  struct ibv_cq *cq;
  // SRQ for recv comms (NULL for send comms)
  struct ibv_srq *srq;
  int srqPosted; // Number of generic WQEs posted to SRQ
  struct mpibGidInfo gidInfo;
};

struct alignas(32) mpibSendFifo {
  uint64_t addr;
  uint64_t size;
  uint32_t rkeys[MPIB_MAX_DEVS];
  uint32_t nreqs;
  uint32_t tag;
  uint64_t idx;
};

struct mpibQp {
  struct ibv_qp *qp;
  int devIndex;
  int remDevIdx;
};

#define NET_IB_MAX_REQUESTS (NCCL_NET_MAX_REQUESTS * MPIB_NET_IB_MAX_RECVS)
static_assert(NET_IB_MAX_REQUESTS <= 256,
              "request id are encoded in wr_id and we need up to 8 requests "
              "ids per completion");

struct mpibRemCompletionsRecords {
  int elems[NET_IB_MAX_REQUESTS][MPIB_NET_IB_MAX_RECVS];
  uint64_t addr;
  uint32_t rkeys[MPIB_MAX_DEVS];
};

struct alignas(8) mpibSendCommDev {
  struct mpibNetCommDevBase base;
  struct ibv_sge sge;
  struct ibv_mr *ctsFifoMr;
  struct ibv_mr *cmplsRecordsMr;
  struct ibv_mr *putSignalScratchpadMr;
};

// Wrapper to track an MR per-device
struct mpibMrHandle {
  ibv_mr *mrs[MPIB_MAX_DEVS];
};

struct alignas(32) mpibNetCommBase {
  int dev;
  struct mpibQp qps[MPIB_IB_MAX_QPS];
  uint64_t fifoHead;
  uint32_t nqps;
  uint32_t nqpsSout; // QP count on SOUT (dev0)
  uint32_t nqpsSup;  // QP count on SUP (dev1)
  struct mpibSocket sock;
  int ready;
  int isSend;
  int nRemDevs;
  struct mpibDevInfo remDevs[MPIB_MAX_DEVS];
  struct mpibStats stats;
  ncclNetVDeviceProps_t vProps;
  struct mpibRequest reqs[NET_IB_MAX_REQUESTS];
  // SRQ: slot→request map for recv comms (used by completion handler)
  struct mpibRequest *slotReq[NET_IB_MAX_REQUESTS];
};

struct mpibSendComm {
  struct mpibNetCommBase base;
  struct mpibSendFifo ctsFifo[NET_IB_MAX_REQUESTS][MPIB_NET_IB_MAX_RECVS];
  struct ibv_sge sges[MPIB_NET_IB_MAX_RECVS];
  struct ibv_send_wr
      wrs[MPIB_NET_IB_MAX_RECVS + 1]; // +1 for extra WR when nreqs > 1
  struct mpibSendCommDev devs[MPIB_MAX_DEVS];
  struct mpibRequest *fifoReqs[NET_IB_MAX_REQUESTS][MPIB_NET_IB_MAX_RECVS];
  struct mpibRemCompletionsRecords remCmplsRecords;
  uint64_t putSignalScratchpad;
  /* Agent integration fields */
  uint32_t conn_id;   /* Connection ID for agent registration */
  uint32_t hint_slot; /* Index into agent hint SHM */
};

static_assert((sizeof(struct mpibNetCommBase) % 32) == 0,
              "mpibNetCommBase size must be 32-byte multiple to ensure "
              "ctsFifo is at proper offset");
static_assert((offsetof(struct mpibSendComm, ctsFifo) % 32) == 0,
              "mpibSendComm ctsFifo must be 32-byte aligned");
static_assert((sizeof(struct mpibSendFifo) % 32) == 0,
              "mpibSendFifo element size must be 32-byte multiples");
static_assert((offsetof(struct mpibSendComm, sges) % 32) == 0,
              "sges must be 32-byte aligned");
static_assert((offsetof(struct mpibSendComm, wrs) % 32) == 0,
              "wrs must be 32-byte aligned");

struct mpibRemCtsFifo {
  struct mpibSendFifo elems[NET_IB_MAX_REQUESTS][MPIB_NET_IB_MAX_RECVS];
  uint64_t addr;
  uint32_t rkeys[MPIB_MAX_DEVS];
  uint32_t flags;
};

struct alignas(16) mpibRecvCommDev {
  struct mpibNetCommDevBase base;
  struct ibv_mr *ctsFifoMr;
  struct ibv_mr *cmplsRecordsMr;
  struct ibv_sge sge;
};

struct mpibRecvComm {
  struct mpibNetCommBase base;
  struct mpibRecvCommDev devs[MPIB_MAX_DEVS];
  struct mpibRemCtsFifo remCtsFifo;
  int cmplsRecords[NET_IB_MAX_REQUESTS][MPIB_NET_IB_MAX_RECVS];
  /* Agent integration fields */
  uint32_t conn_id;   /* Connection ID for agent registration */
  uint32_t hint_slot; /* Index into agent hint SHM */
};

static_assert((offsetof(struct mpibRecvComm, remCtsFifo) % 32) == 0,
              "mpibRecvComm ctsFifo must be 32-byte aligned");

struct mpibCommStage;
struct mpibListenComm {
  int dev;
  struct mpibSocket sock;
  struct mpibCommStage *stage;
};

static inline ncclResult_t mpibStatsInit(struct mpibStats *stat) {
  COMPILER_ATOMIC_STORE(&stat->fatalErrorCount, 0, std::memory_order_relaxed);
  return ncclSuccess;
}
static void mpibStatsFatalError(struct mpibStats *stat) {
  COMPILER_ATOMIC_FETCH_ADD(&stat->fatalErrorCount, 1,
                            std::memory_order_relaxed);
}
static void mpibQpFatalError(struct ibv_qp *qp) {
  mpibStatsFatalError((struct mpibStats *)qp->qp_context);
}
static void mpibCqFatalError(struct ibv_cq *cq) {
  mpibStatsFatalError((struct mpibStats *)cq->cq_context);
}
static void mpibDevFatalError(struct mpibDev *dev) {
  mpibStatsFatalError(&dev->stats);
}
ncclResult_t mpibStatsCheckFatalCount(struct mpibStats *stat,
                                      const char *funcName);

extern ncclProfilerCallback_t mpibProfilerFunction;

extern std::thread mpibAsyncThread;
void *mpibAsyncThreadMain(void *args);

void mpibAddEvent(struct mpibRequest *req, int devIndex);

struct mpibNetCommDevBase *mpibGetNetCommDevBase(struct mpibNetCommBase *base,
                                                 int devIndex);

static inline ncclResult_t
mpibCommBaseGetQpForRequest(struct mpibNetCommBase *baseComm, const uint32_t id,
                            const uint8_t devIndex, struct mpibQp **outQp,
                            uint32_t *outQpIndex) {
  // devIndex: 0=SOUT, 1=SUP
  // id: fifoHead counter for round-robin within device
  // Select one QP from the device's pool using round-robin
  if (devIndex == 0) {
    // SOUT: pick from qps[0..nqpsSout-1]
    *outQpIndex = (baseComm->nqpsSout > 0) ? (id % baseComm->nqpsSout) : 0;
  } else {
    // SUP: pick from qps[nqpsSout..nqps-1]
    *outQpIndex = baseComm->nqpsSout +
                  ((baseComm->nqpsSup > 0) ? (id % baseComm->nqpsSup) : 0);
  }
  *outQp = &(baseComm->qps[*outQpIndex]);
  assert(*outQp != NULL);
  return ncclSuccess;
}

static inline int
mpibCommBaseGetNqpsPerRequest(struct mpibNetCommBase *baseComm) {
  // Each request uses one QP per device (ndevs QPs total)
  return baseComm->vProps.ndevs;
}

ncclResult_t mpibInitCommDevBase(int ibDevN, struct mpibNetCommDevBase *base,
                                 void *cq_context);
ncclResult_t mpibDestroyBase(struct mpibNetCommDevBase *base);
ncclResult_t mpibCreateQp(uint8_t ib_port, struct mpibNetCommDevBase *base,
                          int access_flags, void *qp_context,
                          struct mpibQp *qp);
ncclResult_t mpibRtrQp(struct ibv_qp *qp, struct mpibGidInfo *sGidInfo,
                       uint32_t dest_qp_num, struct mpibDevInfo *info,
                       bool fifoTc, int tc, int sl);
ncclResult_t mpibRtsQp(struct ibv_qp *qp);
ncclResult_t mpibGetGidIndex(struct ibv_context *context, uint8_t portNum,
                             struct ibv_port_attr *portAttr, int *gidIndex);

ncclResult_t mpibGetRequest(struct mpibNetCommBase *base,
                            struct mpibRequest **req);
ncclResult_t mpibFreeRequest(struct mpibRequest *r);

ncclResult_t mpibRegMr(void *comm, void *data, size_t size, int type,
                       void **mhandle);
ncclResult_t mpibRegMrDmaBuf(void *comm, void *data, size_t size, int type,
                             uint64_t offset, int fd, void **mhandle);
ncclResult_t mpibDeregMr(void *comm, void *mhandle);

// GDR support detection (mpib_gdr.cc)
ncclResult_t mpibGdrSupport();
ncclResult_t mpibDmaBufSupport(int dev);

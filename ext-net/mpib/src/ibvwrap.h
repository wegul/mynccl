#pragma once

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include "mpib_compat.h"

#ifdef NCCL_BUILD_RDMA_CORE
#include <infiniband/verbs.h>
#else
#include "ibvcore.h"
#endif

typedef enum ibv_return_enum {
  IBV_SUCCESS = 0,
} ibv_return_t;

ncclResult_t wrap_ibv_symbols(void);

ncclResult_t wrap_ibv_fork_init(void);
ncclResult_t wrap_ibv_get_device_list(struct ibv_device ***ret,
                                      int *num_devices);
ncclResult_t wrap_ibv_free_device_list(struct ibv_device **list);
const char *wrap_ibv_get_device_name(struct ibv_device *device);
ncclResult_t wrap_ibv_open_device(struct ibv_context **ret,
                                  struct ibv_device *device);
ncclResult_t wrap_ibv_close_device(struct ibv_context *context);
ncclResult_t wrap_ibv_get_async_event(struct ibv_context *context,
                                      struct ibv_async_event *event);
ncclResult_t wrap_ibv_ack_async_event(struct ibv_async_event *event);
ncclResult_t wrap_ibv_query_device(struct ibv_context *context,
                                   struct ibv_device_attr *device_attr);
ncclResult_t wrap_ibv_query_port(struct ibv_context *context, uint8_t port_num,
                                 struct ibv_port_attr *port_attr);
ncclResult_t wrap_ibv_query_gid(struct ibv_context *context, uint8_t port_num,
                                int index, union ibv_gid *gid);
ncclResult_t wrap_ibv_query_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr,
                               int attr_mask,
                               struct ibv_qp_init_attr *init_attr);
ncclResult_t wrap_ibv_alloc_pd(struct ibv_pd **ret,
                               struct ibv_context *context);
ncclResult_t wrap_ibv_dealloc_pd(struct ibv_pd *pd);
ncclResult_t wrap_ibv_reg_mr(struct ibv_mr **ret, struct ibv_pd *pd, void *addr,
                             size_t length, int access);
struct ibv_mr *wrap_direct_ibv_reg_mr(struct ibv_pd *pd, void *addr,
                                      size_t length, int access);
ncclResult_t wrap_ibv_reg_mr_iova2(struct ibv_mr **ret, struct ibv_pd *pd,
                                   void *addr, size_t length, uint64_t iova,
                                   int access);

ncclResult_t wrap_ibv_reg_dmabuf_mr(struct ibv_mr **ret, struct ibv_pd *pd,
                                    uint64_t offset, size_t length,
                                    uint64_t iova, int fd, int access);
struct ibv_mr *wrap_direct_ibv_reg_dmabuf_mr(struct ibv_pd *pd, uint64_t offset,
                                             size_t length, uint64_t iova,
                                             int fd, int access);

ncclResult_t wrap_ibv_dereg_mr(struct ibv_mr *mr);
ncclResult_t wrap_ibv_create_cq(struct ibv_cq **ret,
                                struct ibv_context *context, int cqe,
                                void *cq_context,
                                struct ibv_comp_channel *channel,
                                int comp_vector);
ncclResult_t wrap_ibv_destroy_cq(struct ibv_cq *cq);
static inline ncclResult_t wrap_ibv_poll_cq(struct ibv_cq *cq, int num_entries,
                                            struct ibv_wc *wc, int *num_done) {
  int done = cq->context->ops.poll_cq(cq, num_entries, wc);
  if (done < 0)
    return ncclSystemError;
  *num_done = done;
  return ncclSuccess;
}

ncclResult_t wrap_ibv_create_qp(struct ibv_qp **ret, struct ibv_pd *pd,
                                struct ibv_qp_init_attr *qp_init_attr);
ncclResult_t wrap_ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr,
                                int attr_mask);
ncclResult_t wrap_ibv_destroy_qp(struct ibv_qp *qp);
ncclResult_t wrap_ibv_event_type_str(char **str, enum ibv_event_type event);
ncclResult_t wrap_ibv_query_ece(struct ibv_qp *qp, struct ibv_ece *ece,
                                int *supported);
ncclResult_t wrap_ibv_set_ece(struct ibv_qp *qp, struct ibv_ece *ece,
                              int *supported);

static inline ncclResult_t wrap_ibv_post_send(struct ibv_qp *qp,
                                              struct ibv_send_wr *wr,
                                              struct ibv_send_wr **bad_wr) {
  int ret = qp->context->ops.post_send(qp, wr, bad_wr);
  if (ret != IBV_SUCCESS) {
    WARN("ibv_post_send() failed with error %s, Bad WR %p, First WR %p",
         strerror(ret), wr, *bad_wr);
    return ncclSystemError;
  }
  return ncclSuccess;
}

static inline ncclResult_t wrap_ibv_post_recv(struct ibv_qp *qp,
                                              struct ibv_recv_wr *wr,
                                              struct ibv_recv_wr **bad_wr) {
  int ret = qp->context->ops.post_recv(qp, wr, bad_wr);
  if (ret != IBV_SUCCESS) {
    WARN("ibv_post_recv() failed with error %s", strerror(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

// SRQ support
ncclResult_t wrap_ibv_create_srq(struct ibv_srq **ret, struct ibv_pd *pd,
                                 struct ibv_srq_init_attr *srq_init_attr);
ncclResult_t wrap_ibv_destroy_srq(struct ibv_srq *srq);

static inline ncclResult_t wrap_ibv_post_srq_recv(struct ibv_srq *srq,
                                                  struct ibv_recv_wr *wr,
                                                  struct ibv_recv_wr **bad_wr) {
  int ret = srq->context->ops.post_srq_recv(srq, wr, bad_wr);
  if (ret != IBV_SUCCESS) {
    WARN("ibv_post_srq_recv() failed with error %s", strerror(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

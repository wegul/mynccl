#include "ibvsymbols.h"

#include <sys/types.h>
#include <unistd.h>

#ifdef NCCL_BUILD_RDMA_CORE
#include <infiniband/verbs.h>
#define ASSIGN_SYM(container, symbol, name) container->name = &symbol;

struct ibv_mr* ibv_internal_reg_mr(struct ibv_pd* pd, void* addr, size_t length, int access) {
  return ibv_reg_mr(pd, addr, length, access);
}

int ibv_internal_query_port(struct ibv_context* context, uint8_t port_num, struct ibv_port_attr* port_attr) {
  return ibv_query_port(context, port_num, port_attr);
}

ncclResult_t buildIbvSymbols(struct ncclIbvSymbols* ibvSymbols) {
  ASSIGN_SYM(ibvSymbols, ibv_get_device_list, ibv_internal_get_device_list);
  ASSIGN_SYM(ibvSymbols, ibv_free_device_list, ibv_internal_free_device_list);
  ASSIGN_SYM(ibvSymbols, ibv_get_device_name, ibv_internal_get_device_name);
  ASSIGN_SYM(ibvSymbols, ibv_open_device, ibv_internal_open_device);
  ASSIGN_SYM(ibvSymbols, ibv_close_device, ibv_internal_close_device);
  ASSIGN_SYM(ibvSymbols, ibv_get_async_event, ibv_internal_get_async_event);
  ASSIGN_SYM(ibvSymbols, ibv_ack_async_event, ibv_internal_ack_async_event);
  ASSIGN_SYM(ibvSymbols, ibv_query_device, ibv_internal_query_device);
  ASSIGN_SYM(ibvSymbols, ibv_query_gid, ibv_internal_query_gid);
  ASSIGN_SYM(ibvSymbols, ibv_query_qp, ibv_internal_query_qp);
  ASSIGN_SYM(ibvSymbols, ibv_alloc_pd, ibv_internal_alloc_pd);
  ASSIGN_SYM(ibvSymbols, ibv_dealloc_pd, ibv_internal_dealloc_pd);

  ASSIGN_SYM(ibvSymbols, ibv_reg_mr_iova2, ibv_internal_reg_mr_iova2);
  ASSIGN_SYM(ibvSymbols, ibv_reg_dmabuf_mr, ibv_internal_reg_dmabuf_mr);

  ASSIGN_SYM(ibvSymbols, ibv_dereg_mr, ibv_internal_dereg_mr);
  ASSIGN_SYM(ibvSymbols, ibv_create_cq, ibv_internal_create_cq);
  ASSIGN_SYM(ibvSymbols, ibv_destroy_cq, ibv_internal_destroy_cq);
  ASSIGN_SYM(ibvSymbols, ibv_create_qp, ibv_internal_create_qp);
  ASSIGN_SYM(ibvSymbols, ibv_modify_qp, ibv_internal_modify_qp);
  ASSIGN_SYM(ibvSymbols, ibv_destroy_qp, ibv_internal_destroy_qp);
  ASSIGN_SYM(ibvSymbols, ibv_fork_init, ibv_internal_fork_init);
  ASSIGN_SYM(ibvSymbols, ibv_event_type_str, ibv_internal_event_type_str);

  ASSIGN_SYM(ibvSymbols, ibv_query_ece, ibv_internal_query_ece);
  ASSIGN_SYM(ibvSymbols, ibv_set_ece, ibv_internal_set_ece);

  ibvSymbols->ibv_internal_reg_mr = &ibv_internal_reg_mr;
  ibvSymbols->ibv_internal_query_port = &ibv_internal_query_port;
  return ncclSuccess;
}
#else
#include "ibvcore.h"
#include "ibvwrap.h"
#include "ibvsymbols.h"
#include "ibvsymbols.h"
#include "mpib_compat.h"

extern ncclResult_t buildIbvSymbols(struct ncclIbvSymbols* ibvSymbols);
#endif

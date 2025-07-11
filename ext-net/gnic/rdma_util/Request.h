#pragma once

#include "BufPool.h"
#include <cstddef>
namespace gnic {

/**
 @brief provides utility functions for RDMA operations such as send(), recv(),
 regMr, deregMr...
**/
// class RDMAUtil {
// private:
//   /* data */
// public:
//   RDMAUtil(/* args */);
//   ~RDMAUtil();
// };
enum ReqOp { SEND = 0, RECV = 1 };
/**
 * @brief ucclRequest is a handle provided by the user to post a request to UCCL
 * RDMAEndpoint. It is the responsibility of the user to manage the memory of
 * ucclRequest. UCCL RDMAEndpoint will not free the memory of ucclRequest. UCCL
 * fills the ucclRequest with the result of the request. The user can use the
 * ucclRequest to check the status of the request.
 */
struct Request {
  enum ReqOp op;
  union {
    int n;
    int mid; // used for multi-send
  };
  // union {
  //   PollCtx *poll_ctx;
  //   // For reducing overhead of PollCtx for RC and Flush operation.
  //   uint64_t rc_or_flush_done;
  // };
  int sockfd; // For send operation.
  void *context;
  void *req_pool;
  uint32_t engine_idx;
  union {
    struct {
      int data_len[kMaxRecv];
      uint64_t data[kMaxRecv];
      struct FifoItem *elems;
      struct ibv_send_wr wr;
      struct ibv_sge sge;
      struct ibv_qp *qp;
    } recv;
    struct {
      size_t data_len;
      int inc_backlog;
      uint64_t laddr;
      uint64_t raddr;
      uint32_t lkey;
      uint32_t rkey;
      uint32_t rid;
      uint32_t sent_offset;
      uint32_t acked_bytes; // RC only.
    } send;
  };
  uint64_t rtt_tsc;
};
class ReqPool : public BufPool {
  static constexpr size_t nr_elements = kMaxReq << 2; // Send and receive.
  static constexpr size_t element_size = sizeof(Request);

public:
  ReqPool() : BufPool(nr_elements, element_size) {}
  ~ReqPool() = default;
};

} // namespace gnic
#pragma once

#ifdef NCCL_BUILD_MLX5DV
#include <infiniband/mlx5dv.h>
#else
#include "mlx5dvcore.h"
#endif

#include "mpib_compat.h"

struct ncclMlx5dvSymbols {
  bool (*mlx5dv_internal_is_supported)(struct ibv_device* device);
  int (*mlx5dv_internal_get_data_direct_sysfs_path)(struct ibv_context* context, char* buf, size_t buf_len);
  struct ibv_mr* (*mlx5dv_internal_reg_dmabuf_mr)(struct ibv_pd* pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access, int mlx5_access);
};

ncclResult_t buildMlx5dvSymbols(struct ncclMlx5dvSymbols* mlx5dvSymbols);

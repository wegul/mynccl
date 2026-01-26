#pragma once

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef NCCL_BUILD_MLX5DV
#include <infiniband/mlx5dv.h>
#else
#include "mlx5dvcore.h"
#endif

#include "mpib_compat.h"
#include "ibvwrap.h"

typedef enum mlx5dv_return_enum { MLX5DV_SUCCESS = 0 } mlx5dv_return_t;

ncclResult_t wrap_mlx5dv_symbols(void);
bool wrap_mlx5dv_is_supported(struct ibv_device* device);
ncclResult_t wrap_mlx5dv_get_data_direct_sysfs_path(struct ibv_context* context, char* buf, size_t buf_len);

ncclResult_t wrap_mlx5dv_reg_dmabuf_mr(struct ibv_mr** ret, struct ibv_pd* pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access, int mlx5_access);
struct ibv_mr* wrap_direct_mlx5dv_reg_dmabuf_mr(struct ibv_pd* pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access, int mlx5_access);

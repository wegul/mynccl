#pragma once

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
#include <unistd.h>

#include "ibvwrap.h"

enum mlx5dv_reg_dmabuf_access {
  MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT = (1 << 0),
};

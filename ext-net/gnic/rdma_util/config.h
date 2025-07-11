#pragma once

#include "common.h"

extern ncclDebugLogger_t logFunction;
// Logging macros following NCCL convention
#define ERROR(...)                                                             \
  logFunction(NCCL_LOG_ABORT, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define WARN(...)                                                              \
  logFunction(NCCL_LOG_WARN, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define INFO(FLAGS, ...)                                                       \
  logFunction(NCCL_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)

// Maximum number of outstanding receive messages in one recv request.
static constexpr uint32_t kMaxRecv = 1;
// Maximum number of outstanding receive requests in one engine.
static constexpr uint32_t kMaxReq = 128;
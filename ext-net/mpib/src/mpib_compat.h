#pragma once

#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <thread>
#include <unistd.h>

#include "common.h"
#include "net.h"

extern ncclDebugLogger_t mpibLogFunction;

static inline void mpibLog(ncclDebugLogLevel level, unsigned long flags,
                           const char *file, int line, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  if (mpibLogFunction) {
    char buffer[2048];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    mpibLogFunction(level, flags, file, line, "%s", buffer);
  } else {
    fprintf(stderr, "MPIB[%d] %s:%d: ", (int)level, file, line);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
  }
  va_end(args);
}

#define WARN(fmt, ...)                                                         \
  mpibLog(NCCL_LOG_WARN, NCCL_NET, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define INFO(flags, fmt, ...)                                                  \
  mpibLog(NCCL_LOG_INFO, (flags), __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define TRACE(flags, fmt, ...)                                                 \
  mpibLog(NCCL_LOG_TRACE, (flags), __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#define __public __attribute__((visibility("default")))
#define __hidden __attribute__((visibility("hidden")))

static inline bool mpibRateLimitEvery(std::atomic<uint64_t> *counter,
                                      uint64_t every) {
  if (every <= 1)
    return true;
  uint64_t v = counter->fetch_add(1, std::memory_order_relaxed);
  return (v % every) == 0;
}

// Rate-limited INFO for hot polling loops.
// One counter per callsite.
#define MPIB_RL_INFO(flags, every, fmt, ...)                                   \
  do {                                                                         \
    static std::atomic<uint64_t> _mpib_rl_counter{0};                          \
    if (mpibRateLimitEvery(&_mpib_rl_counter, (100 * every)))                  \
      INFO((flags), fmt, ##__VA_ARGS__);                                       \
  } while (0)

#define MPIB_CONVERT_ORDER(order)                                              \
  ((order) == std::memory_order_relaxed   ? __ATOMIC_RELAXED                   \
   : (order) == std::memory_order_consume ? __ATOMIC_CONSUME                   \
   : (order) == std::memory_order_acquire ? __ATOMIC_ACQUIRE                   \
   : (order) == std::memory_order_release ? __ATOMIC_RELEASE                   \
   : (order) == std::memory_order_acq_rel ? __ATOMIC_ACQ_REL                   \
                                          : __ATOMIC_SEQ_CST)

#define COMPILER_ATOMIC_LOAD(ptr, order)                                       \
  __atomic_load_n((ptr), MPIB_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_STORE(ptr, val, order)                                 \
  __atomic_store_n((ptr), (val), MPIB_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_FETCH_ADD(ptr, val, order)                             \
  __atomic_fetch_add((ptr), (val), MPIB_CONVERT_ORDER(order))
#define COMPILER_EXPECT(x, v) __builtin_expect((x), (v))

#define DIVUP(x, y) (((x) + (y) - 1) / (y))

#ifndef NCCL_NET_MR_FLAG_FORCE_SO
#define NCCL_NET_MR_FLAG_FORCE_SO 0x1
#endif

#define NCCLCHECK(call)                                                        \
  do {                                                                         \
    ncclResult_t RES = (call);                                                 \
    if (RES != ncclSuccess)                                                    \
      return RES;                                                              \
  } while (0)

#define NCCLCHECKGOTO(call, RES, label)                                        \
  do {                                                                         \
    RES = (call);                                                              \
    if (RES != ncclSuccess)                                                    \
      goto label;                                                              \
  } while (0)

#define SYSCHECK(statement, name)                                              \
  do {                                                                         \
    int retval = (statement);                                                  \
    if (retval == -1) {                                                        \
      WARN("Call to %s failed: %s", name, strerror(errno));                    \
      return ncclSystemError;                                                  \
    }                                                                          \
  } while (0)

#define SYSCHECKGOTO(statement, name, RES, label)                              \
  do {                                                                         \
    int retval = (statement);                                                  \
    if (retval == -1) {                                                        \
      WARN("Call to %s failed: %s", name, strerror(errno));                    \
      RES = ncclSystemError;                                                   \
      goto label;                                                              \
    }                                                                          \
  } while (0)

#define PTHREADCHECK(statement, name)                                          \
  do {                                                                         \
    int retval = (statement);                                                  \
    if (retval != 0) {                                                         \
      WARN("Call to %s failed: %s", name, strerror(retval));                   \
      return ncclSystemError;                                                  \
    }                                                                          \
  } while (0)

static inline ncclResult_t mpibSetThreadName(std::thread &th, const char *fmt,
                                             int id) {
  char name[16];
  snprintf(name, sizeof(name), fmt, id);
#if defined(__linux__)
  pthread_setname_np(th.native_handle(), name);
#else
  (void)th;
  (void)name;
#endif
  return ncclSuccess;
}

template <typename T> static inline ncclResult_t mpibCalloc(T **ptr, size_t n) {
  *ptr = (T *)calloc(n, sizeof(T));
  return *ptr ? ncclSuccess : ncclSystemError;
}

template <typename T>
static inline ncclResult_t mpibRealloc(T **ptr, size_t oldCount,
                                       size_t newCount) {
  (void)oldCount;
  *ptr = (T *)realloc(*ptr, newCount * sizeof(T));
  return *ptr ? ncclSuccess : ncclSystemError;
}

static inline ncclResult_t mpibMalloc(void **ptr, size_t size) {
  *ptr = malloc(size);
  if (*ptr == nullptr)
    return ncclSystemError;
  memset(*ptr, 0, size);
  return ncclSuccess;
}

#define TIME_START(x)                                                          \
  do {                                                                         \
    (void)(x);                                                                 \
  } while (0)
#define TIME_STOP(x)                                                           \
  do {                                                                         \
    (void)(x);                                                                 \
  } while (0)

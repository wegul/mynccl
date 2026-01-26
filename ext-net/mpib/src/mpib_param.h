#pragma once

#include <stdint.h>
#include <atomic>
#include <string>

const char* mpibGetEnv(const char* name);
void mpibLoadParam(const char* env, int64_t deftVal, int64_t uninitialized, int64_t* cache);

#define MPIB_PARAM(name, env, deftVal) \
  int64_t mpibParam##name() { \
    constexpr int64_t uninitialized = INT64_MIN; \
    static_assert(deftVal != uninitialized, "default value cannot be the uninitialized value."); \
    static int64_t cache = uninitialized; \
    if (__atomic_load_n(&cache, __ATOMIC_RELAXED) == uninitialized) { \
      mpibLoadParam("MPIB_" env, deftVal, uninitialized, &cache); \
    } \
    return cache; \
  }

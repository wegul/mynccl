#include "mpib_param.h"

#include <cstdlib>
#include <cerrno>
#include <cstring>

#include "mpib_compat.h"

const char* mpibGetEnv(const char* name) {
  return std::getenv(name);
}

void mpibLoadParam(const char* env, int64_t deftVal, int64_t uninitialized, int64_t* cache) {
  (void)uninitialized;
  const char* str = mpibGetEnv(env);
  int64_t value = deftVal;
  if (str && std::strlen(str) > 0) {
    errno = 0;
    value = std::strtoll(str, nullptr, 0);
    if (errno) {
      value = deftVal;
      INFO(NCCL_ENV, "Invalid value %s for %s, using default %lld.", str, env, (long long)deftVal);
    } else {
      INFO(NCCL_ENV, "%s set by environment to %lld.", env, (long long)value);
    }
  }
  __atomic_store_n(cache, value, __ATOMIC_RELAXED);
}

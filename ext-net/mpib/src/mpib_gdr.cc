#include "mpib_common.h"

ncclResult_t mpibGdrSupport() { return ncclInternalError; }

ncclResult_t mpibDmaBufSupport(int dev) {
  (void)dev;
  return ncclInvalidUsage;
}

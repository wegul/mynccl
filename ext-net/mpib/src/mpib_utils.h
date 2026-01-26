#pragma once

#include <stdint.h>
#include <stddef.h>
#include "mpib_compat.h"

struct mpibNetIf {
  char prefix[64];
  int port;
};

int mpibParseStringList(const char* string, struct mpibNetIf* ifList, int maxList);
bool mpibMatchIfList(const char* string, int port, struct mpibNetIf* ifList, int listSize, bool matchExact);

static inline ncclResult_t mpibGetRandomData(void* buffer, size_t bytes) {
  if (bytes == 0) return ncclSuccess;
  const size_t one = 1UL;
  FILE* fp = fopen("/dev/urandom", "r");
  if (buffer == nullptr || fp == nullptr || fread(buffer, bytes, one, fp) != one) {
    if (fp) fclose(fp);
    return ncclSystemError;
  }
  fclose(fp);
  return ncclSuccess;
}

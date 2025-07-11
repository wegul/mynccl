/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "utils.h"
#include <stdlib.h>

ncclResult_t int64ToBusId(int64_t id, char *busId) {
  sprintf(busId, "%04lx:%02lx:%02lx.%01lx", (id) >> 20, (id & 0xff000) >> 12,
          (id & 0xff0) >> 4, (id & 0xf));
  return ncclSuccess;
}

ncclResult_t busIdToInt64(const char *busId, int64_t *id) {
  char hexStr[17]; // Longest possible int64 hex string + null terminator.
  int hexOffset = 0;
  for (int i = 0; hexOffset < sizeof(hexStr) - 1; i++) {
    char c = busId[i];
    if (c == '.' || c == ':')
      continue;
    if ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') ||
        (c >= 'a' && c <= 'f')) {
      hexStr[hexOffset++] = busId[i];
    } else
      break;
  }
  hexStr[hexOffset] = '\0';
  *id = strtol(hexStr, NULL, 16);
  return ncclSuccess;
}

int parseStringList(const char *string, struct netIf *ifList, int maxList) {
  if (!string)
    return 0;

  const char *ptr = string;

  int ifNum = 0;
  int ifC = 0;
  char c;
  do {
    c = *ptr;
    if (c == ':') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = atoi(ptr + 1);
        ifNum++;
        ifC = 0;
      }
      while (c != ',' && c != '\0')
        c = *(++ptr);
    } else if (c == ',' || c == '\0') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = -1;
        ifNum++;
        ifC = 0;
      }
    } else {
      ifList[ifNum].prefix[ifC] = c;
      ifC++;
    }
    ptr++;
  } while (ifNum < maxList && c);
  return ifNum;
}

static bool matchIf(const char *string, const char *ref, bool matchExact) {
  // Make sure to include '\0' in the exact case
  int matchLen = matchExact ? strlen(string) + 1 : strlen(ref);
  return strncmp(string, ref, matchLen) == 0;
}

static bool matchPort(const int port1, const int port2) {
  if (port1 == -1)
    return true;
  if (port2 == -1)
    return true;
  if (port1 == port2)
    return true;
  return false;
}

bool matchIfList(const char *string, int port, struct netIf *ifList,
                 int listSize, bool matchExact) {
  // Make an exception for the case where no user list is defined
  if (listSize == 0)
    return true;

  for (int i = 0; i < listSize; i++) {
    if (matchIf(string, ifList[i].prefix, matchExact) &&
        matchPort(port, ifList[i].port)) {
      return true;
    }
  }
  return false;
}

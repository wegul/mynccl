#include "mpib_utils.h"

#include <cstring>
#include <cstdlib>

int mpibParseStringList(const char* string, struct mpibNetIf* ifList, int maxList) {
  if (!string) return 0;

  const char* ptr = string;
  int ifNum = 0;
  int ifC = 0;
  char c;
  do {
    c = *ptr;
    if (c == ':') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = atoi(ptr + 1);
        ifNum++; ifC = 0;
      }
      while (c != ',' && c != '\0') c = *(++ptr);
    } else if (c == ',' || c == '\0') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = -1;
        ifNum++; ifC = 0;
      }
    } else {
      ifList[ifNum].prefix[ifC] = c;
      ifC++;
    }
    ptr++;
  } while (ifNum < maxList && c);
  return ifNum;
}

static bool mpibMatchIf(const char* string, const char* ref, bool matchExact) {
  int matchLen = matchExact ? strlen(string) + 1 : strlen(ref);
  return strncmp(string, ref, matchLen) == 0;
}

static bool mpibMatchPort(const int port1, const int port2) {
  if (port1 == -1) return true;
  if (port2 == -1) return true;
  if (port1 == port2) return true;
  return false;
}

bool mpibMatchIfList(const char* string, int port, struct mpibNetIf* ifList, int listSize, bool matchExact) {
  if (listSize == 0) return true;
  for (int i = 0; i < listSize; i++) {
    if (mpibMatchIf(string, ifList[i].prefix, matchExact) && mpibMatchPort(port, ifList[i].port)) {
      return true;
    }
  }
  return false;
}

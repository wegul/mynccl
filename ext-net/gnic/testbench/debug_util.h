#include "net.h"
#include <chrono>
#include <cstdarg>
#include <cstring>
#include <iomanip>
#include <iostream>

extern const ncclNet_t ncclNetPlugin_v10;

// Mock logger function
void mock_logger(ncclDebugLogLevel level, unsigned long flags, const char *file,
                 int line, const char *fmt, ...) {
  (void)flags; // Suppress unused parameter warning

  // Check if we should print based on NCCL_DEBUG level
  const char *debug_level = "INFO";
  int should_print = 0;

  if (debug_level) {
    if (strcmp(debug_level, "TRACE") == 0) {
      should_print = 1; // Print everything
    } else if (strcmp(debug_level, "INFO") == 0 && level <= NCCL_LOG_INFO) {
      should_print = 1; // Print INFO and below
    } else if (strcmp(debug_level, "WARN") == 0 && level <= NCCL_LOG_WARN) {
      should_print = 1; // Print WARN and below
    }
  }

  if (!should_print)
    return;

  // Convert log level to string
  const char *level_str;
  switch (level) {
  case NCCL_LOG_NONE:
    level_str = "NONE";
    break;
  case NCCL_LOG_VERSION:
    level_str = "VERSION";
    break;
  case NCCL_LOG_WARN:
    level_str = "WARN";
    break;
  case NCCL_LOG_INFO:
    level_str = "INFO";
    break;
  case NCCL_LOG_ABORT:
    level_str = "ABORT";
    break;
  case NCCL_LOG_TRACE:
    level_str = "TRACE";
    break;
  default:
    level_str = "UNKNOWN";
    break;
  }

  // Print log header
  printf("[%s:%s:%d] ", level_str, file, line);

  // Print formatted message
  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);

  printf("\n");
}

void print_nccl_properties(const ncclNetProperties_t *props) {
  if (props == nullptr) {
    std::cout << "Properties pointer is NULL" << std::endl;
    return;
  }

  std::cout << "=== NCCL Network Properties ===" << std::endl;
  std::cout << "name: " << (props->name ? props->name : "NULL") << std::endl;
  std::cout << "pciPath: " << (props->pciPath ? props->pciPath : "NULL")
            << std::endl;
  std::cout << "guid: 0x" << std::hex << std::setw(16) << std::setfill('0')
            << props->guid << std::dec << std::endl;
  std::cout << "ptrSupport: " << props->ptrSupport << std::endl;
  std::cout << "regIsGlobal: " << props->regIsGlobal << std::endl;
  std::cout << "forceFlush: " << props->forceFlush << std::endl;
  std::cout << "speed: " << props->speed << " Mbps" << std::endl;
  std::cout << "port: " << props->port << std::endl;
  std::cout << "latency: " << props->latency << std::endl;
  std::cout << "maxComms: " << props->maxComms << std::endl;
  std::cout << "maxRecvs: " << props->maxRecvs << std::endl;
  std::cout << "netDeviceType: " << props->netDeviceType << std::endl;
  std::cout << "netDeviceVersion: " << props->netDeviceVersion << std::endl;
  std::cout << "maxP2pBytes: " << props->maxP2pBytes << std::endl;
  std::cout << "maxCollBytes: " << props->maxCollBytes << std::endl;
  std::cout << "===============================" << std::endl;
}

#define MAX_BUFFER_SIZE 128
// Logging macros
#define INFO(msg)                                                              \
  do {                                                                         \
    auto now = std::chrono::system_clock::now();                               \
    auto time_t = std::chrono::system_clock::to_time_t(now);                   \
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(           \
                  now.time_since_epoch()) %                                    \
              1000;                                                            \
    std::cout << "TB: [INFO] ["                                                \
              << std::put_time(std::localtime(&time_t), "%H:%M:%S") << "."     \
              << std::setfill('0') << std::setw(3) << ms.count() << "] "       \
              <<msg<< std::endl;                                           \
  } while (0)

#define WARN(msg)                                                              \
  do {                                                                         \
    auto now = std::chrono::system_clock::now();                               \
    auto time_t = std::chrono::system_clock::to_time_t(now);                   \
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(           \
                  now.time_since_epoch()) %                                    \
              1000;                                                            \
    std::cerr << "TB: [WARN] ["                                                \
              << std::put_time(std::localtime(&time_t), "%H:%M:%S") << "."     \
              << std::setfill('0') << std::setw(3) << ms.count() << "] "       \
              <<msg<< std::endl;                                           \
  } while (0)

#define ERR(msg)                                                               \
  do {                                                                         \
    auto now = std::chrono::system_clock::now();                               \
    auto time_t = std::chrono::system_clock::to_time_t(now);                   \
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(           \
                  now.time_since_epoch()) %                                    \
              1000;                                                            \
    std::cerr << "TB: [ERROR] ["                                               \
              << std::put_time(std::localtime(&time_t), "%H:%M:%S") << "."     \
              << std::setfill('0') << std::setw(3) << ms.count() << "] "       \
              <<msg<< std::endl;                                           \
  } while (0)

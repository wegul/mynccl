// mpib testbench: simple control server
// This file is a skeleton modeled after ext-net/gnic/testbench/ctrl_server.cpp.
// It is responsible only for exchanging opaque ncclNet connection handles
// between multiple endpoint_mpib processes.

#include <arpa/inet.h>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <netinet/in.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

// TODO: adjust backlog, address, and ports as needed.
static const int BACKLOG = 16;
static const char *kCtrlServerAddr = "127.0.0.1";
static uint32_t kCtrlServerPort = 8888;

// TODO: define the maximum handle/buffer size you will exchange.
// Typically this is NCCL_NET_HANDLE_MAXSIZE.
static const size_t kMaxHandleSize = 128; // placeholder

struct ClientInfo {
  int socket_fd;
  int client_id;
  std::vector<uint8_t> buffer;
  bool buffer_received;
};

int main(int argc, char **argv) {
  // TODO: parse command-line arguments for addr/port and number of clients.

  // TODO: create a TCP socket, bind to (kCtrlServerAddr, kCtrlServerPort),
  //       and listen with BACKLOG.

  // TODO: accept N incoming connections (e.g., N=2 for point-to-point tests).

  // TODO: for each client, recv exactly kMaxHandleSize bytes into its buffer.

  // TODO: once all handles are received, perform the desired exchange pattern:
  //       - simplest: swap client 0's handle with client 1's, etc.
  //       - or: broadcast all handles to all clients for multi-rank tests.

  // TODO: send the appropriate peer handle(s) back to each client.

  // TODO: close all sockets and exit.

  return 0;
}

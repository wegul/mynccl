#include "debug_util.h"
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

const int BACKLOG = 10;

const char *servAddr_str = "10.0.0.1";
uint32_t servPort = 8888;

struct ClientInfo {
  int socket_fd;
  int client_id;
  std::vector<uint8_t> buffer;
  bool buffer_received;

  ClientInfo() : socket_fd(-1), client_id(-1), buffer_received(false) {
    buffer.reserve(MAX_BUFFER_SIZE);
  }
};

class CtrlServer {
private:
  int server_socket;
  int expected_clients;
  std::vector<ClientInfo> clients;
  std::mutex clients_mutex; // This is to protect vector-clients
  std::condition_variable all_buffers_received;
  bool all_received;
  std::vector<std::thread> client_threads;

public:
  CtrlServer(int num_clients)
      : expected_clients(num_clients), all_received(false) {
    clients.resize(num_clients);
    client_threads.reserve(num_clients);
  }

  bool initialize() {
    // Create socket
    server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
      ERR("Error creating socket");
      return false;
    }

    // Set socket options to reuse address
    int opt = 1;
    if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) <
        0) {
      ERR("Error setting socket options");
      close(server_socket);
      return false;
    }

    // Bind socket
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(servAddr_str);
    server_addr.sin_port = htons(servPort);

    if (bind(server_socket, (struct sockaddr *)&server_addr,
             sizeof(server_addr)) < 0) {
      ERR("Error binding socket to " << servAddr_str << ":" << servPort);
      close(server_socket);
      return false;
    }

    // Listen for connections
    if (listen(server_socket, BACKLOG) < 0) {
      ERR("Error listening on socket");
      close(server_socket);
      return false;
    }

    INFO("Control server initialized on " << servAddr_str << ":" << servPort);
    INFO("Waiting for " << expected_clients << " clients");
    return true;
  }

  void handleClient(int client_socket, int client_id) {
    INFO("Handling client " << client_id);

    // Receive the actual buffer
    uint32_t buffer_size = MAX_BUFFER_SIZE;
    std::vector<uint8_t> buffer(buffer_size);
    ssize_t bytes_received =
        recv(client_socket, buffer.data(), buffer_size, MSG_WAITALL);
    if (bytes_received < 1) {
      ERR("Error receiving buffer from client " << client_id);
      close(client_socket); // Close socket on error
      return;
    }

    // Store client information
    {
      std::lock_guard<std::mutex> lock(clients_mutex);
      clients[client_id].socket_fd = client_socket;
      clients[client_id].client_id = client_id;
      clients[client_id].buffer = std::move(buffer);
      clients[client_id].buffer_received = true;

      INFO("Received " << buffer_size << " bytes from client " << client_id);
      // Check if all buffers have been received
      bool all_received_local = true;
      for (const auto &client : clients) {
        if (!client.buffer_received) {
          all_received_local = false;
          break;
        }
      }

      if (all_received_local) {
        all_received = true;
        all_buffers_received.notify_all();
      }
    }
  }

  void exchangeHandles() {
    INFO("All buffers received. Starting reciprocal exchange...");

    // Reciprocal exchange: server0 ↔ serverN, server1 ↔ serverN-1, etc.
    for (int i = 0; i < expected_clients; i++) {
      int partner_id = expected_clients - 1 - i;

      if (i < partner_id) { // Avoid duplicate exchanges
        INFO("Exchanging between client " << i << " and client " << partner_id);

        // Send client i's buffer to client partner_id
        sendBufferToClient(clients[partner_id].socket_fd, clients[i].buffer);

        // Send client partner_id's buffer to client i (if different clients)
        sendBufferToClient(clients[i].socket_fd, clients[partner_id].buffer);
      }
    }
    INFO("Reciprocal exchange completed!");
  }

  void sendBufferToClient(int client_socket,
                          const std::vector<uint8_t> &buffer) {
    send(client_socket, buffer.data(), buffer.size(), 0);
  }

  void acceptConnections() {
    for (int i = 0; i < expected_clients; i++) {
      struct sockaddr_in client_addr;
      socklen_t client_len = sizeof(client_addr);

      int client_socket =
          accept(server_socket, (struct sockaddr *)&client_addr, &client_len);
      if (client_socket < 0) {
        ERR("Error accepting connection");
        continue;
      }

      // Client ID is ordered as sequence of acceptance
      uint32_t client_id = i;
      INFO("Client <" << client_id << "> connected from "
                      << inet_ntoa(client_addr.sin_addr) << ":"
                      << ntohs(client_addr.sin_port));

      // Handle client in a separate thread (store thread instead of detaching)
      client_threads.emplace_back(&CtrlServer::handleClient, this,
                                  client_socket, client_id);
    }
  }

  void run() {
    acceptConnections();

    // Sync-blocking. Wait for all buffers to be received
    std::unique_lock<std::mutex> lock(clients_mutex);
    all_buffers_received.wait(lock, [this] { return all_received; });
    // Only one thread should perform the exchange
    static std::once_flag exchange_flag;
    std::call_once(exchange_flag, [this]() { exchangeHandles(); });

    // Close all client sockets
    for (auto &client : clients) {
      if (client.socket_fd != -1) {
        close(client.socket_fd);
      }
    }

    close(server_socket);
    server_socket = -1; // Mark as closed to prevent double close in destructor
    INFO("Control server shutting down.");
  }

  ~CtrlServer() {
    // Ensure all threads are joined before destruction
    for (auto &thread : client_threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }

    if (server_socket != -1) {
      close(server_socket);
      server_socket = -1;
    }
  }
};

int main() {
  int num_clients = 2; // Default number of clients

  INFO("Starting control server for " << num_clients << " clients");

  CtrlServer server(num_clients);

  if (!server.initialize()) {
    ERR("Failed to initialize server");
    return 1;
  }

  server.run();

  return 0;
}
/*
 * MPIB Transport Agent - IPC (Unix Socket) Handling
 *
 * Handles registration/deregistration messages over Unix socket.
 * Manages slot allocation with refcounting.
 */

#include "ipc.h"
#include "../include/mpib_agent_iface.h"
#include "shm.h"

#include <cstdio>
#include <cstring>
#include <errno.h>
#include <map>
#include <mutex>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

/* ============================================================================
 * Global State
 * ============================================================================
 */

static int g_listen_fd = -1;

/* Slot management */
struct SlotInfo {
  uint32_t src_ip;
  uint32_t dst_ip;
  int refcount;
};

static std::map<uint64_t, uint32_t> g_flow_to_slot; /* (src<<32|dst) -> slot */
static std::map<uint32_t, SlotInfo> g_slots;        /* slot -> info */
static std::map<uint32_t, uint32_t> g_conn_to_slot; /* conn_id -> slot */
static uint32_t g_next_slot = 0;
static std::mutex g_slot_mutex;

/* ============================================================================
 * Helper Functions
 * ============================================================================
 */

static int recv_all(int fd, void *buf, size_t len) {
  uint8_t *p = (uint8_t *)buf;
  size_t recvd = 0;
  while (recvd < len) {
    ssize_t n = recv(fd, p + recvd, len - recvd, 0);
    if (n < 0) {
      if (errno == EINTR)
        continue;
      return -1;
    }
    if (n == 0)
      return -1; /* Connection closed */
    recvd += n;
  }
  return 0;
}

static int send_all(int fd, const void *buf, size_t len) {
  const uint8_t *p = (const uint8_t *)buf;
  size_t sent = 0;
  while (sent < len) {
    ssize_t n = send(fd, p + sent, len - sent, 0);
    if (n < 0) {
      if (errno == EINTR)
        continue;
      return -1;
    }
    sent += n;
  }
  return 0;
}

/* ============================================================================
 * Socket Management
 * ============================================================================
 */

int ipc_create_socket(void) {
  /* Remove old socket if exists */
  unlink(MPIB_SOCK_PATH);

  g_listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (g_listen_fd < 0) {
    fprintf(stderr, "Failed to create socket: %s\n", strerror(errno));
    return -1;
  }

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, MPIB_SOCK_PATH, sizeof(addr.sun_path) - 1);

  if (bind(g_listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    fprintf(stderr, "Failed to bind %s: %s\n", MPIB_SOCK_PATH, strerror(errno));
    close(g_listen_fd);
    return -1;
  }

  if (listen(g_listen_fd, 16) < 0) {
    fprintf(stderr, "Failed to listen: %s\n", strerror(errno));
    close(g_listen_fd);
    unlink(MPIB_SOCK_PATH);
    return -1;
  }

  printf("Listening on %s\n", MPIB_SOCK_PATH);
  return 0;
}

void ipc_destroy_socket(void) {
  if (g_listen_fd >= 0) {
    close(g_listen_fd);
    g_listen_fd = -1;
  }
  unlink(MPIB_SOCK_PATH);
}

int ipc_get_listen_fd(void) { return g_listen_fd; }

/* ============================================================================
 * Slot Management
 * ============================================================================
 */

uint32_t slot_find_or_allocate(uint32_t src_ip, uint32_t dst_ip,
                               uint32_t initial_bw) {
  std::lock_guard<std::mutex> lock(g_slot_mutex);

  uint64_t key = ((uint64_t)src_ip << 32) | dst_ip;

  auto it = g_flow_to_slot.find(key);
  if (it != g_flow_to_slot.end()) {
    /* Existing slot, increment refcount */
    uint32_t slot = it->second;
    g_slots[slot].refcount++;
    printf("Reusing slot %u for flow %08x->%08x (refcount=%d)\n", slot, src_ip,
           dst_ip, g_slots[slot].refcount);
    return slot;
  }

  /* Allocate new slot */
  if (g_next_slot >= MPIB_HINT_MAX_ENTRIES) {
    fprintf(stderr, "No free slots!\n");
    return UINT32_MAX;
  }

  uint32_t slot = g_next_slot++;
  g_flow_to_slot[key] = slot;
  g_slots[slot] = {src_ip, dst_ip, 1};

  /* Initialize the SHM entry */
  struct mpib_hint_shm *shm = shm_get();
  if (shm) {
    shm->entries[slot].src_ip = src_ip;
    shm->entries[slot].dst_ip = dst_ip;
    mpib_hint_write(&shm->entries[slot], initial_bw);
  }

  printf("Allocated slot %u for flow %08x->%08x\n", slot, src_ip, dst_ip);
  return slot;
}

void slot_release(uint32_t conn_id) {
  std::lock_guard<std::mutex> lock(g_slot_mutex);

  auto it = g_conn_to_slot.find(conn_id);
  if (it == g_conn_to_slot.end()) {
    printf("Deregister: conn_id 0x%08x not found\n", conn_id);
    return;
  }

  uint32_t slot = it->second;
  g_conn_to_slot.erase(it);

  auto slot_it = g_slots.find(slot);
  if (slot_it == g_slots.end()) {
    return;
  }

  slot_it->second.refcount--;
  printf("Released slot %u (refcount=%d)\n", slot, slot_it->second.refcount);

  if (slot_it->second.refcount <= 0) {
    uint64_t key =
        ((uint64_t)slot_it->second.src_ip << 32) | slot_it->second.dst_ip;
    g_flow_to_slot.erase(key);
    g_slots.erase(slot_it);
    printf("Freed slot %u\n", slot);
  }
}

void slot_track_conn(uint32_t conn_id, uint32_t slot) {
  std::lock_guard<std::mutex> lock(g_slot_mutex);
  g_conn_to_slot[conn_id] = slot;
}

void slot_for_each_active(void (*callback)(uint32_t slot, void *ctx),
                          void *ctx) {
  std::lock_guard<std::mutex> lock(g_slot_mutex);
  for (auto &kv : g_slots) {
    callback(kv.first, ctx);
  }
}

/* ============================================================================
 * Client Handling
 * ============================================================================
 */

/* Forward declaration - initial bw value comes from policy module */
extern uint32_t policy_get_initial_bw(void);

void ipc_handle_client(int client_fd) {
  /* Read header first */
  struct mpib_reg_header header;
  if (recv_all(client_fd, &header, sizeof(header)) < 0) {
    close(client_fd);
    return;
  }

  if (header.magic != MPIB_REG_MAGIC) {
    fprintf(stderr, "Invalid magic: 0x%08x\n", header.magic);
    close(client_fd);
    return;
  }

  if (header.msg_type == MPIB_MSG_REGISTER) {
    /* Read rest of register request */
    struct mpib_register_request req;
    req.header = header;
    size_t remaining = MPIB_REGISTER_REQUEST_SIZE - sizeof(header);
    if (recv_all(client_fd, ((uint8_t *)&req) + sizeof(header), remaining) <
        0) {
      close(client_fd);
      return;
    }

    printf("Register: conn_id=0x%08x sout=%08x->%08x sup=%08x->%08x\n",
           req.conn_id, req.sout_src_ip, req.sout_dst_ip, req.sup_src_ip,
           req.sup_dst_ip);

    /* Find or allocate slot based on SOUT flow (per design doc) */
    uint32_t slot = slot_find_or_allocate(req.sout_src_ip, req.sout_dst_ip,
                                          policy_get_initial_bw());

    /* Track conn_id -> slot mapping */
    slot_track_conn(req.conn_id, slot);

    /* Send response */
    struct mpib_register_response resp;
    memset(&resp, 0, sizeof(resp));
    resp.header.magic = MPIB_REG_MAGIC;
    resp.header.msg_type = MPIB_MSG_RESPONSE;
    resp.header.msg_len = MPIB_REGISTER_RESPONSE_SIZE;
    resp.status =
        (slot != UINT32_MAX) ? MPIB_STATUS_SUCCESS : MPIB_STATUS_NO_FREE_SLOTS;
    resp.hint_slot = slot;

    send_all(client_fd, &resp, sizeof(resp));

  } else if (header.msg_type == MPIB_MSG_DEREGISTER) {
    /* Read rest of deregister request */
    struct mpib_deregister_request req;
    req.header = header;
    size_t remaining = MPIB_DEREGISTER_REQUEST_SIZE - sizeof(header);
    if (recv_all(client_fd, ((uint8_t *)&req) + sizeof(header), remaining) <
        0) {
      close(client_fd);
      return;
    }

    printf("Deregister: conn_id=0x%08x\n", req.conn_id);
    slot_release(req.conn_id);

    /* Send response */
    struct mpib_deregister_response resp;
    memset(&resp, 0, sizeof(resp));
    resp.header.magic = MPIB_REG_MAGIC;
    resp.header.msg_type = MPIB_MSG_RESPONSE;
    resp.header.msg_len = MPIB_DEREGISTER_RESPONSE_SIZE;
    resp.status = MPIB_STATUS_SUCCESS;

    send_all(client_fd, &resp, sizeof(resp));

  } else {
    fprintf(stderr, "Unknown message type: %d\n", header.msg_type);
  }

  close(client_fd);
}

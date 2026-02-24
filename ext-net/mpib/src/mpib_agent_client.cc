/*
 * MPIB Agent Client Implementation
 *
 * Handles communication between MPIB plugin and the Transport Agent:
 *   - Shared memory mapping for hint reads (lock-free hot path)
 *   - Unix socket IPC for registration (cold path)
 */

#include "mpib_agent_client.h"

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

/* Include the shared interface definitions */
#include "../include/mpib_agent_iface.h"

/* ============================================================================
 * Global State
 * ============================================================================
 */

static volatile struct mpib_hint_shm *g_mpib_hint_shm = NULL;
static int g_mpib_hint_fd = -1;

/* ============================================================================
 * SHM Initialization / Finalization
 * ============================================================================
 */

ncclResult_t mpibAgentClientInit(void) {
  if (g_mpib_hint_shm != NULL) {
    /* Already initialized */
    return ncclSuccess;
  }

  /* Open the hint file */
  g_mpib_hint_fd = open(MPIB_HINT_PATH, O_RDONLY);
  if (g_mpib_hint_fd < 0) {
    WARN("NET/MPIB : Failed to open hint SHM %s: %s", MPIB_HINT_PATH,
         strerror(errno));
    return ncclSystemError;
  }

  /* Map the shared memory */
  void *ptr =
      mmap(NULL, MPIB_HINT_SHM_SIZE, PROT_READ, MAP_SHARED, g_mpib_hint_fd, 0);
  if (ptr == MAP_FAILED) {
    WARN("NET/MPIB : Failed to mmap hint SHM: %s", strerror(errno));
    close(g_mpib_hint_fd);
    g_mpib_hint_fd = -1;
    return ncclSystemError;
  }

  g_mpib_hint_shm = (volatile struct mpib_hint_shm *)ptr;

  /* Validate magic */
  if (g_mpib_hint_shm->header.magic != MPIB_HINT_MAGIC) {
    WARN("NET/MPIB : Invalid hint SHM magic: 0x%08x (expected 0x%08x)",
         g_mpib_hint_shm->header.magic, MPIB_HINT_MAGIC);
    munmap((void *)g_mpib_hint_shm, MPIB_HINT_SHM_SIZE);
    close(g_mpib_hint_fd);
    g_mpib_hint_shm = NULL;
    g_mpib_hint_fd = -1;
    return ncclSystemError;
  }

  INFO(NCCL_INIT | NCCL_NET, "NET/MPIB : Hint SHM mapped at %p, max_entries=%u",
       g_mpib_hint_shm, g_mpib_hint_shm->header.max_entries);

  return ncclSuccess;
}

ncclResult_t mpibAgentClientDestroy(void) {
  if (g_mpib_hint_shm != NULL) {
    munmap((void *)g_mpib_hint_shm, MPIB_HINT_SHM_SIZE);
    g_mpib_hint_shm = NULL;
  }
  if (g_mpib_hint_fd >= 0) {
    close(g_mpib_hint_fd);
    g_mpib_hint_fd = -1;
  }
  return ncclSuccess;
}

int mpibAgentClientIsInitialized(void) { return (g_mpib_hint_shm != NULL); }

/* ============================================================================
 * Unix Socket Helpers
 * ============================================================================
 */

static int mpibAgentSocketConnect(void) {
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    WARN("NET/MPIB : Failed to create agent socket: %s", strerror(errno));
    return -1;
  }

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, MPIB_SOCK_PATH, sizeof(addr.sun_path) - 1);

  if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    WARN("NET/MPIB : Failed to connect to agent at %s: %s", MPIB_SOCK_PATH,
         strerror(errno));
    close(fd);
    return -1;
  }

  return fd;
}

static ncclResult_t mpibAgentSendAll(int fd, const void *buf, size_t len) {
  const uint8_t *p = (const uint8_t *)buf;
  size_t sent = 0;
  while (sent < len) {
    ssize_t n = send(fd, p + sent, len - sent, 0);
    if (n < 0) {
      if (errno == EINTR)
        continue;
      WARN("NET/MPIB : Agent send failed: %s", strerror(errno));
      return ncclSystemError;
    }
    sent += n;
  }
  return ncclSuccess;
}

static ncclResult_t mpibAgentRecvAll(int fd, void *buf, size_t len) {
  uint8_t *p = (uint8_t *)buf;
  size_t recvd = 0;
  while (recvd < len) {
    ssize_t n = recv(fd, p + recvd, len - recvd, 0);
    if (n < 0) {
      if (errno == EINTR)
        continue;
      WARN("NET/MPIB : Agent recv failed: %s", strerror(errno));
      return ncclSystemError;
    }
    if (n == 0) {
      WARN("NET/MPIB : Agent closed connection unexpectedly");
      return ncclSystemError;
    }
    recvd += n;
  }
  return ncclSuccess;
}

/* ============================================================================
 * Registration / Deregistration
 * ============================================================================
 */

ncclResult_t mpibAgentRegister(uint32_t conn_id, uint32_t sout_src_ip,
                               uint32_t sout_dst_ip, uint32_t sup_src_ip,
                               uint32_t sup_dst_ip, uint32_t *hint_slot) {
  int fd = mpibAgentSocketConnect();
  if (fd < 0) {
    return ncclSystemError;
  }

  /* Build register request */
  struct mpib_register_request req;
  memset(&req, 0, sizeof(req));
  req.header.magic = MPIB_REG_MAGIC;
  req.header.msg_type = MPIB_MSG_REGISTER;
  req.header.msg_len = MPIB_REGISTER_REQUEST_SIZE;
  req.conn_id = conn_id;
  req.sout_src_ip = sout_src_ip;
  req.sout_dst_ip = sout_dst_ip;
  req.sup_src_ip = sup_src_ip;
  req.sup_dst_ip = sup_dst_ip;

  /* Send request */
  ncclResult_t ret = mpibAgentSendAll(fd, &req, sizeof(req));
  if (ret != ncclSuccess) {
    close(fd);
    return ret;
  }

  /* Receive response */
  struct mpib_register_response resp;
  ret = mpibAgentRecvAll(fd, &resp, sizeof(resp));
  close(fd);
  if (ret != ncclSuccess) {
    return ret;
  }

  /* Validate response */
  if (resp.header.magic != MPIB_REG_MAGIC ||
      resp.header.msg_type != MPIB_MSG_RESPONSE) {
    WARN("NET/MPIB : Invalid register response from agent");
    return ncclSystemError;
  }

  if (resp.status != MPIB_STATUS_SUCCESS) {
    WARN("NET/MPIB : Agent registration failed with status %d", resp.status);
    return ncclSystemError;
  }

  *hint_slot = resp.hint_slot;

  return ncclSuccess;
}

ncclResult_t mpibAgentDeregister(uint32_t conn_id) {
  int fd = mpibAgentSocketConnect();
  if (fd < 0) {
    /* Best-effort: don't fail if agent is gone */
    return ncclSuccess;
  }

  /* Build deregister request */
  struct mpib_deregister_request req;
  memset(&req, 0, sizeof(req));
  req.header.magic = MPIB_REG_MAGIC;
  req.header.msg_type = MPIB_MSG_DEREGISTER;
  req.header.msg_len = MPIB_DEREGISTER_REQUEST_SIZE;
  req.conn_id = conn_id;

  /* Send request */
  ncclResult_t ret = mpibAgentSendAll(fd, &req, sizeof(req));
  if (ret != ncclSuccess) {
    close(fd);
    return ncclSuccess; /* Best-effort */
  }

  /* Receive response */
  struct mpib_deregister_response resp;
  ret = mpibAgentRecvAll(fd, &resp, sizeof(resp));
  close(fd);

  if (ret == ncclSuccess) {
    INFO(NCCL_NET, "NET/MPIB : Deregistered conn_id=0x%08x", conn_id);
  }

  return ncclSuccess; /* Best-effort */
}

/* ============================================================================
 * Hint Reading (Hot Path)
 * ============================================================================
 */

uint32_t mpibAgentReadHint(uint32_t hint_slot) {
  return mpib_hint_read(&g_mpib_hint_shm->entries[hint_slot]);
}

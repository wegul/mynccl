/*
 * MPIB Agent Interface Definitions
 *
 * Shared between the MPIB NetPlugin and the Transport Agent daemon.
 * Defines:
 *   - Hint shared memory layout (SHM)
 *   - Registration IPC message formats (Unix socket)
 *
 * See doc/phase1_plan.md for detailed design.
 */

#ifndef MPIB_AGENT_IFACE_H_
#define MPIB_AGENT_IFACE_H_

#include <stdatomic.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Path Definitions
 * ============================================================================
 */

#define MPIB_HINT_DIR "/tmp/mpib"
#define MPIB_HINT_PATH "/tmp/mpib/hints"
#define MPIB_SOCK_PATH "/tmp/mpib/agent.sock"

/* ============================================================================
 * Hint Shared Memory Format
 * ============================================================================
 *
 * Layout:
 *   [Header: 16 bytes]
 *   [Entry[0]: 16 bytes]
 *   [Entry[1]: 16 bytes]
 *   ...
 *   [Entry[255]: 16 bytes]
 *
 * Total size: 16 + 256 * 16 = 4112 bytes
 */

#define MPIB_HINT_MAGIC 0x4D504948 /* "MPIH" in little-endian */
#define MPIB_HINT_MAX_ENTRIES 256

struct alignas(16) mpib_hint_header {
  uint32_t magic;       /* Must be MPIB_HINT_MAGIC */
  uint32_t max_entries; /* Number of entry slots (256) */
};

/*
 * Hint entry for a single flow (identified by SOUT src/dst IP pair).
 *
 * - sup_bw: Unsigned scale factor of SOUT bandwidth.
 *   Interpreted as: sup_bw * sout_bw.
 *   Plugin computes: sup_ratio = sup_bw / (1 + sup_bw)
 *
 * - seq: Sequence number for seqlock-style consistency.
 *   Agent: increment to odd before write, even after write.
 *   Plugin: retry if seq is odd or changed during read.
 *
 * - src_ip, dst_ip: SOUT path IP addresses (network byte order).
 *   All connections with same src-dst share this entry.
 */
struct mpib_hint_entry {
  uint32_t sup_bw; /* SUP bandwidth multiplier of SOUT */
  uint32_t seq;    /* Sequence number for consistency */
  uint32_t src_ip; /* SOUT source IP (network byte order) */
  uint32_t dst_ip; /* SOUT destination IP (network byte order) */
};

struct mpib_hint_shm {
  struct mpib_hint_header header;
  struct mpib_hint_entry entries[MPIB_HINT_MAX_ENTRIES];
};

#define MPIB_HINT_SHM_SIZE (sizeof(struct mpib_hint_shm))

/* ============================================================================
 * Hint Read Helper (Seqlock Pattern)
 * ============================================================================
 *
 * Lock-free read with consistency check:
 *   - Returns sup_bw on success
 *   - Retries if write is in progress (seq odd) or seq changed during read
 */
static inline uint32_t
mpib_hint_read(const volatile struct mpib_hint_entry *entry) {
  uint32_t seq1 = 0, seq2 = 0;
  uint32_t bw = 0;
  do {
    seq1 = __atomic_load_n(&entry->seq, __ATOMIC_ACQUIRE);
    if (seq1 & 1) {
      /* Write in progress, spin */
      continue;
    }
    bw = entry->sup_bw;
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    seq2 = __atomic_load_n(&entry->seq, __ATOMIC_ACQUIRE);
  } while (seq1 != seq2);
  return bw;
}

/* ============================================================================
 * Hint Write Helper (Agent Side)
 * ============================================================================
 */
static inline void mpib_hint_write(volatile struct mpib_hint_entry *entry,
                                   uint32_t sup_bw) {
  /* Increment to odd: write in progress */
  __atomic_fetch_add(&entry->seq, 1, __ATOMIC_RELEASE);
  __atomic_thread_fence(__ATOMIC_RELEASE);

  entry->sup_bw = sup_bw;

  __atomic_thread_fence(__ATOMIC_RELEASE);
  /* Increment to even: write complete */
  __atomic_fetch_add(&entry->seq, 1, __ATOMIC_RELEASE);
}

/* ============================================================================
 * Registry IPC Format (Unix Domain Socket)
 * ============================================================================
 *
 * Message structure:
 *   [Header: 8 bytes]
 *   [Payload: variable]
 */

#define MPIB_REG_MAGIC 0x4D504952 /* "MPIR" in little-endian */

/* Message types */
#define MPIB_MSG_REGISTER 1
#define MPIB_MSG_DEREGISTER 2
#define MPIB_MSG_RESPONSE 128

/* Response status codes */
#define MPIB_STATUS_SUCCESS 0
#define MPIB_STATUS_NO_FREE_SLOTS 1
#define MPIB_STATUS_INVALID_REQ 2

/*
 * Common message header (8 bytes)
 * Note: Fields ordered to avoid padding
 */
struct __attribute__((packed)) mpib_reg_header {
  uint32_t magic;   /* Must be MPIB_REG_MAGIC */
  uint16_t msg_len; /* Total message length including header */
  uint8_t msg_type; /* MPIB_MSG_* */
  uint8_t reserved;
};

/*
 * Register request (msg_type = MPIB_MSG_REGISTER)
 * Total: 8 (header) + 20 (payload) = 28 bytes
 */
struct __attribute__((packed)) mpib_register_request {
  struct mpib_reg_header header;
  uint32_t conn_id;     /* PID << 16 | counter */
  uint32_t sout_src_ip; /* SOUT local IP (network byte order) */
  uint32_t sout_dst_ip; /* SOUT remote IP (network byte order) */
  uint32_t sup_src_ip;  /* SUP local IP (network byte order) */
  uint32_t sup_dst_ip;  /* SUP remote IP (network byte order) */
};

#define MPIB_REGISTER_REQUEST_SIZE sizeof(struct mpib_register_request)

/*
 * Register response (msg_type = MPIB_MSG_RESPONSE after register)
 * Total: 8 (header) + 8 (payload) = 16 bytes
 */
struct __attribute__((packed)) mpib_register_response {
  struct mpib_reg_header header;
  uint8_t status; /* MPIB_STATUS_* */
  uint8_t reserved[3];
  uint32_t hint_slot; /* Index into SHM entry array */
};

#define MPIB_REGISTER_RESPONSE_SIZE sizeof(struct mpib_register_response)

/*
 * Deregister request (msg_type = MPIB_MSG_DEREGISTER)
 * Total: 8 (header) + 4 (payload) = 12 bytes
 */
struct __attribute__((packed)) mpib_deregister_request {
  struct mpib_reg_header header;
  uint32_t conn_id;
};

#define MPIB_DEREGISTER_REQUEST_SIZE sizeof(struct mpib_deregister_request)

/*
 * Deregister response (msg_type = MPIB_MSG_RESPONSE after deregister)
 * Total: 8 (header) + 4 (payload) = 12 bytes
 */
struct __attribute__((packed)) mpib_deregister_response {
  struct mpib_reg_header header;
  uint8_t status; /* MPIB_STATUS_* */
  uint8_t reserved[3];
};

#define MPIB_DEREGISTER_RESPONSE_SIZE sizeof(struct mpib_deregister_response)

/* ============================================================================
 * Helper Macros
 * ============================================================================
 */

/* Build conn_id from PID and counter */
#define MPIB_MAKE_CONN_ID(pid, counter)                                        \
  (((uint32_t)(pid) << 16) | ((uint32_t)(counter) & 0xFFFF))

/* Extract PID from conn_id */
#define MPIB_CONN_ID_PID(conn_id) ((uint32_t)(conn_id) >> 16)

/* Extract counter from conn_id */
#define MPIB_CONN_ID_COUNTER(conn_id) ((uint32_t)(conn_id) & 0xFFFF)

#ifdef __cplusplus
}
#endif

#endif /* MPIB_AGENT_IFACE_H_ */

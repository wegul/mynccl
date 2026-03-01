/*
 * MPIB Agent Client
 *
 * Provides functions for the MPIB plugin to:
 *   - Open/close the hint shared memory
 *   - Register/deregister connections with the agent
 *
 * See doc/phase1_plan.md for design details.
 */

#ifndef MPIB_AGENT_CLIENT_H_
#define MPIB_AGENT_CLIENT_H_

#include "mpib_compat.h"
#include <stddef.h>
#include <stdint.h>

/* Forward declaration for mpibGetSupBw */
struct mpibSendComm;

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize agent client: open and mmap the hint shared memory.
 *
 * Must be called before any mpibAgentRegister/mpibGetSupBw calls.
 * Returns ncclSystemError if agent is not running (SHM doesn't exist).
 */
ncclResult_t mpibAgentClientInit(void);

/*
 * Destroy agent client: munmap the hint shared memory.
 */
ncclResult_t mpibAgentClientDestroy(void);

/*
 * Register a connection with the agent.
 *
 * Sends registration request via Unix socket, receives hint_slot.
 * The hint_slot indexes into the shared memory entries array.
 *
 * @param conn_id      Unique connection ID (PID << 16 | counter)
 * @param sout_src_ip  SOUT local IP (network byte order)
 * @param sout_dst_ip  SOUT remote IP (network byte order)
 * @param sup_src_ip   SUP local IP (network byte order)
 * @param sup_dst_ip   SUP remote IP (network byte order)
 * @param hint_slot    [out] Assigned slot index in SHM
 *
 * Returns ncclSystemError if agent is unavailable.
 */
ncclResult_t mpibAgentRegister(uint32_t conn_id, uint32_t sout_src_ip,
                               uint32_t sout_dst_ip, uint32_t sup_src_ip,
                               uint32_t sup_dst_ip, uint32_t *hint_slot);

/*
 * Deregister a connection from the agent.
 *
 * @param conn_id  Connection ID used during registration
 *
 * Returns ncclSuccess even if agent is unavailable (best-effort).
 */
ncclResult_t mpibAgentDeregister(uint32_t conn_id);

/*
 * Determine SUP bandwidth share for a single transfer.
 *
 * Encapsulates the full policy decision:
 *   - In vanilla mode (MPIB_MODE=0): returns pure topology-driven value
 *     (UINT32_MAX for intra-island, 0 for inter-island). No SHM read.
 *   - In advanced mode (MPIB_MODE=1): reads agent hint from SHM via seqlock.
 *
 * @param comm  Send communicator (carries pathClass and hint_slot)
 * @param size  Transfer size in bytes (for future BDP threshold)
 *
 * Returns parts-per-1024 value, or sentinel (0 / UINT32_MAX).
 */
uint32_t mpibGetSupBw(struct mpibSendComm *comm, size_t size);

/*
 * Check if agent client is initialized.
 */
int mpibAgentClientIsInitialized(void);

#ifdef __cplusplus
}
#endif

#endif /* MPIB_AGENT_CLIENT_H_ */

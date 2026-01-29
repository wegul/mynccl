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
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize agent client: open and mmap the hint shared memory.
 *
 * Must be called before any mpibAgentRegister/mpibAgentReadHint calls.
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
 * Read the SUP bandwidth hint for a given slot.
 *
 * Uses seqlock pattern for consistency.
 *
 * @param hint_slot  Slot index from registration
 *
 * Returns an unsigned multiplier of SOUT bandwidth.
 *   - 0 => all SOUT
 *   - 1 => equal split (SUP = SOUT)
 *   - 2 => SUP gets 2x SOUT bandwidth share (SUP ratio = 2/3)
 */
uint32_t mpibAgentReadHint(uint32_t hint_slot);

/*
 * Check if agent client is initialized.
 */
int mpibAgentClientIsInitialized(void);

#ifdef __cplusplus
}
#endif

#endif /* MPIB_AGENT_CLIENT_H_ */

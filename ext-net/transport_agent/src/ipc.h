/*
 * MPIB Transport Agent - IPC (Unix Socket) Handling
 *
 * Provides functions for registration/deregistration via Unix socket.
 */

#ifndef MPIB_AGENT_IPC_H
#define MPIB_AGENT_IPC_H

#include <cstdint>

/* Create the listen socket at /tmp/mpib/agent.sock */
int ipc_create_socket(void);

/* Destroy the listen socket */
void ipc_destroy_socket(void);

/* Get the listen socket fd for poll() */
int ipc_get_listen_fd(void);

/* Handle a new client connection (blocking call) */
void ipc_handle_client(int client_fd);

/*
 * Slot management API used by ipc.cc internally and policy.cc
 */

/* Find or allocate a slot for a flow, increment refcount */
uint32_t slot_find_or_allocate(uint32_t src_ip, uint32_t dst_ip,
                               uint32_t initial_bw);

/* Release a slot by conn_id, decrement refcount */
void slot_release(uint32_t conn_id);

/* Track a conn_id -> slot mapping */
void slot_track_conn(uint32_t conn_id, uint32_t slot);

/* Get list of active slots for hint updates */
void slot_for_each_active(void (*callback)(uint32_t slot, void *ctx),
                          void *ctx);

#endif /* MPIB_AGENT_IPC_H */

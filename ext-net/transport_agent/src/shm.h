/*
 * MPIB Transport Agent - Shared Memory Management
 *
 * Provides functions to create, initialize, and destroy the hint SHM.
 */

#ifndef MPIB_AGENT_SHM_H
#define MPIB_AGENT_SHM_H

#include "../include/mpib_agent_iface.h"

/* Get the global SHM pointer (set after shm_create) */
struct mpib_hint_shm *shm_get(void);

/* Create and initialize the hint SHM file */
int shm_create(uint32_t initial_bw);

/* Destroy the SHM file and unmap memory */
void shm_destroy(void);

/* Update all active entries with new bandwidth */
void shm_update_all(uint32_t bw);

#endif /* MPIB_AGENT_SHM_H */

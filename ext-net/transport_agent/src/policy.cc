/*
 * MPIB Transport Agent - Simulated Policy Logic
 *
 * Manual policy for testing:
 *   --manual           : Read "slot bw" or "all bw" commands
 */

#include "policy.h"
#include "../include/mpib_agent_iface.h"
#include "ipc.h"
#include "shm.h"

#include <cstdio>
#include <cstring>

/* ============================================================================
 * Global State
 * ============================================================================
 */

static uint32_t g_static_bw = 0;

/* ============================================================================
 * Helper Functions
 * ============================================================================
 */

static void update_slot_callback(uint32_t slot, void *ctx) {
  uint32_t bw = *(uint32_t *)ctx;
  struct mpib_hint_shm *shm = shm_get();
  if (shm) {
    mpib_hint_write(&shm->entries[slot], bw);
  }
}

static void update_all_hints(uint32_t bw) {
  slot_for_each_active(update_slot_callback, &bw);
}

/* ==========================================================================
 * Policy Thread (stub)
 * ==========================================================================
 */

static void policy_thread_func(void) { /* Stub for future policy threads */ }

/* ============================================================================
 * Public API
 * ============================================================================
 */

void policy_init(uint32_t static_bw) { g_static_bw = static_bw; }

uint32_t policy_get_initial_bw(void) { return g_static_bw; }

void policy_start(void) { /* Manual policy: no background updates */ }

void policy_stop(void) { /* Manual policy: no background updates */ }

void policy_handle_stdin_line(const char *line) {
  uint32_t slot;
  uint32_t bw;

  struct mpib_hint_shm *shm = shm_get();
  if (!shm)
    return;

  if (sscanf(line, "%u %u", &slot, &bw) == 2) {
    if (slot < MPIB_HINT_MAX_ENTRIES) {
      mpib_hint_write(&shm->entries[slot], bw);
      printf("Set slot %u to %u\n", slot, bw);
    } else {
      fprintf(stderr, "Invalid slot: %u\n", slot);
    }
  } else if (sscanf(line, "all %u", &bw) == 1) {
    update_all_hints(bw);
    printf("Set all slots to %u\n", bw);
  } else {
    fprintf(stderr, "Invalid input. Use: <slot> <bw> or all <bw>\n");
  }
}

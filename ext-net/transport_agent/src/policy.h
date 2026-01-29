/*
 * MPIB Transport Agent - Simulated Policy Logic
 *
 * Provides different hint update policies for testing.
 */

#ifndef MPIB_AGENT_POLICY_H
#define MPIB_AGENT_POLICY_H
#include <stdint.h>

/* Initialize policy (sets initial bw for new slots) */
void policy_init(uint32_t initial_bw);

/* Get the initial bw multiplier for new slots */
uint32_t policy_get_initial_bw(void);

/* Start the policy loop (no-op for manual) */
void policy_start(void);

/* Stop the policy loop (no-op for manual) */
void policy_stop(void);

/* Handle a line of input for manual policy */
void policy_handle_stdin_line(const char *line);

#endif /* MPIB_AGENT_POLICY_H */

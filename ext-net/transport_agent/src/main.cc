/*
 * MPIB Transport Agent - Entry Point
 *
 * This agent provides:
 *   - Shared memory hint file creation and management
 *   - Unix socket for registration/deregistration
 *   - Manual policy for testing
 *
 * Usage:
 *   ./mpib_agent --manual             # Read "slot bw" lines from stdin
 */

#include "ipc.h"
#include "policy.h"
#include "shm.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

/* ============================================================================
 * Global State
 * ============================================================================
 */

static volatile sig_atomic_t g_running = 1;

/* ============================================================================
 * Signal Handler
 * ============================================================================
 */

static void signal_handler(int sig) {
  (void)sig;
  g_running = 0;
}

/* ============================================================================
 * Usage
 * ============================================================================
 */

static void print_usage(const char *prog) {
  fprintf(stderr, "Usage: %s [options]\n", prog);
  fprintf(stderr, "Options:\n");
  fprintf(stderr,
          "  --manual           Read 'slot bw' or 'all bw' from stdin\n");
  fprintf(stderr, "\nExample:\n");
  fprintf(stderr, "  %s --manual\n", prog);
}

/* ============================================================================
 * Main
 * ============================================================================
 */

int main(int argc, char *argv[]) {
  uint32_t static_bw = 1;

  /* Parse arguments */
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--manual") == 0) {
      /* Manual policy is the default */
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      return 0;
    } else {
      fprintf(stderr, "Unknown option: %s\n", argv[i]);
      print_usage(argv[0]);
      return 1;
    }
  }

  /* Initialize policy */
  policy_init(static_bw);

  printf("MPIB Transport Agent (Phase 1)\n");
  printf("Policy: manual\n");

  /* Set up signal handlers */
  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);

  /* Create SHM */
  if (shm_create(static_bw) < 0) {
    return 1;
  }

  /* Create socket */
  if (ipc_create_socket() < 0) {
    shm_destroy();
    return 1;
  }

  /* Start policy thread */
  policy_start();

  /* Main loop: accept connections and handle stdin */
  struct pollfd fds[2];
  int nfds = 2;

  fds[0].fd = ipc_get_listen_fd();
  fds[0].events = POLLIN;

  fds[1].fd = STDIN_FILENO;
  fds[1].events = POLLIN;

  char stdin_buf[256];

  while (g_running) {
    int ret = poll(fds, nfds, 100);
    if (ret < 0) {
      if (errno == EINTR)
        continue;
      break;
    }

    if (fds[0].revents & POLLIN) {
      int client_fd = accept(ipc_get_listen_fd(), NULL, NULL);
      if (client_fd >= 0) {
        ipc_handle_client(client_fd);
      }
    }

    if (nfds > 1 && (fds[1].revents & POLLIN)) {
      if (fgets(stdin_buf, sizeof(stdin_buf), stdin)) {
        policy_handle_stdin_line(stdin_buf);
      }
    }
  }

  printf("\nShutting down...\n");

  /* Clean up */
  policy_stop();
  ipc_destroy_socket();
  shm_destroy();

  return 0;
}

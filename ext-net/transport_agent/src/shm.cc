/*
 * MPIB Transport Agent - Shared Memory Management
 *
 * Creates and manages the hint SHM file at /tmp/mpib/hints.
 */

#include "shm.h"

#include <cstdio>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* ============================================================================
 * Global State
 * ============================================================================
 */

static struct mpib_hint_shm *g_hint_shm = NULL;
static int g_hint_fd = -1;

/* ============================================================================
 * Public API
 * ============================================================================
 */

struct mpib_hint_shm *shm_get(void) { return g_hint_shm; }

int shm_create(uint32_t initial_bw) {
  /* Create directory */
  if (mkdir(MPIB_HINT_DIR, 0755) < 0 && errno != EEXIST) {
    fprintf(stderr, "Failed to create %s: %s\n", MPIB_HINT_DIR,
            strerror(errno));
    return -1;
  }

  /* Remove old file if exists */
  unlink(MPIB_HINT_PATH);

  /* Create and size the file */
  g_hint_fd = open(MPIB_HINT_PATH, O_RDWR | O_CREAT | O_EXCL, 0644);
  if (g_hint_fd < 0) {
    fprintf(stderr, "Failed to create %s: %s\n", MPIB_HINT_PATH,
            strerror(errno));
    return -1;
  }

  if (ftruncate(g_hint_fd, MPIB_HINT_SHM_SIZE) < 0) {
    fprintf(stderr, "Failed to size %s: %s\n", MPIB_HINT_PATH, strerror(errno));
    close(g_hint_fd);
    unlink(MPIB_HINT_PATH);
    return -1;
  }

  /* Map the file */
  void *ptr = mmap(NULL, MPIB_HINT_SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED,
                   g_hint_fd, 0);
  if (ptr == MAP_FAILED) {
    fprintf(stderr, "Failed to mmap %s: %s\n", MPIB_HINT_PATH, strerror(errno));
    close(g_hint_fd);
    unlink(MPIB_HINT_PATH);
    return -1;
  }

  g_hint_shm = (struct mpib_hint_shm *)ptr;

  /* Initialize header */
  g_hint_shm->header.magic = MPIB_HINT_MAGIC;
  g_hint_shm->header.max_entries = MPIB_HINT_MAX_ENTRIES;

  /* Initialize all entries with default hint */
  for (int i = 0; i < MPIB_HINT_MAX_ENTRIES; i++) {
    g_hint_shm->entries[i].sup_bw = initial_bw;
    g_hint_shm->entries[i].seq = 0;
    g_hint_shm->entries[i].src_ip = 0;
    g_hint_shm->entries[i].dst_ip = 0;
  }

  printf("Created hint SHM at %s (size=%zu)\n", MPIB_HINT_PATH,
         MPIB_HINT_SHM_SIZE);
  return 0;
}

void shm_destroy(void) {
  if (g_hint_shm) {
    munmap(g_hint_shm, MPIB_HINT_SHM_SIZE);
    g_hint_shm = NULL;
  }
  if (g_hint_fd >= 0) {
    close(g_hint_fd);
    g_hint_fd = -1;
  }
  unlink(MPIB_HINT_PATH);
}

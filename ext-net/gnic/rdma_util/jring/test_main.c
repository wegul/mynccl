#include <stdio.h>

// Declare the function from the shared library
extern int jring_library_test();

int main() {
  printf("Testing jring shared library...\n");

  int result = jring_library_test();
  if (result == 0) {
    printf("✓ All tests passed!\n");
    return 0;
  } else {
    printf("✗ Tests failed!\n");
    return 1;
  }
}

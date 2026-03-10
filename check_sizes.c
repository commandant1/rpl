#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include "include/rpl.h"

int main() {
    printf("Sizes:\n");
    printf("  bool: %zu\n", sizeof(bool));
    printf("  uint32_t: %zu\n", sizeof(uint32_t));
    printf("  void*: %zu\n", sizeof(void*));
    printf("  size_t: %zu\n", sizeof(size_t));
    printf("  DeviceType: %zu\n", sizeof(DeviceType));
    printf("  struct Tensor: %zu\n", sizeof(struct Tensor));
    return 0;
}

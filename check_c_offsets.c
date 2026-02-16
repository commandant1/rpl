#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include "include/rpl.h"

int main() {
    printf("Offsets for struct Tensor:\n");
    printf("  data: %zu\n", offsetof(struct Tensor, data));
    printf("  grad: %zu\n", offsetof(struct Tensor, grad));
    printf("  dims: %zu\n", offsetof(struct Tensor, dims));
    printf("  shape: %zu\n", offsetof(struct Tensor, shape));
    printf("  strides: %zu\n", offsetof(struct Tensor, strides));
    printf("  size: %zu\n", offsetof(struct Tensor, size));
    printf("  requires_grad: %zu\n", offsetof(struct Tensor, requires_grad));
    printf("  _allocation: %zu\n", offsetof(struct Tensor, _allocation));
    printf("  _alloc_size: %zu\n", offsetof(struct Tensor, _alloc_size));
    printf("  device: %zu\n", offsetof(struct Tensor, device));
    printf("  gpu_buffer: %zu\n", offsetof(struct Tensor, gpu_buffer));
    printf("  is_leaf: %zu\n", offsetof(struct Tensor, is_leaf));
    printf("  parent1: %zu\n", offsetof(struct Tensor, parent1));
    printf("  parent2: %zu\n", offsetof(struct Tensor, parent2));
    printf("  backward_fn: %zu\n", offsetof(struct Tensor, backward_fn));
    return 0;
}

#include "rpi_ml.h"
#include <stdio.h>
#include <time.h>

void benchmark_matmul() {
    uint32_t size = 512;
    Tensor* A = tensor_create(size, size, 1);
    Tensor* B = tensor_create(size, size, 1);
    Tensor* C = tensor_create(size, size, 1);
    
    tensor_randomize(A);
    tensor_randomize(B);
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    tensor_matmul_neon(A, B, C);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("MatMul %ux%u: %f seconds\n", size, size, elapsed);
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
}

int main() {
    printf("Starting benchmarks...\n");
    benchmark_matmul();
    return 0;
}

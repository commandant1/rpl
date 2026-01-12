/*
 * Performance benchmark for optimized kernels
 */

#include "rpi_ml.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

void benchmark_gemm() {
    printf("\n=== GEMM Benchmark ===\n");
    
    int sizes[] = {128, 256, 512, 1024};
    
    for (int s = 0; s < 4; s++) {
        int M = sizes[s], N = sizes[s], K = sizes[s];
        
        float* A = (float*)aligned_alloc(64, M * K * sizeof(float));
        float* B = (float*)aligned_alloc(64, K * N * sizeof(float));
        float* C = (float*)aligned_alloc(64, M * N * sizeof(float));
        
        // Initialize
        for (int i = 0; i < M*K; i++) A[i] = (float)rand() / RAND_MAX;
        for (int i = 0; i < K*N; i++) B[i] = (float)rand() / RAND_MAX;
        for (int i = 0; i < M*N; i++) C[i] = 0.0f;
        
        // Warmup
        parallel_gemm_optimized(A, B, C, M, N, K);
        
        // Benchmark
        double start = get_time();
        int iterations = (M < 512) ? 10 : 3;
        
        for (int iter = 0; iter < iterations; iter++) {
            parallel_gemm_optimized(A, B, C, M, N, K);
        }
        
        double elapsed = (get_time() - start) / iterations;
        double gflops = (2.0 * M * N * K) / (elapsed * 1e9);
        
        printf("Size %4dx%4d: %.4f s, %.2f GFLOPS\n", M, N, elapsed, gflops);
        
        free(A);
        free(B);
        free(C);
    }
}

void benchmark_conv() {
    printf("\n=== Conv2D Benchmark (Winograd) ===\n");
    
    int channels = 64;
    int height = 56, width = 56;
    
    float* input = (float*)aligned_alloc(64, channels * height * width * sizeof(float));
    float* kernel = (float*)aligned_alloc(64, channels * channels * 9 * sizeof(float));
    float* output = (float*)aligned_alloc(64, channels * height * width * sizeof(float));
    
    for (int i = 0; i < channels*height*width; i++) input[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < channels*channels*9; i++) kernel[i] = (float)rand() / RAND_MAX;
    
    // Warmup
    conv2d_winograd_3x3(input, kernel, output, channels, channels, height, width, 1, 1);
    
    // Benchmark
    double start = get_time();
    int iterations = 5;
    
    for (int iter = 0; iter < iterations; iter++) {
        conv2d_winograd_3x3(input, kernel, output, channels, channels, height, width, 1, 1);
    }
    
    double elapsed = (get_time() - start) / iterations;
    double gflops = (2.0 * channels * channels * 9 * height * width) / (elapsed * 1e9);
    
    printf("Conv2D %dx%dx%d: %.4f s, %.2f GFLOPS\n", channels, height, width, elapsed, gflops);
    
    free(input);
    free(kernel);
    free(output);
}

int main() {
    printf("======================================\n");
    printf("RPiTorch Optimized Kernel Benchmarks\n");
    printf("======================================\n");
    
    srand(42);
    
    benchmark_gemm();
    benchmark_conv();
    
    printf("\n======================================\n");
    printf("Benchmarks complete!\n");
    printf("======================================\n");
    
    return 0;
}

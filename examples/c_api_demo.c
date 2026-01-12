/*
 * RPiTorch C API Example
 * Demonstrates using the library directly from C
 */

#include "rpi_ml.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void example_basic_operations() {
    printf("\n=== Example 1: Basic Tensor Operations ===\n");
    
    // Create tensors
    uint32_t shape_a[2] = {3, 3};
    uint32_t shape_b[2] = {3, 3};
    
    Tensor* A = tensor_create(2, shape_a, true);
    Tensor* B = tensor_create(2, shape_b, true);
    
    // Fill with random values
    tensor_randomize(A);
    tensor_randomize(B);
    
    printf("Created tensors A and B (3x3)\n");
    
    // Matrix multiplication
    Tensor* C = tensor_matmul(A, B);
    printf("Matrix multiplication: C = A * B\n");
    printf("C[0,0] = %f\n", C->data[0]);
    
    // Element-wise operations
    Tensor* D = tensor_add(A, B);
    printf("Element-wise addition: D = A + B\n");
    
    Tensor* E = tensor_relu(A);
    printf("ReLU activation applied\n");
    
    // Cleanup
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
    tensor_free(D);
    tensor_free(E);
}

void example_convolution() {
    printf("\n=== Example 2: Convolutional Layer ===\n");
    
    // Input: 1 channel, 28x28 image
    uint32_t input_shape[3] = {1, 28, 28};
    Tensor* input = tensor_create(3, input_shape, false);
    tensor_randomize(input);
    
    // Kernel: 16 filters, 1 input channel, 3x3
    uint32_t kernel_shape[4] = {16, 1, 3, 3};
    Tensor* kernel = tensor_create(4, kernel_shape, true);
    tensor_randomize(kernel);
    
    printf("Input: 1x28x28\n");
    printf("Kernel: 16x1x3x3\n");
    
    // Convolution with stride=1, padding=1
    Tensor* output = tensor_conv2d(input, kernel, 1, 1);
    
    printf("Output shape: %dx%dx%d\n", 
           output->shape[0], output->shape[1], output->shape[2]);
    
    // Apply ReLU
    Tensor* activated = tensor_relu(output);
    
    // Max pooling
    Tensor* pooled = tensor_maxpool2d(activated, 2, 2);
    printf("After 2x2 max pooling: %dx%dx%d\n",
           pooled->shape[0], pooled->shape[1], pooled->shape[2]);
    
    tensor_free(input);
    tensor_free(kernel);
    tensor_free(output);
    tensor_free(activated);
    tensor_free(pooled);
}

void example_batch_normalization() {
    printf("\n=== Example 3: Batch Normalization ===\n");
    
    // Input: batch=2, channels=3, height=4, width=4
    uint32_t shape[4] = {2, 3, 4, 4};
    Tensor* input = tensor_create(4, shape, false);
    Tensor* output = tensor_create(4, shape, false);
    tensor_randomize(input);
    
    // BatchNorm parameters
    float gamma[3] = {1.0f, 1.0f, 1.0f};
    float beta[3] = {0.0f, 0.0f, 0.0f};
    float running_mean[3] = {0.0f, 0.0f, 0.0f};
    float running_var[3] = {1.0f, 1.0f, 1.0f};
    
    tensor_batchnorm2d_forward(input, output, gamma, beta,
                               running_mean, running_var,
                               1e-5, true, 0.1);
    
    printf("BatchNorm applied to 2x3x4x4 tensor\n");
    printf("Output mean (should be ~0): %f\n", 
           output->data[0] + output->data[1] + output->data[2]);
    
    tensor_free(input);
    tensor_free(output);
}

void example_quantization() {
    printf("\n=== Example 4: INT8 Quantization ===\n");
    
    // Create FP32 tensor
    uint32_t shape[2] = {100, 100};
    Tensor* fp32_tensor = tensor_create(2, shape, false);
    tensor_randomize(fp32_tensor);
    
    // Get min/max for calibration
    float min_val, max_val;
    tensor_get_min_max(fp32_tensor, &min_val, &max_val);
    
    // Compute quantization parameters
    float abs_max = (max_val > -min_val) ? max_val : -min_val;
    float scale = abs_max / 127.0f;
    int32_t zero_point = 0;
    
    printf("FP32 tensor range: [%f, %f]\n", min_val, max_val);
    printf("Quantization scale: %f\n", scale);
    
    // Quantize to INT8
    QuantizedTensor* int8_tensor = tensor_quantize_int8(fp32_tensor, scale, zero_point);
    
    size_t fp32_size = fp32_tensor->size * sizeof(float);
    size_t int8_size = int8_tensor->size * sizeof(int8_t);
    
    printf("FP32 size: %zu bytes\n", fp32_size);
    printf("INT8 size: %zu bytes\n", int8_size);
    printf("Compression ratio: %.1fx\n", (float)fp32_size / int8_size);
    
    // Dequantize back
    Tensor* dequantized = tensor_dequantize_int8(int8_tensor);
    
    // Compute error
    float error = 0.0f;
    for (uint32_t i = 0; i < fp32_tensor->size; i++) {
        float diff = fp32_tensor->data[i] - dequantized->data[i];
        error += diff * diff;
    }
    error = sqrtf(error / fp32_tensor->size);
    printf("Quantization RMSE: %f\n", error);
    
    tensor_free(fp32_tensor);
    tensor_free(dequantized);
    quantized_tensor_free(int8_tensor);
}

void example_int8_gemm() {
    printf("\n=== Example 5: INT8 GEMM (Fast Matrix Multiply) ===\n");
    
    uint32_t M = 128, N = 128, K = 128;
    
    // Allocate INT8 matrices
    int8_t* A = (int8_t*)aligned_alloc(64, M * K * sizeof(int8_t));
    int8_t* B = (int8_t*)aligned_alloc(64, K * N * sizeof(int8_t));
    int32_t* C = (int32_t*)aligned_alloc(64, M * N * sizeof(int32_t));
    
    // Fill with random INT8 values
    for (uint32_t i = 0; i < M * K; i++) A[i] = rand() % 256 - 128;
    for (uint32_t i = 0; i < K * N; i++) B[i] = rand() % 256 - 128;
    
    // Perform INT8 GEMM
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    gemm_int8(A, B, C, M, N, K, 0.01f, 0.01f, 0.01f, 0, 0);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + 
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("INT8 GEMM %dx%d: %.4f seconds\n", M, N, elapsed);
    printf("Performance: %.2f GOPS\n", 
           (2.0 * M * N * K) / (elapsed * 1e9));
    
    free(A);
    free(B);
    free(C);
}

void example_autograd() {
    printf("\n=== Example 6: Automatic Differentiation ===\n");
    
    // Create computation graph: y = (A * B) + C
    uint32_t shape[2] = {2, 2};
    
    Tensor* A = tensor_create(2, shape, true);
    Tensor* B = tensor_create(2, shape, true);
    Tensor* C = tensor_create(2, shape, true);
    
    // Initialize
    A->data[0] = 1.0f; A->data[1] = 2.0f;
    A->data[2] = 3.0f; A->data[3] = 4.0f;
    
    B->data[0] = 5.0f; B->data[1] = 6.0f;
    B->data[2] = 7.0f; B->data[3] = 8.0f;
    
    tensor_fill(C, 1.0f);
    
    // Forward pass
    Tensor* AB = tensor_mul(A, B);  // Element-wise multiply
    Tensor* y = tensor_add(AB, C);
    
    printf("Forward pass: y = (A * B) + C\n");
    printf("y[0] = %f\n", y->data[0]);
    
    // Backward pass
    tensor_backward(y);
    
    printf("Gradients computed via autograd:\n");
    printf("dL/dA[0] = %f\n", A->grad ? A->grad[0] : 0.0f);
    printf("dL/dB[0] = %f\n", B->grad ? B->grad[0] : 0.0f);
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
    tensor_free(AB);
    tensor_free(y);
}

int main() {
    printf("======================================\n");
    printf("RPiTorch C API Examples\n");
    printf("======================================\n");
    
    srand(time(NULL));
    
    example_basic_operations();
    example_convolution();
    example_batch_normalization();
    example_quantization();
    example_int8_gemm();
    example_autograd();
    
    printf("\n======================================\n");
    printf("All examples completed successfully!\n");
    printf("======================================\n");
    
    return 0;
}

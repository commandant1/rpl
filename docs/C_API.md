# RPiTorch C API Documentation

## Overview

RPiTorch provides a complete C API for building and training neural networks. All Python functionality is built on top of this C library.

## Building C Programs

```bash
# Compile your program
gcc -O3 -march=native -fopenmp -I/path/to/include \
    your_program.c -o your_program \
    -L/path/to/build -lrpiml -lm -fopenmp

# Run with library path
LD_LIBRARY_PATH=/path/to/build ./your_program
```

Or use the provided Makefile:
```bash
cd examples
make c_api_demo
make run_demo
```

## Core Data Structures

### Tensor
```c
typedef struct {
    float* data;           // Data array
    float* grad;           // Gradient array (if requires_grad)
    uint32_t dims;         // Number of dimensions
    uint32_t shape[MAX_DIMS];   // Shape array
    uint32_t strides[MAX_DIMS]; // Strides for indexing
    uint32_t size;         // Total number of elements
    bool requires_grad;    // Track gradients?
    
    // Autograd metadata
    struct Tensor* parent1;
    struct Tensor* parent2;
    void (*backward_fn)(struct Tensor*);
} Tensor;
```

### QuantizedTensor
```c
typedef struct {
    int8_t* data;          // INT8 data
    float scale;           // Quantization scale
    int32_t zero_point;    // Zero point
    uint32_t size;         // Number of elements
} QuantizedTensor;
```

## API Reference

### Memory Management

```c
// Create tensor with given shape
Tensor* tensor_create(uint32_t dims, const uint32_t* shape, bool requires_grad);

// Create 2D tensor (convenience)
Tensor* tensor_create_2d(uint32_t rows, uint32_t cols, bool requires_grad);

// Free tensor
void tensor_free(Tensor* t);
```

### Initialization

```c
// Fill with constant value
void tensor_fill(Tensor* t, float value);

// Fill with random values [-1, 1]
void tensor_randomize(Tensor* t);
```

### Basic Operations

```c
// Element-wise addition: C = A + B
Tensor* tensor_add(Tensor* a, Tensor* b);

// Element-wise multiplication: C = A * B
Tensor* tensor_mul(Tensor* a, Tensor* b);

// Matrix multiplication: C = A @ B
Tensor* tensor_matmul(Tensor* a, Tensor* b);
```

### Activations

```c
// ReLU: max(0, x)
Tensor* tensor_relu(Tensor* t);

// Sigmoid: 1 / (1 + exp(-x))
Tensor* tensor_sigmoid(Tensor* t);

// Tanh: tanh(x)
Tensor* tensor_tanh(Tensor* t);
```

### Convolutional Operations

```c
// 2D Convolution
Tensor* tensor_conv2d(Tensor* input, Tensor* kernel, 
                      uint32_t stride, uint32_t padding);

// 2D Max Pooling
Tensor* tensor_maxpool2d(Tensor* input, 
                         uint32_t kernel_size, uint32_t stride);
```

### Normalization

```c
// Batch Normalization 2D
void tensor_batchnorm2d_forward(
    Tensor* input, Tensor* output,
    const float* gamma, const float* beta,
    const float* running_mean, const float* running_var,
    float eps, bool training, float momentum);

// Layer Normalization
void tensor_layernorm_forward(
    Tensor* input, Tensor* output,
    const float* gamma, const float* beta,
    float eps);
```

### Quantization

```c
// Get min/max for calibration
void tensor_get_min_max(Tensor* t, float* min_val, float* max_val);

// Quantize FP32 → INT8
QuantizedTensor* tensor_quantize_int8(
    Tensor* input, float scale, int32_t zero_point);

// Dequantize INT8 → FP32
Tensor* tensor_dequantize_int8(QuantizedTensor* input);

// Free quantized tensor
void quantized_tensor_free(QuantizedTensor* qt);

// INT8 matrix multiplication (NEON-optimized)
void gemm_int8(
    const int8_t* A, const int8_t* B, int32_t* C,
    uint32_t M, uint32_t N, uint32_t K,
    float scale_a, float scale_b, float scale_c,
    int32_t zero_point_a, int32_t zero_point_b);
```

### Autograd

```c
// Compute gradients via backpropagation
void tensor_backward(Tensor* t);

// Zero out gradients
void tensor_zero_grad(Tensor* t);
```

## Example: Simple Neural Network

```c
#include "rpi_ml.h"

int main() {
    // Create layers
    uint32_t w1_shape[2] = {784, 128};
    uint32_t w2_shape[2] = {128, 10};
    
    Tensor* w1 = tensor_create(2, w1_shape, true);
    Tensor* w2 = tensor_create(2, w2_shape, true);
    
    tensor_randomize(w1);
    tensor_randomize(w2);
    
    // Forward pass
    uint32_t x_shape[2] = {32, 784};  // batch_size=32
    Tensor* x = tensor_create(2, x_shape, false);
    tensor_randomize(x);
    
    Tensor* h = tensor_matmul(x, w1);
    Tensor* h_act = tensor_relu(h);
    Tensor* output = tensor_matmul(h_act, w2);
    
    // Backward pass
    tensor_backward(output);
    
    // Update weights (simple SGD)
    float lr = 0.01f;
    for (uint32_t i = 0; i < w1->size; i++) {
        w1->data[i] -= lr * w1->grad[i];
    }
    for (uint32_t i = 0; i < w2->size; i++) {
        w2->data[i] -= lr * w2->grad[i];
    }
    
    // Cleanup
    tensor_free(w1);
    tensor_free(w2);
    tensor_free(x);
    tensor_free(h);
    tensor_free(h_act);
    tensor_free(output);
    
    return 0;
}
```

## Example: INT8 Quantization

```c
// Create FP32 tensor
uint32_t shape[2] = {100, 100};
Tensor* fp32 = tensor_create(2, shape, false);
tensor_randomize(fp32);

// Calibrate
float min_val, max_val;
tensor_get_min_max(fp32, &min_val, &max_val);

float scale = fmaxf(fabsf(min_val), fabsf(max_val)) / 127.0f;
int32_t zero_point = 0;

// Quantize
QuantizedTensor* int8 = tensor_quantize_int8(fp32, scale, zero_point);

printf("Compression: %.1fx\n", 
       (float)(fp32->size * 4) / (int8->size * 1));

// Dequantize
Tensor* dequant = tensor_dequantize_int8(int8);

// Cleanup
tensor_free(fp32);
tensor_free(dequant);
quantized_tensor_free(int8);
```

## Performance Tips

1. **Use OpenMP**: The library automatically parallelizes operations across cores
2. **Enable NEON**: Compile with `-march=armv8-a` on RPi4 for SIMD acceleration
3. **Quantize for Inference**: Use INT8 for 4x compression and 2-4x speedup
4. **Batch Operations**: Larger batches improve throughput
5. **Aligned Memory**: The library uses 64-byte alignment for cache efficiency

## Complete Examples

See the `examples/` directory:
- `c_api_demo.c` - Comprehensive API demonstration
- `simple_nn.c` - 2-layer MLP training from scratch
- `Makefile` - Build system for C examples

## Thread Safety

The library is thread-safe for read operations. For write operations (training), use external synchronization or separate tensor instances per thread.

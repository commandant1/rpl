#include "rpl.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EPSILON 1e-3

void assert_tensor_eq(Tensor* expected, Tensor* actual, const char* name) {
    if (expected->size != actual->size) {
        printf("[FAIL] %s: Size mismatch (%u != %u)\n", name, expected->size, actual->size);
        exit(1);
    }
    
    // Download from GPU if needed
    if (expected->device == DEVICE_GPU) tensor_from_gpu(expected);
    if (actual->device == DEVICE_GPU) tensor_from_gpu(actual);
    
    for (uint32_t i = 0; i < expected->size; i++) {
        float diff = fabsf(expected->data[i] - actual->data[i]);
        if (diff > EPSILON) {
            printf("[FAIL] %s: Value mismatch at index %u: %f != %f\n", name, i, expected->data[i], actual->data[i]);
            exit(1);
        }
    }
    printf("[PASS] %s\n", name);
}

void test_rmsnorm() {
    uint32_t shape[] = {2, 4};
    Tensor* input = tensor_create(2, shape, false);
    for (int i = 0; i < 8; i++) input->data[i] = i * 0.1f;
    
    RMSNormLayer* layer = rmsnorm_create(4, 1e-5f);
    for (int i = 0; i < 4; i++) layer->weight->data[i] = 1.2f;
    
    Tensor* out_cpu = rmsnorm_forward(layer, input);
    
    // Copy for GPU
    Tensor* input_gpu = tensor_create(2, shape, false);
    memcpy(input_gpu->data, input->data, input->size * sizeof(float));
    tensor_to_gpu(input_gpu);
    
    tensor_to_gpu(layer->weight); // move weight to GPU
    
    Tensor* out_gpu = tensor_create(2, shape, false);
    tensor_to_gpu(out_gpu);
    
    tensor_rmsnorm_gpu(out_gpu, input_gpu, layer->weight, 1e-5f);
    
    assert_tensor_eq(out_cpu, out_gpu, "RMSNorm GPU vs CPU");
    
    rmsnorm_free(layer);
    tensor_free(input);
    tensor_free(out_cpu);
    tensor_free(input_gpu);
    tensor_free(out_gpu);
}

void test_rope() {
    uint32_t shape[] = {1, 2, 2, 4}; // batch, seq, heads, dim
    Tensor* q_cpu = tensor_create(4, shape, false);
    Tensor* k_cpu = tensor_create(4, shape, false);
    
    for (int i = 0; i < q_cpu->size; i++) {
        q_cpu->data[i] = i * 0.1f;
        k_cpu->data[i] = i * 0.2f;
    }
    
    Tensor* q_gpu = tensor_create(4, shape, false);
    Tensor* k_gpu = tensor_create(4, shape, false);
    memcpy(q_gpu->data, q_cpu->data, q_cpu->size * sizeof(float));
    memcpy(k_gpu->data, k_cpu->data, k_cpu->size * sizeof(float));
    
    tensor_to_gpu(q_gpu);
    tensor_to_gpu(k_gpu);
    
    tensor_rope(q_cpu, k_cpu, 4, 10000.0f);
    tensor_rope_gpu(q_gpu, k_gpu, 4, 10000.0f);
    
    assert_tensor_eq(q_cpu, q_gpu, "RoPE Q GPU vs CPU");
    assert_tensor_eq(k_cpu, k_gpu, "RoPE K GPU vs CPU");
    
    tensor_free(q_cpu);
    tensor_free(k_cpu);
    tensor_free(q_gpu);
    tensor_free(k_gpu);
}

int main() {
    if (!rpl_gpu_init()) {
        printf("GPU initialization failed. Skipping subset tests.\n");
        return 0;
    }
    
    test_rmsnorm();
    test_rope();
    
    rpl_gpu_shutdown();
    return 0;
}

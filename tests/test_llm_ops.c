#include "rpl.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EPSILON 1e-4

void assert_float_eq(float expected, float actual, const char* name) {
    if (fabsf(expected - actual) > EPSILON) {
        printf("[FAIL] %s: Expected %f, got %f\n", name, expected, actual);
        exit(1);
    }
}

void test_swiglu() {
    uint32_t shape[] = {2, 4};
    uint32_t shape_out[] = {2, 2};
    Tensor* in = tensor_create(2, shape, false);
    Tensor* out = tensor_create(2, shape_out, false);
    
    // In[0,:] = [1.0, -1.0, 2.0, -2.0]
    // In[1,:] = [0.5, 0.0, 1.5, -0.5]
    in->data[0] = 1.0f; in->data[1] = -1.0f; in->data[2] = 2.0f; in->data[3] = -2.0f;
    in->data[4] = 0.5f; in->data[5] = 0.0f; in->data[6] = 1.5f; in->data[7] = -0.5f;
    
    // SwiGLU dims=-1 applied on 'in'
    tensor_swiglu(out, in, -1);
    
    // Expected values manually computed
    // out[0, 0] = swish(1.0) * 2.0 = (1 * sigmoid(1)) * 2 = 1.462117f
    // out[0, 1] = swish(-1.0) * -2.0 = (-1 * sigmoid(-1)) * -2 = 0.53788f
    // out[1, 0] = swish(0.5) * 1.5 = 0.46879f
    // out[1, 1] = swish(0.0) * -0.5 = 0.0
    
    assert_float_eq((1.0f / (1.0f + expf(-1.0f))) * 2.0f, out->data[0], "SwiGLU 0,0");
    assert_float_eq((-1.0f / (1.0f + expf(1.0f))) * -2.0f, out->data[1], "SwiGLU 0,1");
    assert_float_eq((0.5f / (1.0f + expf(-0.5f))) * 1.5f, out->data[2], "SwiGLU 1,0");
    assert_float_eq(0.0f, out->data[3], "SwiGLU 1,1");
    
    printf("[PASS] SwiGLU\n");
    tensor_free(in);
    tensor_free(out);
}

void test_kv_cache() {
    KVCache* cache = kv_cache_create(1024, 1, 2, 64);
    
    uint32_t num_heads = 2, head_dim = 64;
    uint32_t seq_len_new = 4;
    uint32_t shape[] = {1, seq_len_new, num_heads, head_dim};
    
    Tensor* k_new = tensor_zeros(4, shape);
    Tensor* v_new = tensor_zeros(4, shape);
    
    k_new->data[0] = 42.0f; // first element
    v_new->data[127] = 84.0f; // last element of first token across both heads
    
    kv_cache_update(cache, k_new, v_new, seq_len_new);
    
    if (cache->current_pos != 4) {
        printf("[FAIL] KV Cache internal position mismatch!\n");
        exit(1);
    }
    if (cache->key_cache->data[0] != 42.0f) {
        printf("[FAIL] KV Cache data corruption on write!\n");
        exit(1);
    }
    
    printf("[PASS] KV Cache\n");
    
    tensor_free(k_new);
    tensor_free(v_new);
    kv_cache_free(cache);
}

void test_top_k_p() {
    uint32_t shape[] = {5};
    Tensor* logits = tensor_create(1, shape, false);
    
    logits->data[0] = 10.0f;
    logits->data[1] = 9.0f;
    logits->data[2] = -10.0f;
    logits->data[3] = 2.0f;
    logits->data[4] = 8.5f;
    
    uint32_t k_idx = tensor_sample_top_k(logits, 1, 1.0f);
    if (k_idx != 0) {
        printf("[FAIL] Top-k k=1 failed to pick max logit\n");
        exit(1);
    }
    
    // Top-p deterministic logic check
    // probs: exp(10) > exp(9) > exp(8.5).
    // top_p = 0.001 will safely pick the largest
    uint32_t p_idx = tensor_sample_top_p(logits, 0.001f, 1.0f);
    if (p_idx != 0) {
        printf("[FAIL] Top-p p=0.001 failed to pick max logit\n");
        exit(1);
    }
    
    printf("[PASS] Top-K / Top-P\n");
    tensor_free(logits);
}

int main() {
    test_swiglu();
    test_kv_cache();
    test_top_k_p();
    return 0;
}

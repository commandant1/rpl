#include "rpl.h"
#include <stdlib.h>
#include <string.h>

// ============================================================
// KV Cache Management
// ============================================================

KVCache* kv_cache_create(uint32_t max_seq_len, uint32_t batch_size, uint32_t num_heads, uint32_t head_dim) {
    KVCache* cache = (KVCache*)calloc(1, sizeof(KVCache));
    
    cache->max_seq_len = max_seq_len;
    cache->batch_size = batch_size;
    cache->num_heads = num_heads;
    cache->head_dim = head_dim;
    cache->current_pos = 0;
    
    uint32_t shape[] = {batch_size, max_seq_len, num_heads, head_dim};
    cache->key_cache = tensor_create(4, shape, false);
    cache->value_cache = tensor_create(4, shape, false);
    
    tensor_fill(cache->key_cache, 0.0f);
    tensor_fill(cache->value_cache, 0.0f);
    
    return cache;
}

void kv_cache_update(KVCache* cache, const Tensor* k_new, const Tensor* v_new, uint32_t seq_len_new) {
    // k_new, v_new shape expected: [batch_size, seq_len_new, num_heads, head_dim]
    if (cache->current_pos + seq_len_new > cache->max_seq_len) {
        // Handle eviction or overflow (simple circular buffer or drop for this impl)
        // For standard LLM generation, we just clamp or ring-buffer
        seq_len_new = cache->max_seq_len - cache->current_pos;
        if (seq_len_new == 0) return; // Full
    }
    
    uint32_t row_size = cache->num_heads * cache->head_dim;
    
    // Download to CPU before copying if needed
#ifdef USE_GPU
    if (k_new->device == DEVICE_GPU) tensor_from_gpu((Tensor*)k_new);
    if (v_new->device == DEVICE_GPU) tensor_from_gpu((Tensor*)v_new);
#endif

    #pragma omp parallel for collapse(2)
    for (uint32_t b = 0; b < cache->batch_size; b++) {
        for (uint32_t s = 0; s < seq_len_new; s++) {
            uint32_t dst_idx = (b * cache->max_seq_len + cache->current_pos + s) * row_size;
            uint32_t src_idx = (b * seq_len_new + s) * row_size;
            
            memcpy(&cache->key_cache->data[dst_idx], &k_new->data[src_idx], row_size * sizeof(float));
            memcpy(&cache->value_cache->data[dst_idx], &v_new->data[src_idx], row_size * sizeof(float));
        }
    }
    
    cache->current_pos += seq_len_new;
    
#ifdef USE_GPU
    // Sync the cache tensor back to GPU if necessary
    if (cache->key_cache->device == DEVICE_GPU) tensor_to_gpu(cache->key_cache);
    if (cache->value_cache->device == DEVICE_GPU) tensor_to_gpu(cache->value_cache);
#endif
}

void kv_cache_free(KVCache* cache) {
    if (!cache) return;
    tensor_free(cache->key_cache);
    tensor_free(cache->value_cache);
    free(cache);
}

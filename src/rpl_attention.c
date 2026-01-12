/*
 * RPiTorch Attention Mechanisms
 * Scaled Dot-Product Attention, Multi-Head Attention
 */

#include "rpl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// ============================================================
// Scaled Dot-Product Attention
// ============================================================

Tensor* scaled_dot_product_attention(const Tensor* Q, const Tensor* K, const Tensor* V,
                                     const Tensor* mask, float dropout_p, bool training) {
    // Q, K, V: [batch, seq_len, d_k]
    uint32_t batch_size = Q->shape[0];
    uint32_t seq_len_q = Q->shape[1];
    uint32_t seq_len_k = K->shape[1];
    uint32_t d_k = Q->shape[2];
    
    // Compute Q @ K^T
    uint32_t scores_shape[3] = {batch_size, seq_len_q, seq_len_k};
    Tensor* scores = tensor_create(3, scores_shape, Q->requires_grad);
    
    float scale = 1.0f / sqrtf((float)d_k);
    
    #pragma omp parallel for collapse(2)
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t i = 0; i < seq_len_q; i++) {
            for (uint32_t j = 0; j < seq_len_k; j++) {
                float sum = 0.0f;
                
                // Dot product Q[b,i,:] · K[b,j,:]
                for (uint32_t k = 0; k < d_k; k++) {
                    sum += Q->data[b * seq_len_q * d_k + i * d_k + k] *
                          K->data[b * seq_len_k * d_k + j * d_k + k];
                }
                
                scores->data[b * seq_len_q * seq_len_k + i * seq_len_k + j] = sum * scale;
            }
        }
    }
    
    // Apply mask if provided
    if (mask) {
        #pragma omp parallel for
        for (uint32_t i = 0; i < scores->size; i++) {
            if (mask->data[i] == 0.0f) {
                scores->data[i] = -1e9f;  // Large negative value
            }
        }
    }
    
    // Softmax over last dimension
    #pragma omp parallel for collapse(2)
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t i = 0; i < seq_len_q; i++) {
            float* row = &scores->data[b * seq_len_q * seq_len_k + i * seq_len_k];
            
            // Find max for numerical stability
            float max_val = -FLT_MAX;
            for (uint32_t j = 0; j < seq_len_k; j++) {
                if (row[j] > max_val) max_val = row[j];
            }
            
            // Exp and sum
            float sum_exp = 0.0f;
            for (uint32_t j = 0; j < seq_len_k; j++) {
                row[j] = expf(row[j] - max_val);
                sum_exp += row[j];
            }
            
            // Normalize
            for (uint32_t j = 0; j < seq_len_k; j++) {
                row[j] /= sum_exp;
            }
        }
    }
    
    // Apply dropout if training
    if (training && dropout_p > 0.0f) {
        tensor_dropout(scores, scores, dropout_p, true);
    }
    
    // Compute scores @ V
    uint32_t d_v = V->shape[2];
    uint32_t output_shape[3] = {batch_size, seq_len_q, d_v};
    Tensor* output = tensor_create(3, output_shape, Q->requires_grad);
    
    #pragma omp parallel for collapse(2)
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t i = 0; i < seq_len_q; i++) {
            for (uint32_t k = 0; k < d_v; k++) {
                float sum = 0.0f;
                
                for (uint32_t j = 0; j < seq_len_k; j++) {
                    sum += scores->data[b * seq_len_q * seq_len_k + i * seq_len_k + j] *
                          V->data[b * seq_len_k * d_v + j * d_v + k];
                }
                
                output->data[b * seq_len_q * d_v + i * d_v + k] = sum;
            }
        }
    }
    
    tensor_free(scores);
    return output;
}

// ============================================================
// Multi-Head Attention
// ============================================================

struct MultiHeadAttention {
    uint32_t d_model;
    uint32_t num_heads;
    uint32_t d_k;
    uint32_t d_v;
    
    Tensor* W_q;  // [d_model, d_model]
    Tensor* W_k;  // [d_model, d_model]
    Tensor* W_v;  // [d_model, d_model]
    Tensor* W_o;  // [d_model, d_model]
    
    float dropout_p;
};

MultiHeadAttention* multi_head_attention_create(uint32_t d_model, uint32_t num_heads, float dropout_p) {
    MultiHeadAttention* mha = (MultiHeadAttention*)calloc(1, sizeof(MultiHeadAttention));
    
    mha->d_model = d_model;
    mha->num_heads = num_heads;
    mha->d_k = d_model / num_heads;
    mha->d_v = d_model / num_heads;
    mha->dropout_p = dropout_p;
    
    uint32_t weight_shape[2] = {d_model, d_model};
    mha->W_q = tensor_create(2, weight_shape, true);
    mha->W_k = tensor_create(2, weight_shape, true);
    mha->W_v = tensor_create(2, weight_shape, true);
    mha->W_o = tensor_create(2, weight_shape, true);
    
    // Xavier initialization
    float std = sqrtf(2.0f / (d_model + d_model));
    for (uint32_t i = 0; i < d_model * d_model; i++) {
        mha->W_q->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std;
        mha->W_k->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std;
        mha->W_v->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std;
        mha->W_o->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std;
    }
    
    return mha;
}

Tensor* multi_head_attention_forward(MultiHeadAttention* mha, const Tensor* query,
                                    const Tensor* key, const Tensor* value,
                                    const Tensor* mask, bool training) {
    // query, key, value: [batch, seq_len, d_model]
    uint32_t batch_size = query->shape[0];
    uint32_t seq_len = query->shape[1];
    
    // Linear projections with trans_b=true to match [out, in] convention
    uint32_t proj_shape[3] = {batch_size, seq_len, mha->d_model};
    Tensor* Q = tensor_create(3, proj_shape, query->requires_grad);
    Tensor* K = tensor_create(3, proj_shape, key->requires_grad);
    Tensor* V = tensor_create(3, proj_shape, value->requires_grad);
    
    tensor_fill(Q, 0.0f);
    tensor_fill(K, 0.0f);
    tensor_fill(V, 0.0f);
    
    tensor_gemm(Q, query, mha->W_q, 1.0f, 0.0f, false, true);
    tensor_gemm(K, key, mha->W_k, 1.0f, 0.0f, false, true);
    tensor_gemm(V, value, mha->W_v, 1.0f, 0.0f, false, true);
    
    // Reshape to [batch, num_heads, seq_len, d_k]
    // For simplicity, we'll process each head sequentially
    
    Tensor* outputs[mha->num_heads];
    
    for (uint32_t h = 0; h < mha->num_heads; h++) {
        // Extract head h
        uint32_t head_shape[3] = {batch_size, seq_len, mha->d_k};
        Tensor* Q_h = tensor_create(3, head_shape, false);
        Tensor* K_h = tensor_create(3, head_shape, false);
        Tensor* V_h = tensor_create(3, head_shape, false);
        
        for (uint32_t b = 0; b < batch_size; b++) {
            for (uint32_t s = 0; s < seq_len; s++) {
                for (uint32_t k = 0; k < mha->d_k; k++) {
                    uint32_t src_idx = b * seq_len * mha->d_model + s * mha->d_model + h * mha->d_k + k;
                    uint32_t dst_idx = b * seq_len * mha->d_k + s * mha->d_k + k;
                    
                    Q_h->data[dst_idx] = Q->data[src_idx];
                    K_h->data[dst_idx] = K->data[src_idx];
                    V_h->data[dst_idx] = V->data[src_idx];
                }
            }
        }
        
        // Apply attention
        outputs[h] = scaled_dot_product_attention(Q_h, K_h, V_h, mask, mha->dropout_p, training);
        
        tensor_free(Q_h);
        tensor_free(K_h);
        tensor_free(V_h);
    }
    
    // Concatenate heads
    uint32_t concat_shape[3] = {batch_size, seq_len, mha->d_model};
    Tensor* concat = tensor_create(3, concat_shape, query->requires_grad);
    
    for (uint32_t h = 0; h < mha->num_heads; h++) {
        for (uint32_t b = 0; b < batch_size; b++) {
            for (uint32_t s = 0; s < seq_len; s++) {
                for (uint32_t k = 0; k < mha->d_k; k++) {
                    uint32_t src_idx = b * seq_len * mha->d_k + s * mha->d_k + k;
                    uint32_t dst_idx = b * seq_len * mha->d_model + s * mha->d_model + h * mha->d_k + k;
                    concat->data[dst_idx] = outputs[h]->data[src_idx];
                }
            }
        }
        tensor_free(outputs[h]);
    }
    
    // Output projection
    Tensor* output = tensor_create(3, concat_shape, query->requires_grad);
    tensor_fill(output, 0.0f);
    tensor_gemm(output, concat, mha->W_o, 1.0f, 0.0f, false, true);
    
    tensor_free(Q);
    tensor_free(K);
    tensor_free(V);
    tensor_free(concat);
    
    return output;
}

void multi_head_attention_free(MultiHeadAttention* mha) {
    tensor_free(mha->W_q);
    tensor_free(mha->W_k);
    tensor_free(mha->W_v);
    tensor_free(mha->W_o);
    free(mha);
}

// ============================================================
// Positional Encoding
// ============================================================

struct PositionalEncoding {
    uint32_t max_len;
    uint32_t d_model;
    Tensor* encoding;  // [max_len, d_model]
    bool learnable;
};

PositionalEncoding* positional_encoding_create(uint32_t max_len, uint32_t d_model, bool learnable) {
    PositionalEncoding* pe = (PositionalEncoding*)calloc(1, sizeof(PositionalEncoding));
    
    pe->max_len = max_len;
    pe->d_model = d_model;
    pe->learnable = learnable;
    
    uint32_t encoding_shape[2] = {max_len, d_model};
    pe->encoding = tensor_create(2, encoding_shape, learnable);
    
    if (!learnable) {
        // Sinusoidal positional encoding
        for (uint32_t pos = 0; pos < max_len; pos++) {
            for (uint32_t i = 0; i < d_model; i++) {
                float angle = pos / powf(10000.0f, (2.0f * (i / 2)) / d_model);
                
                if (i % 2 == 0) {
                    pe->encoding->data[pos * d_model + i] = sinf(angle);
                } else {
                    pe->encoding->data[pos * d_model + i] = cosf(angle);
                }
            }
        }
    } else {
        // Random initialization for learnable encoding
        for (uint32_t i = 0; i < pe->encoding->size; i++) {
            pe->encoding->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        }
    }
    
    return pe;
}

Tensor* positional_encoding_forward(PositionalEncoding* pe, const Tensor* input) {
    // input: [batch, seq_len, d_model]
    uint32_t batch = input->shape[0];
    uint32_t seq_len = input->shape[1];
    uint32_t d_model = input->shape[2];
    
    Tensor* output = tensor_create(input->dims, input->shape, input->requires_grad);
    
    #pragma omp parallel for collapse(2)
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t s = 0; s < seq_len; s++) {
            for (uint32_t d = 0; d < d_model; d++) {
                output->data[(b * seq_len + s) * d_model + d] = 
                    input->data[(b * seq_len + s) * d_model + d] + 
                    pe->encoding->data[s * d_model + d];
            }
        }
    }
    
    return output;
}

void positional_encoding_free(PositionalEncoding* pe) {
    tensor_free(pe->encoding);
    free(pe);
}

// ============================================================
// Cross-Attention (for Encoder-Decoder)
// ============================================================

Tensor* cross_attention_forward(const Tensor* Q, const Tensor* K, const Tensor* V,
                                const Tensor* mask, float dropout_p, bool training) {
    // Q from decoder: [batch, tgt_len, d_k]
    // K, V from encoder: [batch, src_len, d_k]
    
    uint32_t batch_size = Q->shape[0];
    uint32_t tgt_len = Q->shape[1];
    uint32_t src_len = K->shape[1];
    uint32_t d_k = Q->shape[2];
    
    // Compute Q @ K^T
    uint32_t scores_shape[3] = {batch_size, tgt_len, src_len};
    Tensor* scores = tensor_create(3, scores_shape, Q->requires_grad);
    
    float scale = 1.0f / sqrtf((float)d_k);
    
    #pragma omp parallel for collapse(2)
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t i = 0; i < tgt_len; i++) {
            for (uint32_t j = 0; j < src_len; j++) {
                float sum = 0.0f;
                
                for (uint32_t k = 0; k < d_k; k++) {
                    sum += Q->data[(b * tgt_len + i) * d_k + k] *
                          K->data[(b * src_len + j) * d_k + k];
                }
                
                scores->data[(b * tgt_len + i) * src_len + j] = sum * scale;
            }
        }
    }
    
    // Apply mask if provided
    if (mask) {
        #pragma omp parallel for
        for (uint32_t i = 0; i < scores->size; i++) {
            if (mask->data[i] == 0.0f) {
                scores->data[i] = -1e9f;
            }
        }
    }
    
    // Softmax
    #pragma omp parallel for collapse(2)
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t i = 0; i < tgt_len; i++) {
            float* row = &scores->data[(b * tgt_len + i) * src_len];
            
            float max_val = -FLT_MAX;
            for (uint32_t j = 0; j < src_len; j++) {
                if (row[j] > max_val) max_val = row[j];
            }
            
            float sum_exp = 0.0f;
            for (uint32_t j = 0; j < src_len; j++) {
                row[j] = expf(row[j] - max_val);
                sum_exp += row[j];
            }
            
            for (uint32_t j = 0; j < src_len; j++) {
                row[j] /= sum_exp;
            }
        }
    }
    
    // Apply dropout if training
    if (training && dropout_p > 0.0f) {
        tensor_dropout(scores, scores, dropout_p, true);
    }
    
    // Compute scores @ V
    uint32_t d_v = V->shape[2];
    uint32_t output_shape[3] = {batch_size, tgt_len, d_v};
    Tensor* output = tensor_create(3, output_shape, Q->requires_grad);
    
    #pragma omp parallel for collapse(2)
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t i = 0; i < tgt_len; i++) {
            for (uint32_t k = 0; k < d_v; k++) {
                float sum = 0.0f;
                
                for (uint32_t j = 0; j < src_len; j++) {
                    sum += scores->data[(b * tgt_len + i) * src_len + j] *
                          V->data[(b * src_len + j) * d_v + k];
                }
                
                output->data[(b * tgt_len + i) * d_v + k] = sum;
            }
        }
    }
    
    tensor_free(scores);
    return output;
}

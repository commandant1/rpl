/*
 * RPiTorch Attention Mechanisms
 * Scaled Dot-Product Attention, Multi-Head Attention
 */

#include "rpl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// Forward-declare fast NEON exp from rpl_activations.c (only used internally)
#if RPITORCH_HAS_NEON
#include <arm_neon.h>
static inline float32x4_t fast_exp_neon(float32x4_t x) {
    // 4th-order minimax, same coefficients as rpl_activations.c
    const float32x4_t LOG2E  = vdupq_n_f32(1.442695040f);
    const float32x4_t C1     = vdupq_n_f32(0.0136779459f);
    const float32x4_t C2     = vdupq_n_f32(0.0517869298f);
    const float32x4_t C3     = vdupq_n_f32(0.2413797378f);
    const float32x4_t C4     = vdupq_n_f32(0.6930230856f);
    const float32x4_t ONE    = vdupq_n_f32(1.0f);
    x = vmaxq_f32(vminq_f32(x, vdupq_n_f32(87.0f)), vdupq_n_f32(-87.0f));
    float32x4_t t  = vmulq_f32(x, LOG2E);
    float32x4_t k  = vrndmq_f32(t);
    float32x4_t f  = vsubq_f32(t, k);
    float32x4_t ef = vfmaq_f32(C2, f, C1);
    ef = vfmaq_f32(C3, f, ef);
    ef = vfmaq_f32(C4, f, ef);
    ef = vfmaq_f32(ONE, f, ef);
    int32x4_t ki = vaddq_s32(vcvtq_s32_f32(k), vdupq_n_s32(127));
    return vmulq_f32(vreinterpretq_f32_s32(vshlq_n_s32(ki, 23)), ef);
}
#endif

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

    uint32_t scores_shape[3] = {batch_size, seq_len_q, seq_len_k};
    Tensor* scores = tensor_create(3, scores_shape, Q->requires_grad);

    float scale = 1.0f / sqrtf((float)d_k);

    // ── Q @ Kᵀ  (NEON 8-wide unrolled) ──────────────────────────────────────
    #pragma omp parallel for collapse(2) schedule(static)
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t i = 0; i < seq_len_q; i++) {
            const float* q_row = &Q->data[b * seq_len_q * d_k + i * d_k];
            float* s_row = &scores->data[b * seq_len_q * seq_len_k + i * seq_len_k];

            for (uint32_t j = 0; j < seq_len_k; j++) {
                const float* k_row = &K->data[b * seq_len_k * d_k + j * d_k];
#if RPITORCH_HAS_NEON
                float32x4_t v0 = vdupq_n_f32(0.f), v1 = v0;
                uint32_t d = 0;
                for (; d + 8 <= d_k; d += 8) {
                    v0 = vfmaq_f32(v0, vld1q_f32(&q_row[d]),   vld1q_f32(&k_row[d]));
                    v1 = vfmaq_f32(v1, vld1q_f32(&q_row[d+4]), vld1q_f32(&k_row[d+4]));
                }
                float sum = vaddvq_f32(vaddq_f32(v0, v1));
                for (; d < d_k; d++) sum += q_row[d] * k_row[d];
#else
                float sum = 0.f;
                for (uint32_t d = 0; d < d_k; d++) sum += q_row[d] * k_row[d];
#endif
                s_row[j] = sum * scale;
            }
        }
    }

    // Apply mask
    if (mask) {
        #pragma omp parallel for
        for (uint32_t i = 0; i < scores->size; i++)
            if (mask->data[i] == 0.0f) scores->data[i] = -1e9f;
    }

    // ── Online softmax (numerically stable, NEON exp) ─────────────────────────
    #pragma omp parallel for collapse(2) schedule(static)
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t i = 0; i < seq_len_q; i++) {
            float* row = &scores->data[b * seq_len_q * seq_len_k + i * seq_len_k];

            // Max (scalar — cache-friendly sequential scan)
            float max_val = -FLT_MAX;
            for (uint32_t j = 0; j < seq_len_k; j++)
                if (row[j] > max_val) max_val = row[j];

#if RPITORCH_HAS_NEON
            float32x4_t vmax = vdupq_n_f32(max_val);
            float sum_exp = 0.f;
            float32x4_t vs0 = vdupq_n_f32(0.f), vs1 = vs0;
            uint32_t j = 0;
            for (; j + 8 <= seq_len_k; j += 8) {
                float32x4_t e0 = fast_exp_neon(vsubq_f32(vld1q_f32(&row[j]),   vmax));
                float32x4_t e1 = fast_exp_neon(vsubq_f32(vld1q_f32(&row[j+4]), vmax));
                vst1q_f32(&row[j],   e0); vst1q_f32(&row[j+4], e1);
                vs0 = vaddq_f32(vs0, e0); vs1 = vaddq_f32(vs1, e1);
            }
            sum_exp = vaddvq_f32(vaddq_f32(vs0, vs1));
            for (; j < seq_len_k; j++) {
                row[j] = expf(row[j] - max_val);
                sum_exp += row[j];
            }
#else
            float sum_exp = 0.f;
            for (uint32_t j = 0; j < seq_len_k; j++) {
                row[j] = expf(row[j] - max_val);
                sum_exp += row[j];
            }
#endif
            float inv_sum = 1.f / sum_exp;
#if RPITORCH_HAS_NEON
            float32x4_t vinv = vdupq_n_f32(inv_sum);
            uint32_t jn = 0;
            for (; jn + 8 <= seq_len_k; jn += 8) {
                vst1q_f32(&row[jn],   vmulq_f32(vld1q_f32(&row[jn]),   vinv));
                vst1q_f32(&row[jn+4], vmulq_f32(vld1q_f32(&row[jn+4]), vinv));
            }
            for (; jn < seq_len_k; jn++) row[jn] *= inv_sum;
#else
            for (uint32_t jn = 0; jn < seq_len_k; jn++) row[jn] *= inv_sum;
#endif
        }
    }

    if (training && dropout_p > 0.0f)
        tensor_dropout(scores, scores, dropout_p, true);

    // ── scores @ V  (NEON 8-wide row-major accumulation) ─────────────────────
    uint32_t d_v = V->shape[2];
    uint32_t output_shape[3] = {batch_size, seq_len_q, d_v};
    Tensor* output = tensor_create(3, output_shape, Q->requires_grad);
    tensor_fill(output, 0.f);

    #pragma omp parallel for collapse(2) schedule(static)
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t i = 0; i < seq_len_q; i++) {
            const float* s_row = &scores->data[b * seq_len_q * seq_len_k + i * seq_len_k];
            float*       o_row = &output->data[b * seq_len_q * d_v + i * d_v];

            for (uint32_t j = 0; j < seq_len_k; j++) {
                const float* v_row = &V->data[b * seq_len_k * d_v + j * d_v];
                float w = s_row[j];
#if RPITORCH_HAS_NEON
                float32x4_t vw = vdupq_n_f32(w);
                uint32_t d = 0;
                for (; d + 8 <= d_v; d += 8) {
                    vst1q_f32(&o_row[d],   vfmaq_f32(vld1q_f32(&o_row[d]),   vld1q_f32(&v_row[d]),   vw));
                    vst1q_f32(&o_row[d+4], vfmaq_f32(vld1q_f32(&o_row[d+4]), vld1q_f32(&v_row[d+4]), vw));
                }
                for (; d < d_v; d++) o_row[d] += v_row[d] * w;
#else
                for (uint32_t d = 0; d < d_v; d++) o_row[d] += v_row[d] * w;
#endif
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
// Rotary Position Embedding (RoPE)
// ============================================================

void tensor_rope(Tensor* q, Tensor* k, uint32_t dim_head, float theta) {
    if (!q || !k) return;

    uint32_t seq_len    = q->dims >= 3 ? q->shape[1] : q->shape[0];
    uint32_t num_heads_q = q->dims >= 3 ? q->shape[2] : (q->shape[1] / dim_head);
    uint32_t num_heads_k = k->dims >= 3 ? k->shape[2] : (k->shape[1] / dim_head);
    uint32_t batch_size = q->dims >= 3 ? q->shape[0] : 1;
    uint32_t half_dim   = dim_head / 2;

    // Precompute cos/sin table for all (pos, freq) pairs ONCE.
    // Shape: [seq_len, half_dim] — fits in L1 for typical seq_len ≤ 512, half_dim ≤ 64.
    float* cos_table = (float*)malloc(seq_len * half_dim * sizeof(float));
    float* sin_table = (float*)malloc(seq_len * half_dim * sizeof(float));

    #pragma omp parallel for schedule(static)
    for (uint32_t s = 0; s < seq_len; s++) {
        for (uint32_t d = 0; d < half_dim; d++) {
            float freq = 1.0f / powf(theta, (2.0f * d) / dim_head);
            float angle = (float)s * freq;
            cos_table[s * half_dim + d] = cosf(angle);
            sin_table[s * half_dim + d] = sinf(angle);
        }
    }

    // Apply RoPE to Q
    #pragma omp parallel for collapse(3) schedule(static)
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t s = 0; s < seq_len; s++) {
            for (uint32_t h = 0; h < num_heads_q; h++) {
                float* qh = &q->data[b * seq_len * num_heads_q * dim_head
                                     + s * num_heads_q * dim_head + h * dim_head];
                const float* cv = &cos_table[s * half_dim];
                const float* sv = &sin_table[s * half_dim];

                // Rotate pairs (d, d+half_dim) using precomputed table
                uint32_t d = 0;
#if RPITORCH_HAS_NEON
                for (; d + 4 <= half_dim; d += 4) {
                    float32x4_t q0 = vld1q_f32(&qh[d]);
                    float32x4_t q1 = vld1q_f32(&qh[d + half_dim]);
                    float32x4_t vc = vld1q_f32(&cv[d]);
                    float32x4_t vs = vld1q_f32(&sv[d]);
                    // out0 = q0*cos - q1*sin
                    float32x4_t r0 = vmlsq_f32(vmulq_f32(q0, vc), q1, vs);
                    // out1 = q0*sin + q1*cos
                    float32x4_t r1 = vmlaq_f32(vmulq_f32(q0, vs), q1, vc);
                    vst1q_f32(&qh[d],          r0);
                    vst1q_f32(&qh[d + half_dim], r1);
                }
#endif
                for (; d < half_dim; d++) {
                    float q0 = qh[d], q1 = qh[d + half_dim];
                    qh[d]           = q0 * cv[d] - q1 * sv[d];
                    qh[d + half_dim] = q0 * sv[d] + q1 * cv[d];
                }
            }
        }
    }

    // Apply RoPE to K
    #pragma omp parallel for collapse(3) schedule(static)
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t s = 0; s < seq_len; s++) {
            for (uint32_t h = 0; h < num_heads_k; h++) {
                float* kh = &k->data[b * seq_len * num_heads_k * dim_head
                                     + s * num_heads_k * dim_head + h * dim_head];
                const float* cv = &cos_table[s * half_dim];
                const float* sv = &sin_table[s * half_dim];

                uint32_t d = 0;
#if RPITORCH_HAS_NEON
                for (; d + 4 <= half_dim; d += 4) {
                    float32x4_t k0 = vld1q_f32(&kh[d]);
                    float32x4_t k1 = vld1q_f32(&kh[d + half_dim]);
                    float32x4_t vc = vld1q_f32(&cv[d]);
                    float32x4_t vs = vld1q_f32(&sv[d]);
                    float32x4_t r0 = vmlsq_f32(vmulq_f32(k0, vc), k1, vs);
                    float32x4_t r1 = vmlaq_f32(vmulq_f32(k0, vs), k1, vc);
                    vst1q_f32(&kh[d],          r0);
                    vst1q_f32(&kh[d + half_dim], r1);
                }
#endif
                for (; d < half_dim; d++) {
                    float k0 = kh[d], k1 = kh[d + half_dim];
                    kh[d]           = k0 * cv[d] - k1 * sv[d];
                    kh[d + half_dim] = k0 * sv[d] + k1 * cv[d];
                }
            }
        }
    }

    free(cos_table);
    free(sin_table);
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

// ============================================================
// Gated Attention (DeepSeek-lite / GQA Variant)
// ============================================================

Tensor* gated_attention_forward(const Tensor* Q, const Tensor* K, const Tensor* V,
                                const Tensor* gate, const Tensor* mask) {
    // Q: [batch, seq_len, num_heads_q, d_k]
    // K, V: [batch, seq_len, num_heads_kv, d_k]
    // gate: [batch, seq_len, num_heads_q]

    uint32_t batch_size  = Q->shape[0];
    uint32_t seq_len_q   = Q->shape[1];
    uint32_t seq_len_k   = K->shape[1];
    uint32_t num_heads_q  = Q->shape[2];
    uint32_t num_heads_kv = K->shape[2];
    uint32_t d_k         = Q->shape[3];
    uint32_t group_size  = num_heads_q / num_heads_kv;

    uint32_t output_shape[4] = {batch_size, seq_len_q, num_heads_q, d_k};
    Tensor* output = tensor_create(4, output_shape, Q->requires_grad);
    tensor_fill(output, 0.0f);

    float scale = 1.0f / sqrtf((float)d_k);

    // GQA formulation.
    // scores[] is a VLA allocated once per thread on the stack — avoids
    // per-iteration malloc/free inside the hot parallel region.
    #pragma omp parallel
    {
        // Each thread allocates its own score buffer on its stack.
        // seq_len_k is typically ≤ 2048 (8 KB) — safe for stack.
        float scores_buf[seq_len_k];

        #pragma omp for collapse(3) schedule(dynamic, 1)
        for (uint32_t b = 0; b < batch_size; b++) {
            for (uint32_t h_q = 0; h_q < num_heads_q; h_q++) {
                for (uint32_t i = 0; i < seq_len_q; i++) {
                    uint32_t h_kv   = h_q / group_size;
                    const float* q_vec = &Q->data[((b * seq_len_q + i) * num_heads_q + h_q) * d_k];

                    // 1. Compute scores (causal: j <= i)
                    float max_val = -FLT_MAX;
                    for (uint32_t j = 0; j <= i; j++) {
                        const float* k_vec = &K->data[((b * seq_len_k + j) * num_heads_kv + h_kv) * d_k];
                        float sum = 0.f;
#if RPITORCH_HAS_NEON
                        float32x4_t v0 = vdupq_n_f32(0.f), v1 = v0;
                        uint32_t d = 0;
                        for (; d + 8 <= d_k; d += 8) {
                            v0 = vfmaq_f32(v0, vld1q_f32(&q_vec[d]),   vld1q_f32(&k_vec[d]));
                            v1 = vfmaq_f32(v1, vld1q_f32(&q_vec[d+4]), vld1q_f32(&k_vec[d+4]));
                        }
                        sum = vaddvq_f32(vaddq_f32(v0, v1));
                        for (; d < d_k; d++) sum += q_vec[d] * k_vec[d];
#else
                        for (uint32_t d = 0; d < d_k; d++) sum += q_vec[d] * k_vec[d];
#endif
                        scores_buf[j] = sum * scale;
                        if (scores_buf[j] > max_val) max_val = scores_buf[j];
                    }

                    // 2. Softmax
                    float sum_exp = 0.f;
                    for (uint32_t j = 0; j <= i; j++) {
                        scores_buf[j] = expf(scores_buf[j] - max_val);
                        sum_exp += scores_buf[j];
                    }
                    float inv_sum = 1.f / sum_exp;

                    // 3. Accumulate into output
                    float* out_vec = &output->data[((b * seq_len_q + i) * num_heads_q + h_q) * d_k];
                    float gate_val = gate ? gate->data[(b * seq_len_q + i) * num_heads_q + h_q] : 1.f;

                    for (uint32_t j = 0; j <= i; j++) {
                        float w = scores_buf[j] * inv_sum * gate_val;
                        const float* v_vec = &V->data[((b * seq_len_k + j) * num_heads_kv + h_kv) * d_k];
#if RPITORCH_HAS_NEON
                        float32x4_t vw = vdupq_n_f32(w);
                        uint32_t d = 0;
                        for (; d + 8 <= d_k; d += 8) {
                            vst1q_f32(&out_vec[d],   vfmaq_f32(vld1q_f32(&out_vec[d]),   vld1q_f32(&v_vec[d]),   vw));
                            vst1q_f32(&out_vec[d+4], vfmaq_f32(vld1q_f32(&out_vec[d+4]), vld1q_f32(&v_vec[d+4]), vw));
                        }
                        for (; d < d_k; d++) out_vec[d] += v_vec[d] * w;
#else
                        for (uint32_t d = 0; d < d_k; d++) out_vec[d] += v_vec[d] * w;
#endif
                    }
                }
            }
        }
    } // end omp parallel

    return output;
}

// ============================================================
// Gated DeltaNet (Linear Attention)
// ============================================================

Tensor* gated_deltanet_forward(const Tensor* Q, const Tensor* K, const Tensor* V,
                               const Tensor* gate, const Tensor* beta) {
    // Q, K: [batch, seq_len, num_heads_qk, d_k] (16 heads, d_k=128)
    // V:    [batch, seq_len, num_heads_v, d_v] (16 heads, d_v=128)
    // gate: [batch, seq_len, num_heads_v] (optional)
    // beta: [batch, seq_len, num_heads_qk] (optional decay rates)
    
    uint32_t batch_size = Q->shape[0];
    uint32_t seq_len = Q->shape[1];
    uint32_t num_heads_qk = Q->shape[2];
    uint32_t num_heads_v = V->shape[2];
    uint32_t d_k = Q->shape[3];
    uint32_t d_v = V->shape[3];
    
    uint32_t output_shape[4] = {batch_size, seq_len, num_heads_v, d_v};
    Tensor* output = tensor_create(4, output_shape, Q->requires_grad);
    tensor_fill(output, 0.0f);
    
    // Linear Attention Memory state S: [batch, num_heads, d_k, d_v]
    // Simplifying assuming num_heads_qk == num_heads_v (common in DeltaNet)
    uint32_t h_max = (num_heads_qk > num_heads_v) ? num_heads_qk : num_heads_v;
    
    #pragma omp parallel for
    for (uint32_t b = 0; b < batch_size; b++) {
        // Allocate state S for this batch sequence
        float* S = (float*)calloc(num_heads_v * d_k * d_v, sizeof(float));
        
        for (uint32_t i = 0; i < seq_len; i++) {
            for (uint32_t h = 0; h < h_max; h++) {
                float* q_head = &Q->data[((b * seq_len + i) * num_heads_qk + (h % num_heads_qk)) * d_k];
                float* k_head = &K->data[((b * seq_len + i) * num_heads_qk + (h % num_heads_qk)) * d_k];
                float* v_head = &V->data[((b * seq_len + i) * num_heads_v  + (h % num_heads_v)) * d_v];
                float* out_head = &output->data[((b * seq_len + i) * num_heads_v + (h % num_heads_v)) * d_v];
                float* S_head = &S[(h % num_heads_v) * d_k * d_v];
                
                float beta_val = (beta) ? beta->data[((b * seq_len + i) * num_heads_qk + (h % num_heads_qk))] : 1.0f;
                float gate_val = (gate) ? gate->data[((b * seq_len + i) * num_heads_v + (h % num_heads_v))] : 1.0f;
                
                // Q @ S -> out
                for (uint32_t dk = 0; dk < d_k; dk++) {
                    float q_val = q_head[dk];
                    for (uint32_t dv = 0; dv < d_v; dv++) {
                        out_head[dv] += q_val * S_head[dk * d_v + dv];
                    }
                }
                
                // Apply gate to output
                for (uint32_t dv = 0; dv < d_v; dv++) {
                    out_head[dv] *= gate_val;
                }
                
                // S = beta * S + K^T @ V
                for (uint32_t dk = 0; dk < d_k; dk++) {
                    float k_val = k_head[dk];
                    for (uint32_t dv = 0; dv < d_v; dv++) {
                        S_head[dk * d_v + dv] = (S_head[dk * d_v + dv] * beta_val) + (k_val * v_head[dv]);
                    }
                }
            }
        }
        free(S);
    }
    
    return output;
}

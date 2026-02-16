/*
 * RPiTorch Core Implementation
 * Highly optimized for Cortex-A72 with OpenBLAS, OpenMP, NEON
 */

#include "rpl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <float.h>
#include <assert.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Forward declarations
void tensor_gemm_large(Tensor* C, const Tensor* A, const Tensor* B);
void tensor_gemm_small(Tensor* C, const Tensor* A, const Tensor* B);
void backward_add(Tensor* t);
void backward_mul(Tensor* t);
void backward_matmul(Tensor* t);
void backward_relu(Tensor* t);
void backward_sigmoid(Tensor* t);
void backward_mse(Tensor* t);

// ============================================================
// Memory Management with Alignment
// ============================================================

void* rpitorch_aligned_alloc(size_t alignment, size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) return NULL;
    return ptr;
}

void rpitorch_aligned_free(void* ptr) {
    free(ptr);
}

Tensor* tensor_create(uint32_t dims, const uint32_t* shape, bool requires_grad) {
    Tensor* t = (Tensor*)calloc(1, sizeof(Tensor));
    if (!t) return NULL;
    
    t->dims = dims;
    t->size = 1;
    for (uint32_t i = 0; i < dims; i++) {
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }
    
    // Compute strides
    t->strides[dims-1] = 1;
    for (int i = dims-2; i >= 0; i--) t->strides[i] = t->strides[i+1] * t->shape[i+1];
    
    size_t alloc_size = t->size * sizeof(float);
    t->_allocation = rpitorch_aligned_alloc(64, alloc_size);
    t->data = (float*)t->_allocation;
    
    if (requires_grad) {
        t->grad = (float*)rpitorch_aligned_alloc(64, alloc_size);
        memset(t->grad, 0, alloc_size);
    }
    
    t->requires_grad = requires_grad;
    t->is_leaf = true;
    return t;
}

void tensor_free(Tensor* t) {
    if (!t) return;
    if (t->_allocation) free(t->_allocation);
    if (t->grad) free(t->grad);
    free(t);
}

// ============================================================
// Basic Operations
// ============================================================

void tensor_fill(Tensor* t, float value) {
#if RPITORCH_HAS_NEON
    float32x4_t vval = vdupq_n_f32(value);
    uint32_t i = 0;
    for (; i + 16 <= t->size; i += 16) {
        vst1q_f32(&t->data[i], vval);
        vst1q_f32(&t->data[i+4], vval);
        vst1q_f32(&t->data[i+8], vval);
        vst1q_f32(&t->data[i+12], vval);
    }
    for (; i + 4 <= t->size; i += 4) vst1q_f32(&t->data[i], vval);
    for (; i < t->size; i++) t->data[i] = value;
#else
    #pragma omp parallel for
    for (uint32_t i = 0; i < t->size; i++) t->data[i] = value;
#endif
}

void tensor_randomize(Tensor* t) {
    #pragma omp parallel for
    for (uint32_t i = 0; i < t->size; i++) {
        t->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
}

void tensor_add_out(Tensor* restrict out, const Tensor* restrict a, const Tensor* restrict b) {
    if (a->size == b->size && out->size == a->size) {
#if RPITORCH_HAS_NEON
        const float* restrict pa = a->data;
        const float* restrict pb = b->data;
        float* restrict po = out->data;
        
        #pragma omp parallel for
        for (uint32_t base = 0; base < out->size; base += 512) {
            uint32_t end = (base + 512 < out->size) ? base + 512 : out->size;
            uint32_t i = base;
            
            // Prefetch next block
            __builtin_prefetch(&pa[base + 512], 0, 0);
            __builtin_prefetch(&pb[base + 512], 0, 0);
            
            // 16-wide processing
            for (; i + 16 <= end; i += 16) {
                float32x4_t a0 = vld1q_f32(&pa[i]);
                float32x4_t a1 = vld1q_f32(&pa[i+4]);
                float32x4_t a2 = vld1q_f32(&pa[i+8]);
                float32x4_t a3 = vld1q_f32(&pa[i+12]);
                float32x4_t b0 = vld1q_f32(&pb[i]);
                float32x4_t b1 = vld1q_f32(&pb[i+4]);
                float32x4_t b2 = vld1q_f32(&pb[i+8]);
                float32x4_t b3 = vld1q_f32(&pb[i+12]);
                vst1q_f32(&po[i], vaddq_f32(a0, b0));
                vst1q_f32(&po[i+4], vaddq_f32(a1, b1));
                vst1q_f32(&po[i+8], vaddq_f32(a2, b2));
                vst1q_f32(&po[i+12], vaddq_f32(a3, b3));
            }
            for (; i + 4 <= end; i += 4) {
                vst1q_f32(&po[i], vaddq_f32(vld1q_f32(&pa[i]), vld1q_f32(&pb[i])));
            }
            for (; i < end; i++) po[i] = pa[i] + pb[i];
        }
#else
        #pragma omp parallel for
        for (uint32_t i = 0; i < out->size; i++) out->data[i] = a->data[i] + b->data[i];
#endif
    } else {
        #pragma omp parallel for
        for (uint32_t i = 0; i < out->size; i++) {
            out->data[i] = a->data[i] + b->data[i % b->size];
        }
    }
    
    if (out->requires_grad) {
        out->parent1 = (void*)a;
        out->parent2 = (void*)b;
        out->backward_fn = backward_add;
        out->is_leaf = false;
    }
}

Tensor* tensor_add(const Tensor* a, const Tensor* b) {
    Tensor* out = tensor_create(a->dims, a->shape, a->requires_grad || b->requires_grad);
    tensor_add_out(out, a, b);
    return out;
}

void tensor_mul_out(Tensor* restrict out, const Tensor* restrict a, const Tensor* restrict b) {
    if (a->size == b->size && out->size == a->size) {
#if RPITORCH_HAS_NEON
        const float* restrict pa = a->data;
        const float* restrict pb = b->data;
        float* restrict po = out->data;
        
        #pragma omp parallel for
        for (uint32_t base = 0; base < out->size; base += 512) {
            uint32_t end = (base + 512 < out->size) ? base + 512 : out->size;
            uint32_t i = base;
            
            __builtin_prefetch(&pa[base + 512], 0, 0);
            __builtin_prefetch(&pb[base + 512], 0, 0);
            
            for (; i + 16 <= end; i += 16) {
                float32x4_t a0 = vld1q_f32(&pa[i]);
                float32x4_t a1 = vld1q_f32(&pa[i+4]);
                float32x4_t a2 = vld1q_f32(&pa[i+8]);
                float32x4_t a3 = vld1q_f32(&pa[i+12]);
                float32x4_t b0 = vld1q_f32(&pb[i]);
                float32x4_t b1 = vld1q_f32(&pb[i+4]);
                float32x4_t b2 = vld1q_f32(&pb[i+8]);
                float32x4_t b3 = vld1q_f32(&pb[i+12]);
                vst1q_f32(&po[i], vmulq_f32(a0, b0));
                vst1q_f32(&po[i+4], vmulq_f32(a1, b1));
                vst1q_f32(&po[i+8], vmulq_f32(a2, b2));
                vst1q_f32(&po[i+12], vmulq_f32(a3, b3));
            }
            for (; i + 4 <= end; i += 4) {
                vst1q_f32(&po[i], vmulq_f32(vld1q_f32(&pa[i]), vld1q_f32(&pb[i])));
            }
            for (; i < end; i++) po[i] = pa[i] * pb[i];
        }
#else
        #pragma omp parallel for
        for (uint32_t i = 0; i < out->size; i++) out->data[i] = a->data[i] * b->data[i];
#endif
    } else {
        #pragma omp parallel for
        for (uint32_t i = 0; i < out->size; i++) {
            out->data[i] = a->data[i] * b->data[i % b->size];
        }
    }
    
    if (out->requires_grad) {
        out->parent1 = (void*)a;
        out->parent2 = (void*)b;
        out->backward_fn = backward_mul;
        out->is_leaf = false;
    }
}

Tensor* tensor_mul(const Tensor* a, const Tensor* b) {
    Tensor* out = tensor_create(a->dims, a->shape, a->requires_grad || b->requires_grad);
    tensor_mul_out(out, a, b);
    return out;
}

Tensor* tensor_matmul(const Tensor* a, const Tensor* b) {
    uint32_t M = a->shape[a->dims-2];
    uint32_t N = b->shape[b->dims-1];
    uint32_t shape[2] = {M, N};
    Tensor* out = tensor_create(2, shape, a->requires_grad || b->requires_grad);
    tensor_fill(out, 0.0f);
    
    tensor_gemm(out, a, b, 1.0f, 0.0f, false, false);
    
    if (out->requires_grad) {
        out->parent1 = (void*)a;
        out->parent2 = (void*)b;
        out->backward_fn = backward_matmul;
        out->is_leaf = false;
    }
    return out;
}

// ============================================================
// Optimized GEMM
// ============================================================

void tensor_gemm(Tensor* C, const Tensor* A, const Tensor* B,
                float alpha, float beta, bool trans_a, bool trans_b) {
    uint32_t M = trans_a ? A->shape[1] : A->shape[0];
    uint32_t K = trans_a ? A->shape[0] : A->shape[1];
    uint32_t N = trans_b ? B->shape[0] : B->shape[1];
    
    // Fast path: non-transposed A, transposed B (common for linear layers: A @ B^T)
    // Use optimized GEMM for better performance
    if (!trans_a && trans_b && alpha == 1.0f && beta == 0.0f) {
        // Need to transpose B temporarily for our row-major optimized GEMM
        // For now, we'll use the vectorized loop below which handles trans_b
    }
    
    // Non-transposed case: use optimized GEMM
    if (!trans_a && !trans_b && alpha == 1.0f && beta == 0.0f) {
        tensor_fill(C, 0.0f);
        parallel_gemm_optimized(A->data, B->data, C->data, M, N, K);
        return;
    }
    
    // General case with support for transpose and alpha/beta scaling
#if RPITORCH_HAS_NEON
    #pragma omp parallel for collapse(2)
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            float32x4_t vsum = vdupq_n_f32(0.0f);
            uint32_t k = 0;
            
            for (; k + 4 <= K; k += 4) {
                float a_vals[4], b_vals[4];
                for (int kk = 0; kk < 4; kk++) {
                    a_vals[kk] = trans_a ? A->data[(k+kk)*M + i] : A->data[i*K + k+kk];
                    b_vals[kk] = trans_b ? B->data[j*K + k+kk] : B->data[(k+kk)*N + j];
                }
                float32x4_t va = vld1q_f32(a_vals);
                float32x4_t vb = vld1q_f32(b_vals);
                vsum = vfmaq_f32(vsum, va, vb);
            }
            
            float sum = vaddvq_f32(vsum);
            for (; k < K; k++) {
                float va = trans_a ? A->data[k*M + i] : A->data[i*K + k];
                float vb = trans_b ? B->data[j*K + k] : B->data[k*N + j];
                sum += va * vb;
            }
            
            C->data[i * N + j] = alpha * sum + beta * C->data[i*N + j];
        }
    }
#else
    #pragma omp parallel for collapse(2)
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            float sum = 0;
            for (uint32_t k = 0; k < K; k++) {
                float va = trans_a ? A->data[k*M + i] : A->data[i*K + k];
                float vb = trans_b ? B->data[j*K + k] : B->data[k*N + j];
                sum += va * vb;
            }
            C->data[i * N + j] = alpha * sum + beta * C->data[i*N + j];
        }
    }
#endif
}

// ============================================================
// Activations
// ============================================================

void tensor_relu_inplace(Tensor* t) {
#if RPITORCH_HAS_NEON
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    
    #pragma omp parallel for
    for (uint32_t base = 0; base < t->size; base += 512) {
        uint32_t end = (base + 512 < t->size) ? base + 512 : t->size;
        uint32_t k = base;
        
        // Process 16 elements per iteration (4 registers)
        for (; k + 16 <= end; k += 16) {
            float32x4_t v0 = vmaxq_f32(vld1q_f32(&t->data[k]), vzero);
            float32x4_t v1 = vmaxq_f32(vld1q_f32(&t->data[k+4]), vzero);
            float32x4_t v2 = vmaxq_f32(vld1q_f32(&t->data[k+8]), vzero);
            float32x4_t v3 = vmaxq_f32(vld1q_f32(&t->data[k+12]), vzero);
            vst1q_f32(&t->data[k], v0);
            vst1q_f32(&t->data[k+4], v1);
            vst1q_f32(&t->data[k+8], v2);
            vst1q_f32(&t->data[k+12], v3);
        }
        // Tail: 4 elements
        for (; k + 4 <= end; k += 4) {
            vst1q_f32(&t->data[k], vmaxq_f32(vld1q_f32(&t->data[k]), vzero));
        }
        // Scalar tail
        for (; k < end; k++) if (t->data[k] < 0) t->data[k] = 0;
    }
#else
    #pragma omp parallel for
    for (uint32_t i = 0; i < t->size; i++) if (t->data[i] < 0) t->data[i] = 0;
#endif
    
    if (t->requires_grad && !t->backward_fn) {
        t->parent1 = (void*)t;
        t->backward_fn = backward_relu;
    }
}

void tensor_sigmoid_inplace(Tensor* t) {
#if RPITORCH_HAS_NEON
    // Fast sigmoid using polynomial approximation
    const float32x4_t LOG2E = vdupq_n_f32(1.442695040f);
    const float32x4_t C1 = vdupq_n_f32(0.240226507f);
    const float32x4_t C2 = vdupq_n_f32(0.452920674f);
    const float32x4_t C3 = vdupq_n_f32(0.713483036f);
    const float32x4_t ONE = vdupq_n_f32(1.0f);
    const float32x4_t CLAMP_LO = vdupq_n_f32(-87.0f);
    const float32x4_t CLAMP_HI = vdupq_n_f32(87.0f);
    
    #pragma omp parallel for
    for (uint32_t base = 0; base < t->size; base += 256) {
        uint32_t end = (base + 256 < t->size) ? base + 256 : t->size;
        uint32_t i = base;
        
        for (; i + 4 <= end; i += 4) {
            float32x4_t x = vld1q_f32(&t->data[i]);
            float32x4_t neg_x = vnegq_f32(x);
            
            // Fast exp(-x) approximation
            neg_x = vmaxq_f32(neg_x, CLAMP_LO);
            neg_x = vminq_f32(neg_x, CLAMP_HI);
            float32x4_t tx = vmulq_f32(neg_x, LOG2E);
            float32x4_t k = vrndmq_f32(tx);
            float32x4_t f = vsubq_f32(tx, k);
            float32x4_t exp_f = vfmaq_f32(C2, f, C1);
            exp_f = vfmaq_f32(C3, f, exp_f);
            exp_f = vfmaq_f32(ONE, f, exp_f);
            int32x4_t k_int = vcvtq_s32_f32(k);
            k_int = vaddq_s32(k_int, vdupq_n_s32(127));
            k_int = vshlq_n_s32(k_int, 23);
            float32x4_t exp_neg = vmulq_f32(vreinterpretq_f32_s32(k_int), exp_f);
            
            // sigmoid = 1 / (1 + exp(-x))
            float32x4_t denom = vaddq_f32(ONE, exp_neg);
            float32x4_t recip = vrecpeq_f32(denom);
            recip = vmulq_f32(recip, vrecpsq_f32(denom, recip));
            vst1q_f32(&t->data[i], recip);
        }
        
        for (; i < end; i++) {
            t->data[i] = 1.0f / (1.0f + expf(-t->data[i]));
        }
    }
#else
    #pragma omp parallel for
    for (uint32_t i = 0; i < t->size; i++) t->data[i] = 1.0f / (1.0f + expf(-t->data[i]));
#endif
    
    if (t->requires_grad && !t->backward_fn) {
        t->parent1 = (void*)t;
        t->backward_fn = backward_sigmoid;
    }
}

Tensor* tensor_relu(const Tensor* t) {
    Tensor* out = tensor_create(t->dims, t->shape, t->requires_grad);
    
    #if RPITORCH_HAS_NEON
        #pragma omp parallel for
        for (uint32_t base = 0; base < t->size; base += 1024) {
            uint32_t end = (base + 1024 < t->size) ? base + 1024 : t->size;
            float32x4_t vzero = vdupq_n_f32(0.0f);
            uint32_t k = base;
            for (; k + 4 <= end; k += 4) {
                 float32x4_t val = vld1q_f32(&t->data[k]);
                 vst1q_f32(&out->data[k], vmaxq_f32(val, vzero));
            }
            for (; k < end; k++) out->data[k] = (t->data[k] > 0) ? t->data[k] : 0;
        }
    #else
        #pragma omp parallel for
        for (uint32_t i = 0; i < t->size; i++) out->data[i] = (t->data[i] > 0) ? t->data[i] : 0;
    #endif

    if (out->requires_grad) {
        out->parent1 = (void*)t;
        out->backward_fn = backward_relu;
        out->is_leaf = false;
    }
    return out;
}

Tensor* tensor_sigmoid(const Tensor* t) {
    Tensor* out = tensor_create(t->dims, t->shape, t->requires_grad);
    
#if RPITORCH_HAS_NEON
    const float32x4_t LOG2E = vdupq_n_f32(1.442695040f);
    const float32x4_t C1 = vdupq_n_f32(0.240226507f);
    const float32x4_t C2 = vdupq_n_f32(0.452920674f);
    const float32x4_t C3 = vdupq_n_f32(0.713483036f);
    const float32x4_t ONE = vdupq_n_f32(1.0f);
    const float32x4_t CLAMP_LO = vdupq_n_f32(-87.0f);
    const float32x4_t CLAMP_HI = vdupq_n_f32(87.0f);
    
    #pragma omp parallel for
    for (uint32_t base = 0; base < t->size; base += 256) {
        uint32_t end = (base + 256 < t->size) ? base + 256 : t->size;
        uint32_t i = base;
        
        for (; i + 4 <= end; i += 4) {
            float32x4_t x = vld1q_f32(&t->data[i]);
            float32x4_t neg_x = vnegq_f32(x);
            neg_x = vmaxq_f32(neg_x, CLAMP_LO);
            neg_x = vminq_f32(neg_x, CLAMP_HI);
            float32x4_t tx = vmulq_f32(neg_x, LOG2E);
            float32x4_t k = vrndmq_f32(tx);
            float32x4_t f = vsubq_f32(tx, k);
            float32x4_t exp_f = vfmaq_f32(C2, f, C1);
            exp_f = vfmaq_f32(C3, f, exp_f);
            exp_f = vfmaq_f32(ONE, f, exp_f);
            int32x4_t k_int = vcvtq_s32_f32(k);
            k_int = vaddq_s32(k_int, vdupq_n_s32(127));
            k_int = vshlq_n_s32(k_int, 23);
            float32x4_t exp_neg = vmulq_f32(vreinterpretq_f32_s32(k_int), exp_f);
            float32x4_t denom = vaddq_f32(ONE, exp_neg);
            float32x4_t recip = vrecpeq_f32(denom);
            recip = vmulq_f32(recip, vrecpsq_f32(denom, recip));
            vst1q_f32(&out->data[i], recip);
        }
        for (; i < end; i++) {
            out->data[i] = 1.0f / (1.0f + expf(-t->data[i]));
        }
    }
#else
    #pragma omp parallel for
    for (uint32_t i = 0; i < t->size; i++) out->data[i] = 1.0f / (1.0f + expf(-t->data[i]));
#endif
    
    if (out->requires_grad) {
        out->parent1 = (void*)t;
        out->backward_fn = backward_sigmoid;
        out->is_leaf = false;
    }
    return out;
}

// ============================================================
// Autograd Implementation
// ============================================================

void backward_add(Tensor* t) {
    Tensor* a = (Tensor*)t->parent1;
    Tensor* b = (Tensor*)t->parent2;
    if (a && a->grad) {
        #pragma omp parallel for
        for (uint32_t i = 0; i < t->size; i++) a->grad[i] += t->grad[i];
        if (a->backward_fn) a->backward_fn(a);
    }
    if (b && b->grad) {
        // Handle broadcasting in backward
        #pragma omp parallel for
        for (uint32_t i = 0; i < b->size; i++) {
            float g = 0;
            for (uint32_t j = i; j < t->size; j += b->size) g += t->grad[j];
            b->grad[i] += g;
        }
        if (b->backward_fn) b->backward_fn(b);
    }
}

void backward_mul(Tensor* t) {
    Tensor* a = (Tensor*)t->parent1;
    Tensor* b = (Tensor*)t->parent2;
    if (a && a->grad) {
        #pragma omp parallel for
        for (uint32_t i = 0; i < t->size; i++) {
            float val_b = b->data[i % b->size];
            a->grad[i] += t->grad[i] * val_b;
        }
        if (a->backward_fn) a->backward_fn(a);
    }
    if (b && b->grad) {
        #pragma omp parallel for
        for (uint32_t i = 0; i < b->size; i++) {
            float g = 0;
            for (uint32_t j = i; j < t->size; j += b->size) {
                g += t->grad[j] * a->data[j];
            }
            b->grad[i] += g;
        }
        if (b->backward_fn) b->backward_fn(b);
    }
}

void backward_matmul(Tensor* t) {
    Tensor* a = (Tensor*)t->parent1;
    Tensor* b = (Tensor*)t->parent2;
    uint32_t M = t->shape[0];
    uint32_t N = t->shape[1];
    uint32_t K = a->shape[1]; // Assuming a is [M, K]
    
    // Guess if b was transposed based on shapes
    bool b_trans = (b->shape[0] == N && b->shape[1] == K);
    
    if (a->grad) {
        #pragma omp parallel for collapse(2)
        for (uint32_t i = 0; i < M; i++) {
            for (uint32_t k = 0; k < K; k++) {
                float sum = 0;
                for (uint32_t j = 0; j < N; j++) {
                    float val_b = b_trans ? b->data[j * K + k] : b->data[k * N + j];
                    sum += t->grad[i * N + j] * val_b;
                }
                a->grad[i * K + k] += sum;
            }
        }
        if (a->backward_fn) a->backward_fn(a);
    }
    
    if (b->grad) {
        if (b_trans) {
            // d (a @ b^T) / db_jk = sum_i d t_ij * a_ik
            #pragma omp parallel for collapse(2)
            for (uint32_t j = 0; j < N; j++) {
                for (uint32_t k = 0; k < K; k++) {
                    float sum = 0;
                    for (uint32_t i = 0; i < M; i++) {
                        sum += t->grad[i * N + j] * a->data[i * K + k];
                    }
                    b->grad[j * K + k] += sum;
                }
            }
        } else {
            // d (a @ b) / db_kj = sum_i d t_ij * a_ik
            #pragma omp parallel for collapse(2)
            for (uint32_t k = 0; k < K; k++) {
                for (uint32_t j = 0; j < N; j++) {
                    float sum = 0;
                    for (uint32_t i = 0; i < M; i++) {
                        sum += a->data[i * K + k] * t->grad[i * N + j];
                    }
                    b->grad[k * N + j] += sum;
                }
            }
        }
        if (b->backward_fn) b->backward_fn(b);
    }
}

void backward_relu(Tensor* t) {
    Tensor* a = (Tensor*)t->parent1;
    if (a && a->grad) {
        for (uint32_t i = 0; i < t->size; i++) if (a->data[i] > 0) a->grad[i] += t->grad[i];
        if (a->backward_fn) a->backward_fn(a);
    }
}

void backward_sigmoid(Tensor* t) {
    Tensor* a = (Tensor*)t->parent1;
    if (a && a->grad) {
        for (uint32_t i = 0; i < t->size; i++) {
            float s = t->data[i];
            a->grad[i] += t->grad[i] * s * (1.0f - s);
        }
        if (a->backward_fn) a->backward_fn(a);
    }
}

void backward_mse(Tensor* t) {
    Tensor* pred = (Tensor*)t->parent1;
    Tensor* target = (Tensor*)t->parent2;
    if (pred && pred->grad) {
        float factor = 2.0f / pred->size;
        for (uint32_t i = 0; i < pred->size; i++) {
            pred->grad[i] += t->grad[0] * factor * (pred->data[i] - target->data[i]);
        }
        if (pred->backward_fn) pred->backward_fn(pred);
    }
}

Tensor* tensor_mse_loss(const Tensor* pred, const Tensor* target) {
    uint32_t shape[1] = {1};
    Tensor* out = tensor_create(1, shape, pred->requires_grad);
    float loss = 0;
    for (uint32_t i = 0; i < pred->size; i++) {
        float d = pred->data[i] - target->data[i];
        loss += d * d;
    }
    out->data[0] = loss / pred->size;
    if (out->requires_grad) {
        out->parent1 = (void*)pred;
        out->parent2 = (void*)target;
        out->backward_fn = backward_mse;
        out->is_leaf = false;
    }
    return out;
}

void tensor_backward(Tensor* t) {
    if (!t->requires_grad) return;
    if (!t->grad) {
        t->grad = (float*)rpitorch_aligned_alloc(64, t->size * sizeof(float));
    }
    // Initialize root gradients to 1.0
    for (uint32_t i = 0; i < t->size; i++) t->grad[i] = 1.0f;
    
    if (t->backward_fn) t->backward_fn(t);
}

void tensor_zero_grad(Tensor* t) {
    if (t->grad) memset(t->grad, 0, t->size * sizeof(float));
}

// Placeholders for other things used in the library
void tensor_add_inplace(Tensor* a, const Tensor* b) { tensor_add_out(a, a, b); }
void tensor_mul_inplace(Tensor* a, float scalar) {
    for (uint32_t i = 0; i < a->size; i++) a->data[i] *= scalar;
}
void tensor_fill_buffer(float* buffer, float value, uint32_t size) {
    for (uint32_t i = 0; i < size; i++) buffer[i] = value;
}
void tensor_tanh_inplace(Tensor* t) {
#if RPITORCH_HAS_NEON
    // tanh(x) = 2*sigmoid(2x) - 1
    const float32x4_t TWO = vdupq_n_f32(2.0f);
    const float32x4_t ONE = vdupq_n_f32(1.0f);
    const float32x4_t LOG2E = vdupq_n_f32(1.442695040f);
    const float32x4_t C1 = vdupq_n_f32(0.240226507f);
    const float32x4_t C2 = vdupq_n_f32(0.452920674f);
    const float32x4_t C3 = vdupq_n_f32(0.713483036f);
    
    #pragma omp parallel for
    for (uint32_t base = 0; base < t->size; base += 256) {
        uint32_t end = (base + 256 < t->size) ? base + 256 : t->size;
        uint32_t i = base;
        
        for (; i + 4 <= end; i += 4) {
            float32x4_t x = vld1q_f32(&t->data[i]);
            float32x4_t x2 = vmulq_f32(TWO, x);
            float32x4_t neg_x2 = vnegq_f32(x2);
            neg_x2 = vmaxq_f32(neg_x2, vdupq_n_f32(-87.0f));
            neg_x2 = vminq_f32(neg_x2, vdupq_n_f32(87.0f));
            float32x4_t tx = vmulq_f32(neg_x2, LOG2E);
            float32x4_t k = vrndmq_f32(tx);
            float32x4_t f = vsubq_f32(tx, k);
            float32x4_t exp_f = vfmaq_f32(C2, f, C1);
            exp_f = vfmaq_f32(C3, f, exp_f);
            exp_f = vfmaq_f32(ONE, f, exp_f);
            int32x4_t k_int = vcvtq_s32_f32(k);
            k_int = vaddq_s32(k_int, vdupq_n_s32(127));
            k_int = vshlq_n_s32(k_int, 23);
            float32x4_t exp_neg = vmulq_f32(vreinterpretq_f32_s32(k_int), exp_f);
            float32x4_t denom = vaddq_f32(ONE, exp_neg);
            float32x4_t recip = vrecpeq_f32(denom);
            recip = vmulq_f32(recip, vrecpsq_f32(denom, recip));
            float32x4_t result = vsubq_f32(vmulq_f32(TWO, recip), ONE);
            vst1q_f32(&t->data[i], result);
        }
        for (; i < end; i++) {
            t->data[i] = tanhf(t->data[i]);
        }
    }
#else
    #pragma omp parallel for
    for (uint32_t i = 0; i < t->size; i++) t->data[i] = tanhf(t->data[i]);
#endif
}

void tensor_gelu_inplace(Tensor* t) {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float SQRT_2_PI = 0.7978845608f;  // sqrt(2/pi)
    const float A = 0.044715f;
    
#if RPITORCH_HAS_NEON
    const float32x4_t HALF = vdupq_n_f32(0.5f);
    const float32x4_t ONE = vdupq_n_f32(1.0f);
    const float32x4_t TWO = vdupq_n_f32(2.0f);
    const float32x4_t SQRT2PI = vdupq_n_f32(SQRT_2_PI);
    const float32x4_t COEFF_A = vdupq_n_f32(A);
    const float32x4_t LOG2E = vdupq_n_f32(1.442695040f);
    const float32x4_t C1 = vdupq_n_f32(0.240226507f);
    const float32x4_t C2 = vdupq_n_f32(0.452920674f);
    const float32x4_t C3 = vdupq_n_f32(0.713483036f);
    
    #pragma omp parallel for
    for (uint32_t base = 0; base < t->size; base += 256) {
        uint32_t end = (base + 256 < t->size) ? base + 256 : t->size;
        uint32_t i = base;
        
        for (; i + 4 <= end; i += 4) {
            float32x4_t x = vld1q_f32(&t->data[i]);
            float32x4_t x3 = vmulq_f32(vmulq_f32(x, x), x);
            float32x4_t inner = vmulq_f32(SQRT2PI, vfmaq_f32(x, COEFF_A, x3));
            
            // tanh(inner) = 2*sigmoid(2*inner) - 1
            float32x4_t inner2 = vmulq_f32(TWO, inner);
            float32x4_t neg_inner2 = vnegq_f32(inner2);
            neg_inner2 = vmaxq_f32(neg_inner2, vdupq_n_f32(-87.0f));
            neg_inner2 = vminq_f32(neg_inner2, vdupq_n_f32(87.0f));
            float32x4_t tx = vmulq_f32(neg_inner2, LOG2E);
            float32x4_t k = vrndmq_f32(tx);
            float32x4_t f = vsubq_f32(tx, k);
            float32x4_t exp_f = vfmaq_f32(C2, f, C1);
            exp_f = vfmaq_f32(C3, f, exp_f);
            exp_f = vfmaq_f32(ONE, f, exp_f);
            int32x4_t k_int = vcvtq_s32_f32(k);
            k_int = vaddq_s32(k_int, vdupq_n_s32(127));
            k_int = vshlq_n_s32(k_int, 23);
            float32x4_t exp_neg = vmulq_f32(vreinterpretq_f32_s32(k_int), exp_f);
            float32x4_t denom = vaddq_f32(ONE, exp_neg);
            float32x4_t recip = vrecpeq_f32(denom);
            recip = vmulq_f32(recip, vrecpsq_f32(denom, recip));
            float32x4_t tanh_val = vsubq_f32(vmulq_f32(TWO, recip), ONE);
            
            // GELU = 0.5 * x * (1 + tanh)
            float32x4_t result = vmulq_f32(vmulq_f32(HALF, x), vaddq_f32(ONE, tanh_val));
            vst1q_f32(&t->data[i], result);
        }
        for (; i < end; i++) {
            float x = t->data[i];
            t->data[i] = 0.5f * x * (1.0f + tanhf(SQRT_2_PI * (x + A * x * x * x)));
        }
    }
#else
    #pragma omp parallel for
    for (uint32_t i = 0; i < t->size; i++) {
        float x = t->data[i];
        t->data[i] = 0.5f * x * (1.0f + tanhf(SQRT_2_PI * (x + A * x * x * x)));
    }
#endif
}
void tensor_softmax_inplace(Tensor* t) {
    uint32_t last_dim = t->shape[t->dims - 1];
    uint32_t num_rows = t->size / last_dim;
    
#if RPITORCH_HAS_NEON
    const float32x4_t LOG2E = vdupq_n_f32(1.442695040f);
    const float32x4_t C1 = vdupq_n_f32(0.240226507f);
    const float32x4_t C2 = vdupq_n_f32(0.452920674f);
    const float32x4_t C3 = vdupq_n_f32(0.713483036f);
    const float32x4_t ONE = vdupq_n_f32(1.0f);
    
    #pragma omp parallel for
    for (uint32_t r = 0; r < num_rows; r++) {
        float* row = &t->data[r * last_dim];
        
        // Find max (NEON reduction)
        float32x4_t vmax = vdupq_n_f32(-FLT_MAX);
        uint32_t i = 0;
        for (; i + 4 <= last_dim; i += 4) {
            float32x4_t v = vld1q_f32(&row[i]);
            vmax = vmaxq_f32(vmax, v);
        }
        float max_val = vmaxvq_f32(vmax);
        for (; i < last_dim; i++) if (row[i] > max_val) max_val = row[i];
        
        // Compute exp(x - max) and sum
        float32x4_t vmax_scalar = vdupq_n_f32(max_val);
        float32x4_t vsum = vdupq_n_f32(0.0f);
        for (i = 0; i + 4 <= last_dim; i += 4) {
            float32x4_t v = vsubq_f32(vld1q_f32(&row[i]), vmax_scalar);
            v = vmaxq_f32(v, vdupq_n_f32(-87.0f));
            v = vminq_f32(v, vdupq_n_f32(87.0f));
            float32x4_t tx = vmulq_f32(v, LOG2E);
            float32x4_t k = vrndmq_f32(tx);
            float32x4_t f = vsubq_f32(tx, k);
            float32x4_t exp_f = vfmaq_f32(C2, f, C1);
            exp_f = vfmaq_f32(C3, f, exp_f);
            exp_f = vfmaq_f32(ONE, f, exp_f);
            int32x4_t k_int = vcvtq_s32_f32(k);
            k_int = vaddq_s32(k_int, vdupq_n_s32(127));
            k_int = vshlq_n_s32(k_int, 23);
            float32x4_t exp_v = vmulq_f32(vreinterpretq_f32_s32(k_int), exp_f);
            vst1q_f32(&row[i], exp_v);
            vsum = vaddq_f32(vsum, exp_v);
        }
        float sum_exp = vaddvq_f32(vsum);
        for (; i < last_dim; i++) {
            row[i] = expf(row[i] - max_val);
            sum_exp += row[i];
        }
        
        // Normalize
        float32x4_t vinv_sum = vdupq_n_f32(1.0f / sum_exp);
        for (i = 0; i + 4 <= last_dim; i += 4) {
            float32x4_t v = vld1q_f32(&row[i]);
            vst1q_f32(&row[i], vmulq_f32(v, vinv_sum));
        }
        for (; i < last_dim; i++) row[i] /= sum_exp;
    }
#else
    for (uint32_t r = 0; r < num_rows; r++) {
        float* row = &t->data[r * last_dim];
        float max_val = -FLT_MAX;
        for (uint32_t i = 0; i < last_dim; i++) if (row[i] > max_val) max_val = row[i];
        float sum_exp = 0;
        for (uint32_t i = 0; i < last_dim; i++) { row[i] = expf(row[i] - max_val); sum_exp += row[i]; }
        for (uint32_t i = 0; i < last_dim; i++) row[i] /= sum_exp;
    }
#endif
}
QuantizedTensor* tensor_quantize_int8(const Tensor* input, float scale, int32_t zero_point) {
    QuantizedTensor* qt = (QuantizedTensor*)malloc(sizeof(QuantizedTensor));
    qt->size = input->size;
    qt->dims = input->dims;
    memcpy(qt->shape, input->shape, input->dims * sizeof(uint32_t));
    qt->scale = scale;
    qt->zero_point = zero_point;
    
    qt->data = (int8_t*)rpitorch_aligned_alloc(64, qt->size);
    
    #pragma omp parallel for
    for (uint32_t i = 0; i < input->size; i++) {
        int32_t quantized = (int32_t)roundf(input->data[i] / scale) + zero_point;
        if (quantized < -128) quantized = -128;
        if (quantized > 127) quantized = 127;
        qt->data[i] = (int8_t)quantized;
    }
    
    return qt;
}

void tensor_batchnorm2d(Tensor* out, const Tensor* in, float* weight, float* bias, float* running_mean, float* running_var, float eps, bool training, float momentum) {}
void tensor_dropout(Tensor* out, const Tensor* in, float p, bool training) {}
void gemm_init_buffers();
void gemm_free_buffers();
void parallel_gemm_optimized(float* A, float* B, float* C, uint32_t M, uint32_t N, uint32_t K);
void conv2d_winograd_2x2_3x3(const Tensor* input, const Tensor* weight, Tensor* output) {}

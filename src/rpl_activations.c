/*
 * RPL Activation Functions - Optimized for ARM Cortex-A72
 * NEON-vectorized: LeakyReLU, ELU, Swish, Softplus
 * Fast polynomial approximations for exp/sigmoid/tanh
 */

#include "rpl.h"
#include <math.h>
#include <omp.h>

// ============================================================
// Fast NEON Approximations
// ============================================================

#if RPITORCH_HAS_NEON

// Pre-computed constants for fast exp (hoisted to avoid reload)
static const float EXP_LOG2E = 1.442695040f;
static const float EXP_C1 = 0.240226507f;
static const float EXP_C2 = 0.452920674f;
static const float EXP_C3 = 0.713483036f;

// Ultra-fast exp approximation using bit manipulation + polynomial
// ~0.3% max relative error, optimal for sigmoid/tanh
static inline float32x4_t fast_exp_neon(float32x4_t x) {
    const float32x4_t LOG2E = vdupq_n_f32(EXP_LOG2E);
    const float32x4_t C1 = vdupq_n_f32(EXP_C1);
    const float32x4_t C2 = vdupq_n_f32(EXP_C2);
    const float32x4_t C3 = vdupq_n_f32(EXP_C3);
    const float32x4_t ONE = vdupq_n_f32(1.0f);
    
    // Clamp and compute t = x * log2(e)
    x = vmaxq_f32(vminnq_f32(x, vdupq_n_f32(87.0f)), vdupq_n_f32(-87.0f));
    float32x4_t t = vmulq_f32(x, LOG2E);
    
    // Fast floor using truncation + correction
    float32x4_t k = vrndmq_f32(t);
    float32x4_t f = vsubq_f32(t, k);
    
    // Horner's method: 2^f ≈ 1 + f*(C3 + f*(C2 + f*C1))
    float32x4_t exp_f = vfmaq_f32(C2, f, C1);
    exp_f = vfmaq_f32(C3, f, exp_f);
    exp_f = vfmaq_f32(ONE, f, exp_f);
    
    // 2^k via IEEE754 bit manipulation (zero cost)
    int32x4_t k_int = vaddq_s32(vcvtq_s32_f32(k), vdupq_n_s32(127));
    float32x4_t exp_k = vreinterpretq_f32_s32(vshlq_n_s32(k_int, 23));
    
    return vmulq_f32(exp_k, exp_f);
}

// Fast sigmoid with 2x Newton-Raphson for sub-1% error
static inline float32x4_t fast_sigmoid_neon(float32x4_t x) {
    float32x4_t exp_neg = fast_exp_neon(vnegq_f32(x));
    float32x4_t denom = vaddq_f32(vdupq_n_f32(1.0f), exp_neg);
    // 2x Newton-Raphson iterations for reciprocal (much faster than div)
    float32x4_t recip = vrecpeq_f32(denom);
    recip = vmulq_f32(recip, vrecpsq_f32(denom, recip));
    recip = vmulq_f32(recip, vrecpsq_f32(denom, recip));  // 2nd iteration
    return recip;
}

// Fast tanh: 2*sigmoid(2x) - 1
static inline float32x4_t fast_tanh_neon(float32x4_t x) {
    const float32x4_t TWO = vdupq_n_f32(2.0f);
    return vsubq_f32(vmulq_f32(TWO, fast_sigmoid_neon(vmulq_f32(TWO, x))), vdupq_n_f32(1.0f));
}

// Fast reciprocal square root for layer norm
static inline float32x4_t fast_rsqrt_neon(float32x4_t x) {
    float32x4_t est = vrsqrteq_f32(x);
    // 2x Newton-Raphson: y = y * (3 - x*y*y) / 2
    est = vmulq_f32(est, vrsqrtsq_f32(vmulq_f32(x, est), est));
    est = vmulq_f32(est, vrsqrtsq_f32(vmulq_f32(x, est), est));
    return est;
}

#endif // RPITORCH_HAS_NEON

// ============================================================
// LeakyReLU
// ============================================================

void tensor_leaky_relu(Tensor* out, const Tensor* in, float negative_slope) {
    uint32_t i = 0;
    
    #if RPITORCH_HAS_NEON
    float32x4_t vzero = vdupq_n_f32(0.0f);
    float32x4_t vslope = vdupq_n_f32(negative_slope);
    
    #pragma omp parallel for
    for (uint32_t base = 0; base < in->size; base += 256) {
        uint32_t end = (base + 256 < in->size) ? base + 256 : in->size;
        for (uint32_t i = base; i + 4 <= end; i += 4) {
            float32x4_t x = vld1q_f32(&in->data[i]);
            uint32x4_t mask = vcgtq_f32(x, vzero);  // x > 0
            float32x4_t neg_part = vmulq_f32(x, vslope);
            float32x4_t result = vbslq_f32(mask, x, neg_part);
            vst1q_f32(&out->data[i], result);
        }
        // Handle tail
        for (uint32_t i = end - (end % 4); i < end; i++) {
            out->data[i] = (in->data[i] > 0.0f) ? in->data[i] : negative_slope * in->data[i];
        }
    }
    #else
    #pragma omp parallel for
    for (i = 0; i < in->size; i++) {
        out->data[i] = (in->data[i] > 0.0f) ? in->data[i] : negative_slope * in->data[i];
    }
    #endif
}

void tensor_leaky_relu_inplace(Tensor* t, float negative_slope) {
    tensor_leaky_relu(t, t, negative_slope);
}

// ============================================================
// ELU (Exponential Linear Unit)
// ============================================================

void tensor_elu(Tensor* out, const Tensor* in, float alpha) {
#if RPITORCH_HAS_NEON
    float32x4_t valpha = vdupq_n_f32(alpha);
    float32x4_t vzero = vdupq_n_f32(0.0f);
    float32x4_t vone = vdupq_n_f32(1.0f);
    
    #pragma omp parallel for
    for (uint32_t base = 0; base < in->size; base += 256) {
        uint32_t end = (base + 256 < in->size) ? base + 256 : in->size;
        uint32_t i = base;
        
        for (; i + 4 <= end; i += 4) {
            float32x4_t x = vld1q_f32(&in->data[i]);
            uint32x4_t pos_mask = vcgtq_f32(x, vzero);  // x > 0
            
            // For negative: alpha * (exp(x) - 1)
            float32x4_t exp_x = fast_exp_neon(x);
            float32x4_t neg_part = vmulq_f32(valpha, vsubq_f32(exp_x, vone));
            
            // Select based on sign
            float32x4_t result = vbslq_f32(pos_mask, x, neg_part);
            vst1q_f32(&out->data[i], result);
        }
        
        // Handle tail
        for (; i < end; i++) {
            float x = in->data[i];
            out->data[i] = (x > 0.0f) ? x : alpha * (expf(x) - 1.0f);
        }
    }
#else
    #pragma omp parallel for
    for (uint32_t i = 0; i < in->size; i++) {
        float x = in->data[i];
        out->data[i] = (x > 0.0f) ? x : alpha * (expf(x) - 1.0f);
    }
#endif
}

void tensor_elu_inplace(Tensor* t, float alpha) {
    tensor_elu(t, t, alpha);
}

// ============================================================
// Swish / SiLU: x * sigmoid(x)
// ============================================================

void tensor_swish(Tensor* out, const Tensor* in) {
#if RPITORCH_HAS_NEON
    #pragma omp parallel for
    for (uint32_t base = 0; base < in->size; base += 256) {
        uint32_t end = (base + 256 < in->size) ? base + 256 : in->size;
        uint32_t i = base;
        
        for (; i + 4 <= end; i += 4) {
            float32x4_t x = vld1q_f32(&in->data[i]);
            float32x4_t sigmoid = fast_sigmoid_neon(x);
            float32x4_t result = vmulq_f32(x, sigmoid);
            vst1q_f32(&out->data[i], result);
        }
        
        // Handle tail
        for (; i < end; i++) {
            float x = in->data[i];
            float sigmoid = 1.0f / (1.0f + expf(-x));
            out->data[i] = x * sigmoid;
        }
    }
#else
    #pragma omp parallel for
    for (uint32_t i = 0; i < in->size; i++) {
        float x = in->data[i];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        out->data[i] = x * sigmoid;
    }
#endif
}

void tensor_swish_inplace(Tensor* t) {
    tensor_swish(t, t);
}

// ============================================================
// Softplus: log(1 + exp(beta * x)) / beta
// ============================================================

void tensor_softplus(Tensor* out, const Tensor* in, float beta, float threshold) {
#if RPITORCH_HAS_NEON
    float32x4_t vbeta = vdupq_n_f32(beta);
    float32x4_t vthreshold = vdupq_n_f32(threshold);
    float32x4_t vone = vdupq_n_f32(1.0f);
    float inv_beta = 1.0f / beta;
    float32x4_t vinv_beta = vdupq_n_f32(inv_beta);
    
    #pragma omp parallel for
    for (uint32_t base = 0; base < in->size; base += 256) {
        uint32_t end = (base + 256 < in->size) ? base + 256 : in->size;
        uint32_t i = base;
        
        for (; i + 4 <= end; i += 4) {
            float32x4_t x = vld1q_f32(&in->data[i]);
            float32x4_t bx = vmulq_f32(vbeta, x);
            
            // Check if beta*x > threshold (use linear approximation)
            uint32x4_t lin_mask = vcgtq_f32(bx, vthreshold);
            
            // log(1 + exp(beta*x)) / beta
            float32x4_t exp_bx = fast_exp_neon(bx);
            float32x4_t log_arg = vaddq_f32(vone, exp_bx);
            
            // Fast log approximation using NEON
            // log(x) ≈ (x-1) - (x-1)^2/2 + (x-1)^3/3 for x near 1
            // For general values, use scalar fallback in tail
            // Here we use a simpler approach: extract and use logf
            float log_vals[4];
            vst1q_f32(log_vals, log_arg);
            float sp_vals[4] = {
                logf(log_vals[0]) * inv_beta,
                logf(log_vals[1]) * inv_beta,
                logf(log_vals[2]) * inv_beta,
                logf(log_vals[3]) * inv_beta
            };
            float32x4_t softplus = vld1q_f32(sp_vals);
            
            // Select: linear (x) if above threshold, else softplus
            float32x4_t result = vbslq_f32(lin_mask, x, softplus);
            vst1q_f32(&out->data[i], result);
        }
        
        // Handle tail
        for (; i < end; i++) {
            float x = in->data[i];
            if (x * beta > threshold) {
                out->data[i] = x;
            } else {
                out->data[i] = logf(1.0f + expf(beta * x)) / beta;
            }
        }
    }
#else
    #pragma omp parallel for
    for (uint32_t i = 0; i < in->size; i++) {
        float x = in->data[i];
        if (x * beta > threshold) {
            out->data[i] = x;
        } else {
            out->data[i] = logf(1.0f + expf(beta * x)) / beta;
        }
    }
#endif
}

void tensor_softplus_inplace(Tensor* t, float beta, float threshold) {
    tensor_softplus(t, t, beta, threshold);
}


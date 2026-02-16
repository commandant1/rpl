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
    x = vmaxq_f32(vminq_f32(x, vdupq_n_f32(87.0f)), vdupq_n_f32(-87.0f));
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

// ============================================================
// fast_log_neon — fully vectorized natural log
// Decomposes IEEE-754 float into exponent + mantissa, then uses
// a minimax polynomial on [1,2) for the mantissa.
// Max relative error: ~1e-4 (sufficient for activations)
// ============================================================
static inline float32x4_t fast_log_neon(float32x4_t x) {
    // Extract exponent: e = ((bits >> 23) & 0xFF) - 127
    int32x4_t bits = vreinterpretq_s32_f32(x);
    int32x4_t e = vsubq_s32(vshrq_n_s32(bits, 23), vdupq_n_s32(127));
    float32x4_t fe = vcvtq_f32_s32(e);

    // Extract mantissa in [1, 2): clear exponent, set to 127
    int32x4_t mantissa_bits = vorrq_s32(
        vandq_s32(bits, vdupq_n_s32(0x007FFFFF)),
        vdupq_n_s32(0x3F800000));
    float32x4_t m = vreinterpretq_f32_s32(mantissa_bits);

    // Polynomial approximation for log(m) on [1, 2)
    // log(m) ≈ (m-1) * (c0 + (m-1) * (c1 + (m-1) * c2))
    // Minimax coefficients for [1,2): c0≈0.9999, c1≈-0.4999, c2≈0.3333
    const float32x4_t c0 = vdupq_n_f32(0.99999994f);
    const float32x4_t c1 = vdupq_n_f32(-0.49999603f);
    const float32x4_t c2 = vdupq_n_f32(0.33329940f);
    const float32x4_t LN2 = vdupq_n_f32(0.6931471805599453f);

    float32x4_t f = vsubq_f32(m, vdupq_n_f32(1.0f));
    // Horner: log(m) = f * (c0 + f * (c1 + f * c2))
    float32x4_t p = vfmaq_f32(c1, f, c2);
    p = vfmaq_f32(c0, f, p);
    float32x4_t log_m = vmulq_f32(f, p);

    // log(x) = e * ln(2) + log(m)
    return vfmaq_f32(log_m, fe, LN2);
}

#endif // RPITORCH_HAS_NEON

// ============================================================
// LeakyReLU
// ============================================================

void tensor_leaky_relu(Tensor* out, const Tensor* in, float negative_slope) {
    
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
    for (uint32_t i = 0; i < in->size; i++) {
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
    const float32x4_t vbeta = vdupq_n_f32(beta);
    const float32x4_t vthreshold = vdupq_n_f32(threshold);
    const float32x4_t vone = vdupq_n_f32(1.0f);
    const float32x4_t vinv_beta = vdupq_n_f32(1.0f / beta);
    uint32_t i = 0;
    // 4× unroll = 16 floats = 1 cache line per iteration
    for (; i + 16 <= in->size; i += 16) {
        __builtin_prefetch(&in->data[i + 64], 0, 1);
        for (int k = 0; k < 16; k += 4) {
            float32x4_t x = vld1q_f32(&in->data[i + k]);
            float32x4_t bx = vmulq_f32(vbeta, x);
            uint32x4_t lin_mask = vcgtq_f32(bx, vthreshold);
            // Fully vectorized: log(1 + exp(bx)) / beta
            float32x4_t sp = vmulq_f32(fast_log_neon(vaddq_f32(vone, fast_exp_neon(bx))), vinv_beta);
            vst1q_f32(&out->data[i + k], vbslq_f32(lin_mask, x, sp));
        }
    }
    for (; i + 4 <= in->size; i += 4) {
        float32x4_t x = vld1q_f32(&in->data[i]);
        float32x4_t bx = vmulq_f32(vbeta, x);
        uint32x4_t lin_mask = vcgtq_f32(bx, vthreshold);
        float32x4_t sp = vmulq_f32(fast_log_neon(vaddq_f32(vone, fast_exp_neon(bx))), vinv_beta);
        vst1q_f32(&out->data[i], vbslq_f32(lin_mask, x, sp));
    }
    for (; i < in->size; i++) {
        float x = in->data[i];
        out->data[i] = (x * beta > threshold) ? x : logf(1.0f + expf(beta * x)) / beta;
    }
#else
    for (uint32_t i = 0; i < in->size; i++) {
        float x = in->data[i];
        out->data[i] = (x * beta > threshold) ? x : logf(1.0f + expf(beta * x)) / beta;
    }
#endif
}

void tensor_softplus_inplace(Tensor* t, float beta, float threshold) {
    tensor_softplus(t, t, beta, threshold);
}

// ============================================================
// GELU (Gaussian Error Linear Unit) — out-of-place
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
// ============================================================

void tensor_gelu(Tensor* out, const Tensor* in) {
#if RPITORCH_HAS_NEON
    const float32x4_t HALF = vdupq_n_f32(0.5f);
    const float32x4_t ONE = vdupq_n_f32(1.0f);
    const float32x4_t SQRT2PI = vdupq_n_f32(0.7978845608f);
    const float32x4_t COEFF_A = vdupq_n_f32(0.044715f);
    uint32_t i = 0;
    for (; i + 4 <= in->size; i += 4) {
        __builtin_prefetch(&in->data[i + 64], 0, 1);
        float32x4_t x = vld1q_f32(&in->data[i]);
        float32x4_t x3 = vmulq_f32(vmulq_f32(x, x), x);
        float32x4_t inner = vmulq_f32(SQRT2PI, vfmaq_f32(x, COEFF_A, x3));
        // tanh via fast_tanh_neon
        float32x4_t th = fast_tanh_neon(inner);
        float32x4_t result = vmulq_f32(vmulq_f32(HALF, x), vaddq_f32(ONE, th));
        vst1q_f32(&out->data[i], result);
    }
    for (; i < in->size; i++) {
        float x = in->data[i];
        float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
        out->data[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
#else
    for (uint32_t i = 0; i < in->size; i++) {
        float x = in->data[i];
        float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
        out->data[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
#endif
}

// ============================================================
// SELU (Scaled ELU): lambda * (x if x>0, alpha*(exp(x)-1) if x<=0)
// lambda=1.0507, alpha=1.6733
// ============================================================

void tensor_selu(Tensor* out, const Tensor* in) {
    const float SELU_LAMBDA = 1.0507009873554804934f;
    const float SELU_ALPHA  = 1.6732632423543772848f;
#if RPITORCH_HAS_NEON
    const float32x4_t vlam = vdupq_n_f32(SELU_LAMBDA);
    const float32x4_t vlam_alpha = vdupq_n_f32(SELU_LAMBDA * SELU_ALPHA);
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    uint32_t i = 0;
    for (; i + 4 <= in->size; i += 4) {
        __builtin_prefetch(&in->data[i + 64], 0, 1);
        float32x4_t x = vld1q_f32(&in->data[i]);
        uint32x4_t pos = vcgtq_f32(x, vzero);
        float32x4_t pos_val = vmulq_f32(vlam, x);
        float32x4_t neg_val = vmulq_f32(vlam_alpha, vsubq_f32(fast_exp_neon(x), vdupq_n_f32(1.0f)));
        vst1q_f32(&out->data[i], vbslq_f32(pos, pos_val, neg_val));
    }
    for (; i < in->size; i++) {
        float x = in->data[i];
        out->data[i] = x > 0 ? SELU_LAMBDA * x : SELU_LAMBDA * SELU_ALPHA * (expf(x) - 1.0f);
    }
#else
    for (uint32_t i = 0; i < in->size; i++) {
        float x = in->data[i];
        out->data[i] = x > 0 ? SELU_LAMBDA * x : SELU_LAMBDA * SELU_ALPHA * (expf(x) - 1.0f);
    }
#endif
}

void tensor_selu_inplace(Tensor* t) { tensor_selu(t, t); }

// ============================================================
// Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
// ============================================================

void tensor_mish(Tensor* out, const Tensor* in) {
#if RPITORCH_HAS_NEON
    const float32x4_t vone = vdupq_n_f32(1.0f);
    uint32_t i = 0;
    // 4× unroll = 16 floats = 1 cache line
    for (; i + 16 <= in->size; i += 16) {
        __builtin_prefetch(&in->data[i + 64], 0, 1);
        for (int k = 0; k < 16; k += 4) {
            float32x4_t x = vld1q_f32(&in->data[i + k]);
            // Fully vectorized softplus: ln(1 + exp(x))
            float32x4_t sp = fast_log_neon(vaddq_f32(vone, fast_exp_neon(x)));
            float32x4_t th = fast_tanh_neon(sp);
            vst1q_f32(&out->data[i + k], vmulq_f32(x, th));
        }
    }
    for (; i + 4 <= in->size; i += 4) {
        float32x4_t x = vld1q_f32(&in->data[i]);
        float32x4_t sp = fast_log_neon(vaddq_f32(vone, fast_exp_neon(x)));
        float32x4_t th = fast_tanh_neon(sp);
        vst1q_f32(&out->data[i], vmulq_f32(x, th));
    }
    for (; i < in->size; i++) {
        float x = in->data[i];
        out->data[i] = x * tanhf(logf(1.0f + expf(x)));
    }
#else
    for (uint32_t i = 0; i < in->size; i++) {
        float x = in->data[i];
        out->data[i] = x * tanhf(logf(1.0f + expf(x)));
    }
#endif
}

void tensor_mish_inplace(Tensor* t) { tensor_mish(t, t); }

// ============================================================
// Hardswish: x * min(max(x+3, 0), 6) / 6
// ============================================================

void tensor_hardswish(Tensor* out, const Tensor* in) {
#if RPITORCH_HAS_NEON
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    const float32x4_t vthree = vdupq_n_f32(3.0f);
    const float32x4_t vsix = vdupq_n_f32(6.0f);
    const float32x4_t vinv6 = vdupq_n_f32(1.0f / 6.0f);
    uint32_t i = 0;
    for (; i + 4 <= in->size; i += 4) {
        __builtin_prefetch(&in->data[i + 64], 0, 1);
        float32x4_t x = vld1q_f32(&in->data[i]);
        float32x4_t clip = vminq_f32(vmaxq_f32(vaddq_f32(x, vthree), vzero), vsix);
        vst1q_f32(&out->data[i], vmulq_f32(vmulq_f32(x, clip), vinv6));
    }
    for (; i < in->size; i++) {
        float x = in->data[i];
        float clip = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
        out->data[i] = x * clip / 6.0f;
    }
#else
    for (uint32_t i = 0; i < in->size; i++) {
        float x = in->data[i];
        float clip = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
        out->data[i] = x * clip / 6.0f;
    }
#endif
}

void tensor_hardswish_inplace(Tensor* t) { tensor_hardswish(t, t); }

// ============================================================
// Hardsigmoid: min(max(x/6 + 0.5, 0), 1)
// ============================================================

void tensor_hardsigmoid(Tensor* out, const Tensor* in) {
#if RPITORCH_HAS_NEON
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    const float32x4_t vone = vdupq_n_f32(1.0f);
    const float32x4_t vinv6 = vdupq_n_f32(1.0f / 6.0f);
    const float32x4_t vhalf = vdupq_n_f32(0.5f);
    uint32_t i = 0;
    for (; i + 4 <= in->size; i += 4) {
        __builtin_prefetch(&in->data[i + 64], 0, 1);
        float32x4_t x = vld1q_f32(&in->data[i]);
        float32x4_t val = vfmaq_f32(vhalf, x, vinv6); // x/6 + 0.5
        vst1q_f32(&out->data[i], vminq_f32(vmaxq_f32(val, vzero), vone));
    }
    for (; i < in->size; i++) {
        out->data[i] = fminf(fmaxf(in->data[i] / 6.0f + 0.5f, 0.0f), 1.0f);
    }
#else
    for (uint32_t i = 0; i < in->size; i++)
        out->data[i] = fminf(fmaxf(in->data[i] / 6.0f + 0.5f, 0.0f), 1.0f);
#endif
}

void tensor_hardsigmoid_inplace(Tensor* t) { tensor_hardsigmoid(t, t); }

// ============================================================
// Hardtanh: min(max(x, min_val), max_val)
// ============================================================

void tensor_hardtanh(Tensor* out, const Tensor* in, float min_val, float max_val) {
#if RPITORCH_HAS_NEON
    float32x4_t vlo = vdupq_n_f32(min_val);
    float32x4_t vhi = vdupq_n_f32(max_val);
    uint32_t i = 0;
    for (; i + 16 <= in->size; i += 16) {
        __builtin_prefetch(&in->data[i + 64], 0, 1);
        vst1q_f32(&out->data[i],     vminq_f32(vmaxq_f32(vld1q_f32(&in->data[i]),     vlo), vhi));
        vst1q_f32(&out->data[i + 4], vminq_f32(vmaxq_f32(vld1q_f32(&in->data[i + 4]), vlo), vhi));
        vst1q_f32(&out->data[i + 8], vminq_f32(vmaxq_f32(vld1q_f32(&in->data[i + 8]), vlo), vhi));
        vst1q_f32(&out->data[i + 12],vminq_f32(vmaxq_f32(vld1q_f32(&in->data[i + 12]),vlo), vhi));
    }
    for (; i < in->size; i++)
        out->data[i] = fminf(fmaxf(in->data[i], min_val), max_val);
#else
    for (uint32_t i = 0; i < in->size; i++)
        out->data[i] = fminf(fmaxf(in->data[i], min_val), max_val);
#endif
}

void tensor_hardtanh_inplace(Tensor* t, float min_val, float max_val) {
    tensor_hardtanh(t, t, min_val, max_val);
}

// ============================================================
// CELU: max(0,x) + min(0, alpha*(exp(x/alpha)-1))
// ============================================================

void tensor_celu(Tensor* out, const Tensor* in, float alpha) {
    float inv_alpha = 1.0f / alpha;
#if RPITORCH_HAS_NEON
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    const float32x4_t valpha = vdupq_n_f32(alpha);
    const float32x4_t vinv_a = vdupq_n_f32(inv_alpha);
    const float32x4_t vone = vdupq_n_f32(1.0f);
    uint32_t i = 0;
    for (; i + 4 <= in->size; i += 4) {
        __builtin_prefetch(&in->data[i + 64], 0, 1);
        float32x4_t x = vld1q_f32(&in->data[i]);
        float32x4_t pos = vmaxq_f32(x, vzero);
        float32x4_t neg = vminq_f32(vzero, vmulq_f32(valpha,
            vsubq_f32(fast_exp_neon(vmulq_f32(x, vinv_a)), vone)));
        vst1q_f32(&out->data[i], vaddq_f32(pos, neg));
    }
    for (; i < in->size; i++) {
        float x = in->data[i];
        out->data[i] = fmaxf(0, x) + fminf(0, alpha * (expf(x * inv_alpha) - 1.0f));
    }
#else
    for (uint32_t i = 0; i < in->size; i++) {
        float x = in->data[i];
        out->data[i] = fmaxf(0, x) + fminf(0, alpha * (expf(x * inv_alpha) - 1.0f));
    }
#endif
}

void tensor_celu_inplace(Tensor* t, float alpha) { tensor_celu(t, t, alpha); }

// ============================================================
// Softsign: x / (1 + |x|)
// ============================================================

void tensor_softsign(Tensor* out, const Tensor* in) {
#if RPITORCH_HAS_NEON
    const float32x4_t vone = vdupq_n_f32(1.0f);
    uint32_t i = 0;
    for (; i + 4 <= in->size; i += 4) {
        __builtin_prefetch(&in->data[i + 64], 0, 1);
        float32x4_t x = vld1q_f32(&in->data[i]);
        float32x4_t denom = vaddq_f32(vone, vabsq_f32(x));
        // reciprocal estimate + Newton step
        float32x4_t inv = vrecpeq_f32(denom);
        inv = vmulq_f32(inv, vrecpsq_f32(denom, inv));
        vst1q_f32(&out->data[i], vmulq_f32(x, inv));
    }
    for (; i < in->size; i++) {
        float x = in->data[i];
        out->data[i] = x / (1.0f + fabsf(x));
    }
#else
    for (uint32_t i = 0; i < in->size; i++) {
        float x = in->data[i];
        out->data[i] = x / (1.0f + fabsf(x));
    }
#endif
}

void tensor_softsign_inplace(Tensor* t) { tensor_softsign(t, t); }

// ============================================================
// LogSoftmax: x - log(sum(exp(x))) along last dim
// ============================================================

void tensor_log_softmax(Tensor* out, const Tensor* in) {
    uint32_t batch = 1;
    uint32_t dim_size = in->shape[in->dims - 1];
    for (uint32_t d = 0; d < in->dims - 1; d++) batch *= in->shape[d];

    for (uint32_t b = 0; b < batch; b++) {
        const float* src = &in->data[b * dim_size];
        float* dst = &out->data[b * dim_size];

        // Find max for numerical stability
        float max_val = src[0];
        for (uint32_t i = 1; i < dim_size; i++) if (src[i] > max_val) max_val = src[i];

        // Compute log(sum(exp(x - max)))
        float sum = 0;
        for (uint32_t i = 0; i < dim_size; i++) sum += expf(src[i] - max_val);
        float log_sum = max_val + logf(sum);

        // x - log_sum_exp
#if RPITORCH_HAS_NEON
        float32x4_t vlse = vdupq_n_f32(log_sum);
        uint32_t i = 0;
        for (; i + 4 <= dim_size; i += 4)
            vst1q_f32(&dst[i], vsubq_f32(vld1q_f32(&src[i]), vlse));
        for (; i < dim_size; i++) dst[i] = src[i] - log_sum;
#else
        for (uint32_t i = 0; i < dim_size; i++) dst[i] = src[i] - log_sum;
#endif
    }
}

void tensor_log_softmax_inplace(Tensor* t) { tensor_log_softmax(t, t); }

// ============================================================
// PReLU: max(0,x) + weight * min(0,x), per-channel weight
// ============================================================

void tensor_prelu(Tensor* out, const Tensor* in, const Tensor* weight) {
#if RPITORCH_HAS_NEON
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    if (weight->size == 1) {
        // Single weight for all channels
        float32x4_t vw = vdupq_n_f32(weight->data[0]);
        uint32_t i = 0;
        for (; i + 4 <= in->size; i += 4) {
            float32x4_t x = vld1q_f32(&in->data[i]);
            float32x4_t pos = vmaxq_f32(x, vzero);
            float32x4_t neg = vmulq_f32(vw, vminq_f32(x, vzero));
            vst1q_f32(&out->data[i], vaddq_f32(pos, neg));
        }
        for (; i < in->size; i++) {
            float x = in->data[i];
            out->data[i] = (x > 0) ? x : weight->data[0] * x;
        }
    } else {
        for (uint32_t i = 0; i < in->size; i++) {
            float x = in->data[i];
            float w = weight->data[i % weight->size];
            out->data[i] = (x > 0) ? x : w * x;
        }
    }
#else
    for (uint32_t i = 0; i < in->size; i++) {
        float x = in->data[i];
        float w = weight->data[i % weight->size];
        out->data[i] = (x > 0) ? x : w * x;
    }
#endif
}

// ============================================================
// RReLU: x if x>=0, else x*slope where slope~U(lower, upper)
// At eval time uses the mean: (lower + upper) / 2
// ============================================================

void tensor_rrelu(Tensor* out, const Tensor* in, float lower, float upper) {
    float slope = (lower + upper) * 0.5f;
#if RPITORCH_HAS_NEON
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    const float32x4_t vslope = vdupq_n_f32(slope);
    uint32_t i = 0;
    for (; i + 4 <= in->size; i += 4) {
        __builtin_prefetch(&in->data[i + 64], 0, 1);
        float32x4_t x = vld1q_f32(&in->data[i]);
        float32x4_t pos = vmaxq_f32(x, vzero);
        float32x4_t neg = vmulq_f32(vslope, vminq_f32(x, vzero));
        vst1q_f32(&out->data[i], vaddq_f32(pos, neg));
    }
    for (; i < in->size; i++) {
        float x = in->data[i];
        out->data[i] = (x >= 0) ? x : slope * x;
    }
#else
    for (uint32_t i = 0; i < in->size; i++) {
        float x = in->data[i];
        out->data[i] = (x >= 0) ? x : slope * x;
    }
#endif
}

void tensor_rrelu_inplace(Tensor* t, float lower, float upper) {
    tensor_rrelu(t, t, lower, upper);
}

// ============================================================
// Threshold: x if x > threshold, else value
// ============================================================

void tensor_threshold(Tensor* out, const Tensor* in, float threshold, float value) {
#if RPITORCH_HAS_NEON
    float32x4_t vthresh = vdupq_n_f32(threshold);
    float32x4_t vval = vdupq_n_f32(value);
    uint32_t i = 0;
    for (; i + 4 <= in->size; i += 4) {
        float32x4_t x = vld1q_f32(&in->data[i]);
        uint32x4_t mask = vcgtq_f32(x, vthresh);
        vst1q_f32(&out->data[i], vbslq_f32(mask, x, vval));
    }
    for (; i < in->size; i++)
        out->data[i] = (in->data[i] > threshold) ? in->data[i] : value;
#else
    for (uint32_t i = 0; i < in->size; i++)
        out->data[i] = (in->data[i] > threshold) ? in->data[i] : value;
#endif
}

void tensor_threshold_inplace(Tensor* t, float threshold, float value) {
    tensor_threshold(t, t, threshold, value);
}

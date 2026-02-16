/*
 * RPL Math Operations — trig, exp/log, rounding, power, special functions
 */
#include "rpl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>

static inline Tensor* _like(const Tensor* t) {
    return tensor_create(t->dims, t->shape, false);
}

#define DEFINE_UNARY_OP(name, expr) \
Tensor* tensor_##name(const Tensor* t) { \
    Tensor* out = _like(t); \
    _Pragma("omp parallel for if(t->size >= RPL_OMP_THRESHOLD)") \
    for (uint32_t i = 0; i < t->size; i++) { \
        float x = t->data[i]; \
        out->data[i] = (expr); \
    } \
    return out; \
} \
void tensor_##name##_inplace(Tensor* t) { \
    _Pragma("omp parallel for if(t->size >= RPL_OMP_THRESHOLD)") \
    for (uint32_t i = 0; i < t->size; i++) { \
        float x = t->data[i]; \
        t->data[i] = (expr); \
    } \
}

#define DEFINE_BINARY_OP(name, expr) \
Tensor* tensor_##name(const Tensor* a, const Tensor* b) { \
    Tensor* out = _like(a); \
    _Pragma("omp parallel for if(a->size >= RPL_OMP_THRESHOLD)") \
    for (uint32_t i = 0; i < a->size; i++) { \
        float x = a->data[i]; \
        float y = b->data[i % b->size]; \
        out->data[i] = (expr); \
    } \
    return out; \
}

// --- Forward declare helpers ---
static float erfinvf_approx(float x);
static float digamma_impl(float x);
static float i0f_approx(float x);

// Trig
DEFINE_UNARY_OP(sin, sinf(x))
DEFINE_UNARY_OP(cos, cosf(x))
DEFINE_UNARY_OP(tan, tanf(x))
DEFINE_UNARY_OP(asin, asinf(x))
DEFINE_UNARY_OP(acos, acosf(x))
DEFINE_UNARY_OP(atan, atanf(x))
DEFINE_BINARY_OP(atan2, atan2f(x, y))
DEFINE_BINARY_OP(hypot, hypotf(x, y))

// Hyperbolic
DEFINE_UNARY_OP(sinh, sinhf(x))
DEFINE_UNARY_OP(cosh, coshf(x))
DEFINE_UNARY_OP(asinh, asinhf(x))
DEFINE_UNARY_OP(acosh, acoshf(x))
DEFINE_UNARY_OP(atanh, atanhf(x))

// Exp/Log
DEFINE_UNARY_OP(exp, expf(x))
DEFINE_UNARY_OP(expm1, expm1f(x))
DEFINE_UNARY_OP(exp2, exp2f(x))
DEFINE_UNARY_OP(log, logf(x))
DEFINE_UNARY_OP(log2, log2f(x))
DEFINE_UNARY_OP(log10, log10f(x))
DEFINE_UNARY_OP(log1p, log1pf(x))
DEFINE_BINARY_OP(logaddexp, logf(expf(x) + expf(y)))
DEFINE_BINARY_OP(logaddexp2, log2f(exp2f(x) + exp2f(y)))

// Rounding
DEFINE_UNARY_OP(round_op, roundf(x))
DEFINE_UNARY_OP(floor_op, floorf(x))
DEFINE_UNARY_OP(ceil_op, ceilf(x))
DEFINE_UNARY_OP(trunc_op, truncf(x))
DEFINE_UNARY_OP(frac, x - truncf(x))

// Power & Root
DEFINE_BINARY_OP(pow_op, powf(x, y))
DEFINE_UNARY_OP(sqrt_op, sqrtf(x))
DEFINE_UNARY_OP(cbrt, cbrtf(x))

// rsqrt — NEON estimate + 1 Newton step for full precision
Tensor* tensor_rsqrt(const Tensor* t) {
    Tensor* out = _like(t);
#if RPITORCH_HAS_NEON
    uint32_t i = 0;
    for (; i + 16 <= t->size; i += 16) {
        __builtin_prefetch(&t->data[i + 64], 0, 1);
        for (int k = 0; k < 16; k += 4) {
            float32x4_t v = vld1q_f32(&t->data[i + k]);
            float32x4_t est = vrsqrteq_f32(v);
            est = vmulq_f32(est, vrsqrtsq_f32(vmulq_f32(v, est), est)); // Newton
            vst1q_f32(&out->data[i + k], est);
        }
    }
    for (; i < t->size; i++) out->data[i] = 1.0f / sqrtf(t->data[i]);
#else
    for (uint32_t i = 0; i < t->size; i++) out->data[i] = 1.0f / sqrtf(t->data[i]);
#endif
    return out;
}
void tensor_rsqrt_inplace(Tensor* t) {
#if RPITORCH_HAS_NEON
    uint32_t i = 0;
    for (; i + 4 <= t->size; i += 4) {
        float32x4_t v = vld1q_f32(&t->data[i]);
        float32x4_t est = vrsqrteq_f32(v);
        est = vmulq_f32(est, vrsqrtsq_f32(vmulq_f32(v, est), est));
        vst1q_f32(&t->data[i], est);
    }
    for (; i < t->size; i++) t->data[i] = 1.0f / sqrtf(t->data[i]);
#else
    for (uint32_t i = 0; i < t->size; i++) t->data[i] = 1.0f / sqrtf(t->data[i]);
#endif
}

// square — NEON vmulq_f32(v, v)
Tensor* tensor_square(const Tensor* t) {
    Tensor* out = _like(t);
#if RPITORCH_HAS_NEON
    uint32_t i = 0;
    for (; i + 16 <= t->size; i += 16) {
        __builtin_prefetch(&t->data[i + 64], 0, 1);
        for (int k = 0; k < 16; k += 4) {
            float32x4_t v = vld1q_f32(&t->data[i + k]);
            vst1q_f32(&out->data[i + k], vmulq_f32(v, v));
        }
    }
    for (; i < t->size; i++) out->data[i] = t->data[i] * t->data[i];
#else
    for (uint32_t i = 0; i < t->size; i++) out->data[i] = t->data[i] * t->data[i];
#endif
    return out;
}
void tensor_square_inplace(Tensor* t) {
#if RPITORCH_HAS_NEON
    uint32_t i = 0;
    for (; i + 4 <= t->size; i += 4) {
        float32x4_t v = vld1q_f32(&t->data[i]);
        vst1q_f32(&t->data[i], vmulq_f32(v, v));
    }
    for (; i < t->size; i++) t->data[i] *= t->data[i];
#else
    for (uint32_t i = 0; i < t->size; i++) t->data[i] *= t->data[i];
#endif
}

// reciprocal — NEON estimate + 1 Newton step
Tensor* tensor_reciprocal(const Tensor* t) {
    Tensor* out = _like(t);
#if RPITORCH_HAS_NEON
    uint32_t i = 0;
    for (; i + 16 <= t->size; i += 16) {
        __builtin_prefetch(&t->data[i + 64], 0, 1);
        for (int k = 0; k < 16; k += 4) {
            float32x4_t v = vld1q_f32(&t->data[i + k]);
            float32x4_t est = vrecpeq_f32(v);
            est = vmulq_f32(est, vrecpsq_f32(v, est)); // Newton step
            vst1q_f32(&out->data[i + k], est);
        }
    }
    for (; i < t->size; i++) out->data[i] = 1.0f / t->data[i];
#else
    for (uint32_t i = 0; i < t->size; i++) out->data[i] = 1.0f / t->data[i];
#endif
    return out;
}
void tensor_reciprocal_inplace(Tensor* t) {
#if RPITORCH_HAS_NEON
    uint32_t i = 0;
    for (; i + 4 <= t->size; i += 4) {
        float32x4_t v = vld1q_f32(&t->data[i]);
        float32x4_t est = vrecpeq_f32(v);
        est = vmulq_f32(est, vrecpsq_f32(v, est));
        vst1q_f32(&t->data[i], est);
    }
    for (; i < t->size; i++) t->data[i] = 1.0f / t->data[i];
#else
    for (uint32_t i = 0; i < t->size; i++) t->data[i] = 1.0f / t->data[i];
#endif
}

// abs — NEON vabsq_f32
Tensor* tensor_abs_op(const Tensor* t) {
    Tensor* out = _like(t);
#if RPITORCH_HAS_NEON
    uint32_t i = 0;
    for (; i + 16 <= t->size; i += 16) {
        __builtin_prefetch(&t->data[i + 64], 0, 1);
        vst1q_f32(&out->data[i],     vabsq_f32(vld1q_f32(&t->data[i])));
        vst1q_f32(&out->data[i + 4], vabsq_f32(vld1q_f32(&t->data[i + 4])));
        vst1q_f32(&out->data[i + 8], vabsq_f32(vld1q_f32(&t->data[i + 8])));
        vst1q_f32(&out->data[i + 12],vabsq_f32(vld1q_f32(&t->data[i + 12])));
    }
    for (; i < t->size; i++) out->data[i] = fabsf(t->data[i]);
#else
    for (uint32_t i = 0; i < t->size; i++) out->data[i] = fabsf(t->data[i]);
#endif
    return out;
}
void tensor_abs_op_inplace(Tensor* t) {
#if RPITORCH_HAS_NEON
    uint32_t i = 0;
    for (; i + 4 <= t->size; i += 4)
        vst1q_f32(&t->data[i], vabsq_f32(vld1q_f32(&t->data[i])));
    for (; i < t->size; i++) t->data[i] = fabsf(t->data[i]);
#else
    for (uint32_t i = 0; i < t->size; i++) t->data[i] = fabsf(t->data[i]);
#endif
}

// neg — NEON vnegq_f32
Tensor* tensor_neg(const Tensor* t) {
    Tensor* out = _like(t);
#if RPITORCH_HAS_NEON
    uint32_t i = 0;
    for (; i + 16 <= t->size; i += 16) {
        __builtin_prefetch(&t->data[i + 64], 0, 1);
        vst1q_f32(&out->data[i],     vnegq_f32(vld1q_f32(&t->data[i])));
        vst1q_f32(&out->data[i + 4], vnegq_f32(vld1q_f32(&t->data[i + 4])));
        vst1q_f32(&out->data[i + 8], vnegq_f32(vld1q_f32(&t->data[i + 8])));
        vst1q_f32(&out->data[i + 12],vnegq_f32(vld1q_f32(&t->data[i + 12])));
    }
    for (; i < t->size; i++) out->data[i] = -t->data[i];
#else
    for (uint32_t i = 0; i < t->size; i++) out->data[i] = -t->data[i];
#endif
    return out;
}
void tensor_neg_inplace(Tensor* t) {
#if RPITORCH_HAS_NEON
    uint32_t i = 0;
    for (; i + 4 <= t->size; i += 4)
        vst1q_f32(&t->data[i], vnegq_f32(vld1q_f32(&t->data[i])));
    for (; i < t->size; i++) t->data[i] = -t->data[i];
#else
    for (uint32_t i = 0; i < t->size; i++) t->data[i] = -t->data[i];
#endif
}

DEFINE_UNARY_OP(sign, (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f))
DEFINE_UNARY_OP(signbit_op, (x < 0.0f) ? 1.0f : 0.0f)
DEFINE_BINARY_OP(copysign_op, copysignf(x, y))
DEFINE_BINARY_OP(heaviside, (x < 0.0f) ? 0.0f : ((x > 0.0f) ? 1.0f : y))

// Angular
DEFINE_UNARY_OP(deg2rad, x * (float)(M_PI / 180.0))
DEFINE_UNARY_OP(rad2deg, x * (float)(180.0 / M_PI))

// Special
DEFINE_UNARY_OP(erf, erff(x))
DEFINE_UNARY_OP(erfc, erfcf(x))
DEFINE_UNARY_OP(erfinv, erfinvf_approx(x))
DEFINE_UNARY_OP(lgamma_op, lgammaf(x))
DEFINE_UNARY_OP(digamma, digamma_impl(x))
DEFINE_UNARY_OP(sinc, (fabsf(x) < 1e-7f) ? 1.0f : sinf((float)M_PI * x) / ((float)M_PI * x))
DEFINE_UNARY_OP(i0, i0f_approx(x))
DEFINE_UNARY_OP(logit, logf(x / (1.0f - x)))

// Binary math
DEFINE_BINARY_OP(fmod_op, fmodf(x, y))
DEFINE_BINARY_OP(remainder_op, remainderf(x, y))
DEFINE_BINARY_OP(floor_divide, floorf(x / y))
DEFINE_BINARY_OP(true_divide, x / y)

// --- Helper implementations ---

static float erfinvf_approx(float x) {
    if (x <= -1.0f) return -INFINITY;
    if (x >= 1.0f) return INFINITY;
    float a = 0.147f;
    float ln1mx2 = logf(1.0f - x * x);
    float t = 2.0f / ((float)M_PI * a) + ln1mx2 * 0.5f;
    float s = (x < 0) ? -1.0f : 1.0f;
    return s * sqrtf(sqrtf(t * t - ln1mx2 / a) - t);
}

static float digamma_impl(float x) {
    float result = 0.0f;
    while (x < 6.0f) { result -= 1.0f / x; x += 1.0f; }
    result += logf(x) - 0.5f / x - 1.0f / (12.0f * x * x);
    return result;
}

static float i0f_approx(float x) {
    float ax = fabsf(x);
    if (ax < 3.75f) {
        float t = (x / 3.75f); t *= t;
        return 1.0f + t*(3.5156229f + t*(3.0899424f + t*(1.2067492f +
               t*(0.2659732f + t*(0.0360768f + t*0.0045813f)))));
    }
    float t = 3.75f / ax;
    return (expf(ax) / sqrtf(ax)) * (0.39894228f + t*(0.01328592f +
           t*(0.00225319f + t*(-0.00157565f + t*(0.00916281f +
           t*(-0.02057706f + t*(0.02635537f + t*(-0.01647633f +
           t*0.00392377f))))))));
}

// Clamp (NEON)
Tensor* tensor_clamp(const Tensor* t, float lo, float hi) {
    Tensor* out = _like(t);
#if RPITORCH_HAS_NEON
    float32x4_t vlo = vdupq_n_f32(lo), vhi = vdupq_n_f32(hi);
    uint32_t i = 0;
    for (; i + 4 <= t->size; i += 4)
        vst1q_f32(&out->data[i], vminq_f32(vmaxq_f32(vld1q_f32(&t->data[i]), vlo), vhi));
    for (; i < t->size; i++) out->data[i] = fmaxf(lo, fminf(hi, t->data[i]));
#else
    for (uint32_t i = 0; i < t->size; i++) out->data[i] = fmaxf(lo, fminf(hi, t->data[i]));
#endif
    return out;
}
void tensor_clamp_inplace(Tensor* t, float lo, float hi) {
#if RPITORCH_HAS_NEON
    float32x4_t vlo = vdupq_n_f32(lo), vhi = vdupq_n_f32(hi);
    uint32_t i = 0;
    for (; i + 4 <= t->size; i += 4)
        vst1q_f32(&t->data[i], vminq_f32(vmaxq_f32(vld1q_f32(&t->data[i]), vlo), vhi));
    for (; i < t->size; i++) t->data[i] = fmaxf(lo, fminf(hi, t->data[i]));
#else
    for (uint32_t i = 0; i < t->size; i++) t->data[i] = fmaxf(lo, fminf(hi, t->data[i]));
#endif
}

// NaN handling
Tensor* tensor_nan_to_num(const Tensor* t, float nan_v, float posinf_v, float neginf_v) {
    Tensor* out = _like(t);
    #pragma omp parallel for
    for (uint32_t i = 0; i < t->size; i++) {
        float x = t->data[i];
        out->data[i] = isnan(x) ? nan_v : (isinf(x) ? (x > 0 ? posinf_v : neginf_v) : x);
    }
    return out;
}

// Lerp (NEON)
Tensor* tensor_lerp(const Tensor* a, const Tensor* b, float w) {
    Tensor* out = _like(a);
#if RPITORCH_HAS_NEON
    float32x4_t vw = vdupq_n_f32(w);
    uint32_t i = 0;
    for (; i + 4 <= a->size; i += 4) {
        float32x4_t va = vld1q_f32(&a->data[i]);
        vst1q_f32(&out->data[i], vfmaq_f32(va, vsubq_f32(vld1q_f32(&b->data[i]), va), vw));
    }
    for (; i < a->size; i++) out->data[i] = a->data[i] + w * (b->data[i] - a->data[i]);
#else
    for (uint32_t i = 0; i < a->size; i++) out->data[i] = a->data[i] + w * (b->data[i] - a->data[i]);
#endif
    return out;
}

// Sub / Div (NEON)
Tensor* tensor_sub(const Tensor* a, const Tensor* b) {
    Tensor* out = _like(a);
#ifdef USE_GPU
    if (a->device == DEVICE_GPU || b->device == DEVICE_GPU) {
        if (a->size == b->size) {
            tensor_sub_gpu(out, a, b);
            return;
        } else {
            tensor_from_gpu((Tensor*)a);
            tensor_from_gpu((Tensor*)b);
        }
    }
#endif
#if RPITORCH_HAS_NEON
    uint32_t i = 0;
    if (a->size == b->size) {
        for (; i + 16 <= a->size; i += 16) {
            vst1q_f32(&out->data[i],    vsubq_f32(vld1q_f32(&a->data[i]),    vld1q_f32(&b->data[i])));
            vst1q_f32(&out->data[i+4],  vsubq_f32(vld1q_f32(&a->data[i+4]),  vld1q_f32(&b->data[i+4])));
            vst1q_f32(&out->data[i+8],  vsubq_f32(vld1q_f32(&a->data[i+8]),  vld1q_f32(&b->data[i+8])));
            vst1q_f32(&out->data[i+12], vsubq_f32(vld1q_f32(&a->data[i+12]), vld1q_f32(&b->data[i+12])));
        }
        for (; i + 4 <= a->size; i += 4)
            vst1q_f32(&out->data[i], vsubq_f32(vld1q_f32(&a->data[i]), vld1q_f32(&b->data[i])));
    }
    for (; i < a->size; i++) out->data[i] = a->data[i] - b->data[i % b->size];
#else
    for (uint32_t i = 0; i < a->size; i++) out->data[i] = a->data[i] - b->data[i % b->size];
#endif
    return out;
}

Tensor* tensor_div(const Tensor* a, const Tensor* b) {
    Tensor* out = _like(a);
#ifdef USE_GPU
    if (a->device == DEVICE_GPU || b->device == DEVICE_GPU) {
        if (a->size == b->size) {
            tensor_div_gpu(out, a, b);
            return;
        } else {
            tensor_from_gpu((Tensor*)a);
            tensor_from_gpu((Tensor*)b);
        }
    }
#endif
    #pragma omp parallel for
    for (uint32_t i = 0; i < a->size; i++) out->data[i] = a->data[i] / b->data[i % b->size];
    return out;
}

// xlogy, addcdiv, addcmul
Tensor* tensor_xlogy(const Tensor* a, const Tensor* b) {
    Tensor* out = _like(a);
    #pragma omp parallel for
    for (uint32_t i = 0; i < a->size; i++) {
        float x = a->data[i];
        out->data[i] = (x == 0.0f) ? 0.0f : x * logf(b->data[i % b->size]);
    }
    return out;
}

Tensor* tensor_addcdiv(const Tensor* input, const Tensor* t1, const Tensor* t2, float value) {
    Tensor* out = _like(input);
    #pragma omp parallel for
    for (uint32_t i = 0; i < input->size; i++)
        out->data[i] = input->data[i] + value * (t1->data[i] / t2->data[i]);
    return out;
}

// addcmul — NEON vfmaq_f32 for fused multiply-add
Tensor* tensor_addcmul(const Tensor* input, const Tensor* t1, const Tensor* t2, float value) {
    Tensor* out = _like(input);
#if RPITORCH_HAS_NEON
    float32x4_t vval = vdupq_n_f32(value);
    uint32_t i = 0;
    for (; i + 16 <= input->size; i += 16) {
        __builtin_prefetch(&input->data[i + 64], 0, 1);
        __builtin_prefetch(&t1->data[i + 64], 0, 1);
        __builtin_prefetch(&t2->data[i + 64], 0, 1);
        for (int k = 0; k < 16; k += 4) {
            float32x4_t prod = vmulq_f32(vld1q_f32(&t1->data[i+k]), vld1q_f32(&t2->data[i+k]));
            vst1q_f32(&out->data[i+k], vfmaq_f32(vld1q_f32(&input->data[i+k]), prod, vval));
        }
    }
    for (; i < input->size; i++)
        out->data[i] = input->data[i] + value * (t1->data[i] * t2->data[i]);
#else
    #pragma omp parallel for
    for (uint32_t i = 0; i < input->size; i++)
        out->data[i] = input->data[i] + value * (t1->data[i] * t2->data[i]);
#endif
    return out;
}

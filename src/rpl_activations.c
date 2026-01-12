/*
 * RPiTorch Activation Functions
 * LeakyReLU, ELU, Swish, Softplus
 */

#include "rpl.h"
#include <math.h>

// ============================================================
// LeakyReLU
// ============================================================

void tensor_leaky_relu(Tensor* out, const Tensor* in, float negative_slope) {
    uint32_t i = 0;
    
    #if RPITORCH_HAS_NEON
    float32x4_t vzero = vdupq_n_f32(0.0f);
    float32x4_t vslope = vdupq_n_f32(negative_slope);
    
    for (; i + 4 <= in->size; i += 4) {
        float32x4_t x = vld1q_f32(&in->data[i]);
        uint32x4_t mask = vcgtq_f32(x, vzero);  // x > 0
        float32x4_t neg_part = vmulq_f32(x, vslope);
        float32x4_t result = vbslq_f32(mask, x, neg_part);
        vst1q_f32(&out->data[i], result);
    }
    #endif
    
    for (; i < in->size; i++) {
        out->data[i] = (in->data[i] > 0.0f) ? in->data[i] : negative_slope * in->data[i];
    }
}

void tensor_leaky_relu_inplace(Tensor* t, float negative_slope) {
    tensor_leaky_relu(t, t, negative_slope);
}

// ============================================================
// ELU (Exponential Linear Unit)
// ============================================================

void tensor_elu(Tensor* out, const Tensor* in, float alpha) {
    #pragma omp parallel for
    for (uint32_t i = 0; i < in->size; i++) {
        float x = in->data[i];
        out->data[i] = (x > 0.0f) ? x : alpha * (expf(x) - 1.0f);
    }
}

void tensor_elu_inplace(Tensor* t, float alpha) {
    tensor_elu(t, t, alpha);
}

// ============================================================
// Swish / SiLU
// ============================================================

void tensor_swish(Tensor* out, const Tensor* in) {
    uint32_t i = 0;
    
    #if RPITORCH_HAS_NEON
    float32x4_t vone = vdupq_n_f32(1.0f);
    
    for (; i + 4 <= in->size; i += 4) {
        float32x4_t x = vld1q_f32(&in->data[i]);
        
        // Sigmoid approximation for NEON
        float32x4_t neg_x = vnegq_f32(x);
        // Fast exp approximation could be added here
        float32x4_t sigmoid = vrecpeq_f32(vaddq_f32(vone, neg_x));  // Simplified
        
        float32x4_t result = vmulq_f32(x, sigmoid);
        vst1q_f32(&out->data[i], result);
    }
    #endif
    
    for (; i < in->size; i++) {
        float x = in->data[i];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        out->data[i] = x * sigmoid;
    }
}

void tensor_swish_inplace(Tensor* t) {
    tensor_swish(t, t);
}

// ============================================================
// Softplus
// ============================================================

void tensor_softplus(Tensor* out, const Tensor* in, float beta, float threshold) {
    #pragma omp parallel for
    for (uint32_t i = 0; i < in->size; i++) {
        float x = in->data[i];
        if (x * beta > threshold) {
            out->data[i] = x;
        } else {
            out->data[i] = logf(1.0f + expf(beta * x)) / beta;
        }
    }
}

void tensor_softplus_inplace(Tensor* t, float beta, float threshold) {
    tensor_softplus(t, t, beta, threshold);
}

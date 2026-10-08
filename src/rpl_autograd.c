#include "rpl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#if RPITORCH_HAS_NEON
#include <arm_neon.h>
#endif

void parallel_gemm_optimized_trans(const float* A, const float* B, float* C, uint32_t M, uint32_t N, uint32_t K, bool trans_a, bool trans_b);

// Thread-local gradient tracking flag
static __thread bool rpl_grad_enabled = true;

void rpl_set_grad_enabled(bool enabled) {
    rpl_grad_enabled = enabled;
}

bool rpl_is_grad_enabled(void) {
    return rpl_grad_enabled;
}

// Reference Counting
void tensor_retain(Tensor* t) {
    if (!t) return;
    t->_refcount++;
}

void tensor_release(Tensor* t) {
    if (!t) return;
    t->_refcount--;
    if (t->_refcount <= 0) {
        tensor_free(t);
    }
}

// Structure for stack frames in iterative DFS
typedef struct {
    Tensor* node;
    int parent_idx;
} StackFrame;

// Backward Implementations

void backward_add(Tensor* t) {
    Tensor* a = t->_parents[0];
    Tensor* b = t->_parents[1];
    if (a && a->requires_grad) {
#if RPITORCH_HAS_NEON
        #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
        for (uint32_t base = 0; base < t->size; base += 512) {
            uint32_t end = (base + 512 < t->size) ? base + 512 : t->size;
            uint32_t i = base;
            for (; i + 16 <= end; i += 16) {
                float32x4_t ag0 = vld1q_f32(&a->grad[i]);
                float32x4_t ag1 = vld1q_f32(&a->grad[i + 4]);
                float32x4_t ag2 = vld1q_f32(&a->grad[i + 8]);
                float32x4_t ag3 = vld1q_f32(&a->grad[i + 12]);
                vst1q_f32(&a->grad[i],      vaddq_f32(ag0, vld1q_f32(&t->grad[i])));
                vst1q_f32(&a->grad[i + 4],  vaddq_f32(ag1, vld1q_f32(&t->grad[i + 4])));
                vst1q_f32(&a->grad[i + 8],  vaddq_f32(ag2, vld1q_f32(&t->grad[i + 8])));
                vst1q_f32(&a->grad[i + 12], vaddq_f32(ag3, vld1q_f32(&t->grad[i + 12])));
            }
            for (; i + 4 <= end; i += 4) {
                vst1q_f32(&a->grad[i], vaddq_f32(vld1q_f32(&a->grad[i]), vld1q_f32(&t->grad[i])));
            }
            for (; i < end; i++) a->grad[i] += t->grad[i];
        }
#else
        #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
        for (uint32_t i = 0; i < t->size; i++) a->grad[i] += t->grad[i];
#endif
    }
    if (b && b->requires_grad) {
        if (b->size == t->size) {
#if RPITORCH_HAS_NEON
            #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
            for (uint32_t base = 0; base < t->size; base += 512) {
                uint32_t end = (base + 512 < t->size) ? base + 512 : t->size;
                uint32_t i = base;
                for (; i + 16 <= end; i += 16) {
                    float32x4_t bg0 = vld1q_f32(&b->grad[i]);
                    float32x4_t bg1 = vld1q_f32(&b->grad[i + 4]);
                    float32x4_t bg2 = vld1q_f32(&b->grad[i + 8]);
                    float32x4_t bg3 = vld1q_f32(&b->grad[i + 12]);
                    vst1q_f32(&b->grad[i],      vaddq_f32(bg0, vld1q_f32(&t->grad[i])));
                    vst1q_f32(&b->grad[i + 4],  vaddq_f32(bg1, vld1q_f32(&t->grad[i + 4])));
                    vst1q_f32(&b->grad[i + 8],  vaddq_f32(bg2, vld1q_f32(&t->grad[i + 8])));
                    vst1q_f32(&b->grad[i + 12], vaddq_f32(bg3, vld1q_f32(&t->grad[i + 12])));
                }
                for (; i + 4 <= end; i += 4) {
                    vst1q_f32(&b->grad[i], vaddq_f32(vld1q_f32(&b->grad[i]), vld1q_f32(&t->grad[i])));
                }
                for (; i < end; i++) b->grad[i] += t->grad[i];
            }
#else
            #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
            for (uint32_t i = 0; i < t->size; i++) b->grad[i] += t->grad[i];
#endif
        } else {
            #pragma omp parallel for if(b->size >= RPL_OMP_THRESHOLD)
            for (uint32_t i = 0; i < b->size; i++) {
                float g = 0;
                for (uint32_t j = i; j < t->size; j += b->size) {
                    g += t->grad[j];
                }
                b->grad[i] += g;
            }
        }
    }
}

void backward_sub(Tensor* t) {
    Tensor* a = t->_parents[0];
    Tensor* b = t->_parents[1];
    if (a && a->requires_grad) {
#if RPITORCH_HAS_NEON
        #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
        for (uint32_t base = 0; base < t->size; base += 512) {
            uint32_t end = (base + 512 < t->size) ? base + 512 : t->size;
            uint32_t i = base;
            for (; i + 16 <= end; i += 16) {
                float32x4_t ag0 = vld1q_f32(&a->grad[i]);
                float32x4_t ag1 = vld1q_f32(&a->grad[i + 4]);
                float32x4_t ag2 = vld1q_f32(&a->grad[i + 8]);
                float32x4_t ag3 = vld1q_f32(&a->grad[i + 12]);
                vst1q_f32(&a->grad[i],      vaddq_f32(ag0, vld1q_f32(&t->grad[i])));
                vst1q_f32(&a->grad[i + 4],  vaddq_f32(ag1, vld1q_f32(&t->grad[i + 4])));
                vst1q_f32(&a->grad[i + 8],  vaddq_f32(ag2, vld1q_f32(&t->grad[i + 8])));
                vst1q_f32(&a->grad[i + 12], vaddq_f32(ag3, vld1q_f32(&t->grad[i + 12])));
            }
            for (; i + 4 <= end; i += 4) {
                vst1q_f32(&a->grad[i], vaddq_f32(vld1q_f32(&a->grad[i]), vld1q_f32(&t->grad[i])));
            }
            for (; i < end; i++) a->grad[i] += t->grad[i];
        }
#else
        #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
        for (uint32_t i = 0; i < t->size; i++) a->grad[i] += t->grad[i];
#endif
    }
    if (b && b->requires_grad) {
        if (b->size == t->size) {
#if RPITORCH_HAS_NEON
            #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
            for (uint32_t base = 0; base < t->size; base += 512) {
                uint32_t end = (base + 512 < t->size) ? base + 512 : t->size;
                uint32_t i = base;
                for (; i + 16 <= end; i += 16) {
                    float32x4_t bg0 = vld1q_f32(&b->grad[i]);
                    float32x4_t bg1 = vld1q_f32(&b->grad[i + 4]);
                    float32x4_t bg2 = vld1q_f32(&b->grad[i + 8]);
                    float32x4_t bg3 = vld1q_f32(&b->grad[i + 12]);
                    vst1q_f32(&b->grad[i],      vsubq_f32(bg0, vld1q_f32(&t->grad[i])));
                    vst1q_f32(&b->grad[i + 4],  vsubq_f32(bg1, vld1q_f32(&t->grad[i + 4])));
                    vst1q_f32(&b->grad[i + 8],  vsubq_f32(bg2, vld1q_f32(&t->grad[i + 8])));
                    vst1q_f32(&b->grad[i + 12], vsubq_f32(bg3, vld1q_f32(&t->grad[i + 12])));
                }
                for (; i + 4 <= end; i += 4) {
                    vst1q_f32(&b->grad[i], vsubq_f32(vld1q_f32(&b->grad[i]), vld1q_f32(&t->grad[i])));
                }
                for (; i < end; i++) b->grad[i] -= t->grad[i];
            }
#else
            #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
            for (uint32_t i = 0; i < t->size; i++) b->grad[i] -= t->grad[i];
#endif
        } else {
            #pragma omp parallel for if(b->size >= RPL_OMP_THRESHOLD)
            for (uint32_t i = 0; i < b->size; i++) {
                float g = 0;
                for (uint32_t j = i; j < t->size; j += b->size) {
                    g += t->grad[j];
                }
                b->grad[i] -= g;
            }
        }
    }
}

void backward_mul(Tensor* t) {
    Tensor* a = t->_parents[0];
    Tensor* b = t->_parents[1];
    if (a && a->requires_grad) {
        if (a->size == b->size && t->size == a->size) {
#if RPITORCH_HAS_NEON
            #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
            for (uint32_t base = 0; base < t->size; base += 512) {
                uint32_t end = (base + 512 < t->size) ? base + 512 : t->size;
                uint32_t i = base;
                for (; i + 16 <= end; i += 16) {
                    float32x4_t ag0 = vld1q_f32(&a->grad[i + 0]);
                    float32x4_t ag1 = vld1q_f32(&a->grad[i + 4]);
                    float32x4_t ag2 = vld1q_f32(&a->grad[i + 8]);
                    float32x4_t ag3 = vld1q_f32(&a->grad[i + 12]);
                    ag0 = vfmaq_f32(ag0, vld1q_f32(&t->grad[i + 0]),  vld1q_f32(&b->data[i + 0]));
                    ag1 = vfmaq_f32(ag1, vld1q_f32(&t->grad[i + 4]),  vld1q_f32(&b->data[i + 4]));
                    ag2 = vfmaq_f32(ag2, vld1q_f32(&t->grad[i + 8]),  vld1q_f32(&b->data[i + 8]));
                    ag3 = vfmaq_f32(ag3, vld1q_f32(&t->grad[i + 12]), vld1q_f32(&b->data[i + 12]));
                    vst1q_f32(&a->grad[i + 0],  ag0);
                    vst1q_f32(&a->grad[i + 4],  ag1);
                    vst1q_f32(&a->grad[i + 8],  ag2);
                    vst1q_f32(&a->grad[i + 12], ag3);
                }
                for (; i + 4 <= end; i += 4) {
                    float32x4_t ag = vld1q_f32(&a->grad[i]);
                    ag = vfmaq_f32(ag, vld1q_f32(&t->grad[i]), vld1q_f32(&b->data[i]));
                    vst1q_f32(&a->grad[i], ag);
                }
                for (; i < end; i++) a->grad[i] += t->grad[i] * b->data[i];
            }
#else
            #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
            for (uint32_t i = 0; i < t->size; i++) a->grad[i] += t->grad[i] * b->data[i];
#endif
        } else {
            #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
            for (uint32_t i = 0; i < t->size; i++) {
                float val_b = b->data[i % b->size];
                a->grad[i] += t->grad[i] * val_b;
            }
        }
    }
    if (b && b->requires_grad) {
        if (a->size == b->size && t->size == b->size) {
#if RPITORCH_HAS_NEON
            #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
            for (uint32_t base = 0; base < t->size; base += 512) {
                uint32_t end = (base + 512 < t->size) ? base + 512 : t->size;
                uint32_t i = base;
                for (; i + 16 <= end; i += 16) {
                    float32x4_t bg0 = vld1q_f32(&b->grad[i + 0]);
                    float32x4_t bg1 = vld1q_f32(&b->grad[i + 4]);
                    float32x4_t bg2 = vld1q_f32(&b->grad[i + 8]);
                    float32x4_t bg3 = vld1q_f32(&b->grad[i + 12]);
                    bg0 = vfmaq_f32(bg0, vld1q_f32(&t->grad[i + 0]),  vld1q_f32(&a->data[i + 0]));
                    bg1 = vfmaq_f32(bg1, vld1q_f32(&t->grad[i + 4]),  vld1q_f32(&a->data[i + 4]));
                    bg2 = vfmaq_f32(bg2, vld1q_f32(&t->grad[i + 8]),  vld1q_f32(&a->data[i + 8]));
                    bg3 = vfmaq_f32(bg3, vld1q_f32(&t->grad[i + 12]), vld1q_f32(&a->data[i + 12]));
                    vst1q_f32(&b->grad[i + 0],  bg0);
                    vst1q_f32(&b->grad[i + 4],  bg1);
                    vst1q_f32(&b->grad[i + 8],  bg2);
                    vst1q_f32(&b->grad[i + 12], bg3);
                }
                for (; i + 4 <= end; i += 4) {
                    float32x4_t bg = vld1q_f32(&b->grad[i]);
                    bg = vfmaq_f32(bg, vld1q_f32(&t->grad[i]), vld1q_f32(&a->data[i]));
                    vst1q_f32(&b->grad[i], bg);
                }
                for (; i < end; i++) b->grad[i] += t->grad[i] * a->data[i];
            }
#else
            #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
            for (uint32_t i = 0; i < t->size; i++) b->grad[i] += t->grad[i] * a->data[i];
#endif
        } else {
            #pragma omp parallel for if(b->size >= RPL_OMP_THRESHOLD)
            for (uint32_t i = 0; i < b->size; i++) {
                float g = 0;
                for (uint32_t j = i; j < t->size; j += b->size) {
                    g += t->grad[j] * a->data[j];
                }
                b->grad[i] += g;
            }
        }
    }
}

void backward_div(Tensor* t) {
    Tensor* a = t->_parents[0];
    Tensor* b = t->_parents[1];
    if (a && a->requires_grad) {
        #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
        for (uint32_t i = 0; i < t->size; i++) {
            float val_b = b->data[i % b->size];
            a->grad[i] += t->grad[i] / (val_b + 1e-12f);
        }
    }
    if (b && b->requires_grad) {
        #pragma omp parallel for if(b->size >= RPL_OMP_THRESHOLD)
        for (uint32_t i = 0; i < b->size; i++) {
            float g = 0;
            for (uint32_t j = i; j < t->size; j += b->size) {
                float val_b = b->data[i];
                g += t->grad[j] * (-a->data[j] / (val_b * val_b + 1e-12f));
            }
            b->grad[i] += g;
        }
    }
}

void backward_matmul(Tensor* t) {
    Tensor* a = t->_parents[0];
    Tensor* b = t->_parents[1];
    uint32_t M = t->shape[0];
    uint32_t N = t->shape[1];
    uint32_t K = a->shape[1];
    
    bool b_trans = (b->shape[0] == N && b->shape[1] == K);
    
    if (a && a->requires_grad) {
        parallel_gemm_optimized_trans(t->grad, b->data, a->grad, M, K, N, false, !b_trans);
    }
    
    if (b && b->requires_grad) {
        if (b_trans) {
            parallel_gemm_optimized_trans(t->grad, a->data, b->grad, N, K, M, true, false);
        } else {
            parallel_gemm_optimized_trans(a->data, t->grad, b->grad, K, N, M, true, false);
        }
    }
}

void backward_relu(Tensor* t) {
    Tensor* input = t->_parents[0];
    if (!input || !input->requires_grad) return;
#if RPITORCH_HAS_NEON
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    const uint32_t neon_end = (t->size / 4) * 4;
    #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
    for (uint32_t i = 0; i < neon_end; i += 4) {
        uint32x4_t mask = vcgtq_f32(vld1q_f32(&input->data[i]), vzero);
        float32x4_t grad_out = vld1q_f32(&t->grad[i]);
        float32x4_t result = vreinterpretq_f32_u32(vandq_u32(
            vreinterpretq_u32_f32(grad_out), mask));
        vst1q_f32(&input->grad[i], vaddq_f32(vld1q_f32(&input->grad[i]), result));
    }
    for (uint32_t i = (t->size / 4) * 4; i < t->size; i++) {
        if (input->data[i] > 0.0f) input->grad[i] += t->grad[i];
    }
#else
    #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
    for (uint32_t i = 0; i < t->size; i++) {
        if (input->data[i] > 0.0f) input->grad[i] += t->grad[i];
    }
#endif
}

void backward_sigmoid(Tensor* t) {
    Tensor* a = t->_parents[0];
    if (!a || !a->requires_grad) return;
#if RPITORCH_HAS_NEON
    float32x4_t vone = vdupq_n_f32(1.0f);
    const uint32_t neon_end = (t->size / 4) * 4;
    #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
    for (uint32_t i = 0; i < neon_end; i += 4) {
        float32x4_t s = vld1q_f32(&t->data[i]);
        float32x4_t grad_out = vld1q_f32(&t->grad[i]);
        float32x4_t one_minus_s = vsubq_f32(vone, s);
        float32x4_t ds = vmulq_f32(s, one_minus_s);
        float32x4_t result = vmulq_f32(grad_out, ds);
        vst1q_f32(&a->grad[i], vaddq_f32(vld1q_f32(&a->grad[i]), result));
    }
    for (uint32_t i = (t->size / 4) * 4; i < t->size; i++) {
        float s = t->data[i];
        a->grad[i] += t->grad[i] * s * (1.0f - s);
    }
#else
    #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
    for (uint32_t i = 0; i < t->size; i++) {
        float s = t->data[i];
        a->grad[i] += t->grad[i] * s * (1.0f - s);
    }
#endif
}

void backward_tanh(Tensor* t) {
    Tensor* a = t->_parents[0];
    if (!a || !a->requires_grad) return;
#if RPITORCH_HAS_NEON
    float32x4_t vone = vdupq_n_f32(1.0f);
    const uint32_t neon_end = (t->size / 4) * 4;
    #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
    for (uint32_t i = 0; i < neon_end; i += 4) {
        float32x4_t s = vld1q_f32(&t->data[i]);
        float32x4_t grad_out = vld1q_f32(&t->grad[i]);
        float32x4_t s2 = vmulq_f32(s, s);
        float32x4_t dt = vsubq_f32(vone, s2);
        float32x4_t result = vmulq_f32(grad_out, dt);
        vst1q_f32(&a->grad[i], vaddq_f32(vld1q_f32(&a->grad[i]), result));
    }
    for (uint32_t i = (t->size / 4) * 4; i < t->size; i++) {
        float s = t->data[i];
        a->grad[i] += t->grad[i] * (1.0f - s * s);
    }
#else
    #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
    for (uint32_t i = 0; i < t->size; i++) {
        float s = t->data[i];
        a->grad[i] += t->grad[i] * (1.0f - s * s);
    }
#endif
}

void backward_gelu(Tensor* t) {
    Tensor* input = t->_parents[0];
    if (!input || !input->requires_grad) return;
    #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
    for (uint32_t i = 0; i < t->size; i++) {
        float x = input->data[i];
        float x2 = x * x;
        float inner = 0.79788456f * (x + 0.044715f * x * x2);
        float th = tanhf(inner);
        float sech2 = 1.0f - th * th;
        float d_inner = 0.79788456f * (1.0f + 3.0f * 0.044715f * x2);
        float dy_dx = 0.5f * (1.0f + th) + 0.5f * x * sech2 * d_inner;
        input->grad[i] += t->grad[i] * dy_dx;
    }
}

void backward_softmax(Tensor* t) {
    Tensor* input = t->_parents[0];
    if (!input || !input->requires_grad) return;
    uint32_t dim_size = t->shape[t->dims - 1];
    uint32_t batch = t->size / dim_size;
    
    #pragma omp parallel for
    for (uint32_t b = 0; b < batch; b++) {
        float* dst_grad = &input->grad[b * dim_size];
        const float* out_grad = &t->grad[b * dim_size];
        const float* out_data = &t->data[b * dim_size];
        
        float sum = 0;
        for (uint32_t i = 0; i < dim_size; i++) {
            sum += out_grad[i] * out_data[i];
        }
        
        for (uint32_t i = 0; i < dim_size; i++) {
            dst_grad[i] += out_data[i] * (out_grad[i] - sum);
        }
    }
}

void backward_leaky_relu(Tensor* t) {
    Tensor* input = t->_parents[0];
    if (!input || !input->requires_grad) return;
    float slope = t->_saved_scalar;
#if RPITORCH_HAS_NEON
    float32x4_t vzero = vdupq_n_f32(0.0f);
    float32x4_t vslope = vdupq_n_f32(slope);
    float32x4_t vone = vdupq_n_f32(1.0f);
    const uint32_t neon_end = (t->size / 4) * 4;
    #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
    for (uint32_t i = 0; i < neon_end; i += 4) {
        float32x4_t x = vld1q_f32(&input->data[i]);
        float32x4_t grad_out = vld1q_f32(&t->grad[i]);
        uint32x4_t mask = vcgtq_f32(x, vzero);
        float32x4_t scale = vbslq_f32(mask, vone, vslope);
        float32x4_t result = vmulq_f32(grad_out, scale);
        vst1q_f32(&input->grad[i], vaddq_f32(vld1q_f32(&input->grad[i]), result));
    }
    for (uint32_t i = (t->size / 4) * 4; i < t->size; i++) {
        input->grad[i] += t->grad[i] * (input->data[i] > 0.0f ? 1.0f : slope);
    }
#else
    #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
    for (uint32_t i = 0; i < t->size; i++) {
        input->grad[i] += t->grad[i] * (input->data[i] > 0.0f ? 1.0f : slope);
    }
#endif
}

void backward_elu(Tensor* t) {
    Tensor* input = t->_parents[0];
    if (!input || !input->requires_grad) return;
    float alpha = t->_saved_scalar;
#if RPITORCH_HAS_NEON
    float32x4_t vzero = vdupq_n_f32(0.0f);
    float32x4_t valpha = vdupq_n_f32(alpha);
    float32x4_t vone = vdupq_n_f32(1.0f);
    const uint32_t neon_end = (t->size / 4) * 4;
    #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
    for (uint32_t i = 0; i < neon_end; i += 4) {
        float32x4_t x = vld1q_f32(&input->data[i]);
        float32x4_t y = vld1q_f32(&t->data[i]);
        float32x4_t grad_out = vld1q_f32(&t->grad[i]);
        uint32x4_t mask = vcgtq_f32(x, vzero);
        float32x4_t neg_scale = vaddq_f32(y, valpha);
        float32x4_t scale = vbslq_f32(mask, vone, neg_scale);
        float32x4_t result = vmulq_f32(grad_out, scale);
        vst1q_f32(&input->grad[i], vaddq_f32(vld1q_f32(&input->grad[i]), result));
    }
    for (uint32_t i = (t->size / 4) * 4; i < t->size; i++) {
        input->grad[i] += t->grad[i] * (input->data[i] > 0.0f ? 1.0f : t->data[i] + alpha);
    }
#else
    #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
    for (uint32_t i = 0; i < t->size; i++) {
        input->grad[i] += t->grad[i] * (input->data[i] > 0.0f ? 1.0f : t->data[i] + alpha);
    }
#endif
}

void backward_swish(Tensor* t) {
    Tensor* input = t->_parents[0];
    if (!input || !input->requires_grad) return;
    #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
    for (uint32_t i = 0; i < t->size; i++) {
        float x = input->data[i];
        float sig = 1.0f / (1.0f + expf(-x));
        float y = t->data[i];
        float dy_dx = sig + y * (1.0f - sig);
        input->grad[i] += t->grad[i] * dy_dx;
    }
}

void backward_mse(Tensor* t) {
    Tensor* pred = t->_parents[0];
    Tensor* target = t->_parents[1];
    if (pred && pred->requires_grad) {
        float factor = 2.0f / pred->size;
        #pragma omp parallel for if(pred->size >= RPL_OMP_THRESHOLD)
        for (uint32_t i = 0; i < pred->size; i++) {
            pred->grad[i] += t->grad[0] * factor * (pred->data[i] - target->data[i]);
        }
    }
}

void backward_log(Tensor* t) {
    Tensor* a = t->_parents[0];
    if (!a || !a->requires_grad) return;
    #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
    for (uint32_t i = 0; i < t->size; i++) {
        a->grad[i] += t->grad[i] / (a->data[i] + 1e-12f);
    }
}

void backward_exp(Tensor* t) {
    Tensor* a = t->_parents[0];
    if (!a || !a->requires_grad) return;
    #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
    for (uint32_t i = 0; i < t->size; i++) {
        a->grad[i] += t->grad[i] * t->data[i];
    }
}

void backward_sqrt(Tensor* t) {
    Tensor* a = t->_parents[0];
    if (!a || !a->requires_grad) return;
    #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
    for (uint32_t i = 0; i < t->size; i++) {
        a->grad[i] += t->grad[i] / (2.0f * t->data[i] + 1e-12f);
    }
}

void backward_neg(Tensor* t) {
    Tensor* a = t->_parents[0];
    if (!a || !a->requires_grad) return;
    #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
    for (uint32_t i = 0; i < t->size; i++) {
        a->grad[i] -= t->grad[i];
    }
}

void backward_abs(Tensor* t) {
    Tensor* a = t->_parents[0];
    if (!a || !a->requires_grad) return;
    #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
    for (uint32_t i = 0; i < t->size; i++) {
        float val = a->data[i];
        float sign = (val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f);
        a->grad[i] += t->grad[i] * sign;
    }
}

void backward_sum(Tensor* t) {
    Tensor* a = t->_parents[0];
    if (!a || !a->requires_grad) return;
    
    if (t->size == 1) {
        float g = t->grad[0];
        #pragma omp parallel for if(a->size >= RPL_OMP_THRESHOLD)
        for (uint32_t i = 0; i < a->size; i++) {
            a->grad[i] += g;
        }
    } else {
        #pragma omp parallel for if(a->size >= RPL_OMP_THRESHOLD)
        for (uint32_t i = 0; i < a->size; i++) {
            uint32_t t_idx = 0;
            uint32_t temp = i;
            for (int32_t d = (int32_t)a->dims - 1; d >= 0; d--) {
                uint32_t coord = temp % a->shape[d];
                temp /= a->shape[d];
                if (t->shape[d] > 1) {
                    t_idx += coord * t->strides[d];
                }
            }
            a->grad[i] += t->grad[t_idx];
        }
    }
}

void backward_mean(Tensor* t) {
    Tensor* a = t->_parents[0];
    if (!a || !a->requires_grad) return;
    
    uint32_t N = a->size / t->size;
    float inv_N = 1.0f / N;
    
    if (t->size == 1) {
        float g = t->grad[0] * inv_N;
        #pragma omp parallel for if(a->size >= RPL_OMP_THRESHOLD)
        for (uint32_t i = 0; i < a->size; i++) {
            a->grad[i] += g;
        }
    } else {
        #pragma omp parallel for if(a->size >= RPL_OMP_THRESHOLD)
        for (uint32_t i = 0; i < a->size; i++) {
            uint32_t t_idx = 0;
            uint32_t temp = i;
            for (int32_t d = (int32_t)a->dims - 1; d >= 0; d--) {
                uint32_t coord = temp % a->shape[d];
                temp /= a->shape[d];
                if (t->shape[d] > 1) {
                    t_idx += coord * t->strides[d];
                }
            }
            a->grad[i] += t->grad[t_idx] * inv_N;
        }
    }
}

void backward_add_scalar(Tensor* t) {
    Tensor* a = t->_parents[0];
    if (!a || !a->requires_grad) return;
    #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
    for (uint32_t i = 0; i < t->size; i++) {
        a->grad[i] += t->grad[i];
    }
}

void backward_mul_scalar(Tensor* t) {
    Tensor* a = t->_parents[0];
    if (!a || !a->requires_grad) return;
    float scalar = t->_saved_scalar;
    #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
    for (uint32_t i = 0; i < t->size; i++) {
        a->grad[i] += t->grad[i] * scalar;
    }
}

void backward_selu(Tensor* t) {
    Tensor* input = t->_parents[0];
    if (!input || !input->requires_grad) return;
    const float lambda = 1.0507009873554804934f;
    const float lambda_alpha = 1.75809934f;
    #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
    for (uint32_t i = 0; i < t->size; i++) {
        float x = input->data[i];
        float y = t->data[i];
        float dy_dx = (x > 0.0f) ? lambda : (y + lambda_alpha);
        input->grad[i] += t->grad[i] * dy_dx;
    }
}

void backward_mish(Tensor* t) {
    Tensor* input = t->_parents[0];
    if (!input || !input->requires_grad) return;
    #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
    for (uint32_t i = 0; i < t->size; i++) {
        float x = input->data[i];
        float sp = logf(1.0f + expf(x));
        float th = tanhf(sp);
        float sig = 1.0f / (1.0f + expf(-x));
        float dy_dx = th + x * sig * (1.0f - th * th);
        input->grad[i] += t->grad[i] * dy_dx;
    }
}

void backward_dropout(Tensor* t) {
    Tensor* input = t->_parents[0];
    if (!input || !input->requires_grad) return;
    float p = t->_saved_scalar;
    float scale = 1.0f / (1.0f - p);
    #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
    for (uint32_t i = 0; i < t->size; i++) {
        if (t->data[i] == 0.0f) {
            // was zeroed out, no grad flows back
        } else {
            input->grad[i] += t->grad[i] * scale;
        }
    }
}

void backward_embedding(Tensor* t) {
    Tensor* weight = t->_parents[0];
    Tensor* indices = t->_parents[1];
    if (!weight || !weight->requires_grad) return;
    
    uint32_t embedding_dim = weight->shape[1];
    uint32_t num_indices = indices->size;
    
    for (uint32_t i = 0; i < num_indices; i++) {
        uint32_t idx = (uint32_t)indices->data[i];
        if (idx >= weight->shape[0]) continue;
        
        for (uint32_t j = 0; j < embedding_dim; j++) {
            weight->grad[idx * embedding_dim + j] += t->grad[i * embedding_dim + j];
        }
    }
}

void backward_conv2d(Tensor* t) {
    Tensor* input = t->_parents[0];
    Tensor* kernel = t->_parents[1];
    if (!input || !kernel) return;
    
    uint32_t encoded = (uint32_t)t->_saved_scalar;
    uint32_t stride_val = encoded & 0xFFFF;
    uint32_t padding_val = (encoded >> 16) & 0xFFFF;
    if (stride_val == 0) stride_val = 1;
    
    uint32_t batch = input->shape[0];
    uint32_t out_channels = kernel->shape[0];
    uint32_t in_channels = kernel->shape[1];
    uint32_t kh = kernel->shape[2];
    uint32_t kw = kernel->shape[3];
    uint32_t in_h = input->shape[2];
    uint32_t in_w = input->shape[3];
    uint32_t out_h = t->shape[2];
    uint32_t out_w = t->shape[3];
    
    if (input->requires_grad) {
        #pragma omp parallel for collapse(4)
        for (uint32_t b = 0; b < batch; b++) {
            for (uint32_t ic = 0; ic < in_channels; ic++) {
                for (uint32_t ih = 0; ih < in_h; ih++) {
                    for (uint32_t iw = 0; iw < in_w; iw++) {
                        float sum = 0.0f;
                        for (uint32_t oc = 0; oc < out_channels; oc++) {
                            for (uint32_t k_h = 0; k_h < kh; k_h++) {
                                int32_t oh_stride = (int32_t)ih + (int32_t)padding_val - (int32_t)k_h;
                                if (oh_stride % stride_val != 0) continue;
                                int32_t oh = oh_stride / (int32_t)stride_val;
                                if (oh < 0 || oh >= (int32_t)out_h) continue;
                                
                                for (uint32_t k_w = 0; k_w < kw; k_w++) {
                                    int32_t ow_stride = (int32_t)iw + (int32_t)padding_val - (int32_t)k_w;
                                    if (ow_stride % stride_val != 0) continue;
                                    int32_t ow = ow_stride / (int32_t)stride_val;
                                    if (ow < 0 || ow >= (int32_t)out_w) continue;
                                    
                                    sum += t->grad[((b * out_channels + oc) * out_h + oh) * out_w + ow] *
                                           kernel->data[((oc * in_channels + ic) * kh + k_h) * kw + k_w];
                                }
                            }
                        }
                        input->grad[((b * in_channels + ic) * in_h + ih) * in_w + iw] += sum;
                    }
                }
            }
        }
    }
    
    if (kernel->requires_grad) {
        #pragma omp parallel for collapse(4)
        for (uint32_t oc = 0; oc < out_channels; oc++) {
            for (uint32_t ic = 0; ic < in_channels; ic++) {
                for (uint32_t k_h = 0; k_h < kh; k_h++) {
                    for (uint32_t k_w = 0; k_w < kw; k_w++) {
                        float sum = 0.0f;
                        for (uint32_t b = 0; b < batch; b++) {
                            for (uint32_t oh = 0; oh < out_h; oh++) {
                                for (uint32_t ow = 0; ow < out_w; ow++) {
                                    int32_t ih = oh * stride_val + k_h - padding_val;
                                    int32_t iw = ow * stride_val + k_w - padding_val;
                                    if (ih >= 0 && ih < (int32_t)in_h && iw >= 0 && iw < (int32_t)in_w) {
                                        sum += t->grad[((b * out_channels + oc) * out_h + oh) * out_w + ow] *
                                               input->data[((b * in_channels + ic) * in_h + ih) * in_w + iw];
                                    }
                                }
                            }
                        }
                        kernel->grad[((oc * in_channels + ic) * kh + k_h) * kw + k_w] += sum;
                    }
                }
            }
        }
    }
    
    Tensor* bias = t->_n_parents > 2 ? t->_parents[2] : NULL;
    if (bias && bias->requires_grad) {
        #pragma omp parallel for
        for (uint32_t oc = 0; oc < out_channels; oc++) {
            float sum = 0.0f;
            for (uint32_t b = 0; b < batch; b++) {
                for (uint32_t oh = 0; oh < out_h; oh++) {
                    for (uint32_t ow = 0; ow < out_w; ow++) {
                        sum += t->grad[((b * out_channels + oc) * out_h + oh) * out_w + ow];
                    }
                }
            }
            bias->grad[oc] += sum;
        }
    }
}

void backward_maxpool2d(Tensor* t) {
    Tensor* input = t->_parents[0];
    if (!input || !input->requires_grad) return;
    
    uint32_t encoded = (uint32_t)t->_saved_scalar;
    uint32_t kernel_size = encoded & 0xFFFF;
    uint32_t stride = (encoded >> 16) & 0xFFFF;
    if (kernel_size == 0) kernel_size = 2;
    if (stride == 0) stride = 2;
    
    uint32_t batch = input->shape[0];
    uint32_t channels = input->shape[1];
    uint32_t in_h = input->shape[2];
    uint32_t in_w = input->shape[3];
    uint32_t out_h = t->shape[2];
    uint32_t out_w = t->shape[3];
    
    #pragma omp parallel for collapse(4)
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t c = 0; c < channels; c++) {
            for (uint32_t oh = 0; oh < out_h; oh++) {
                for (uint32_t ow = 0; ow < out_w; ow++) {
                    float max_val = -FLT_MAX;
                    uint32_t max_ih = 0;
                    uint32_t max_iw = 0;
                    
                    for (uint32_t kh = 0; kh < kernel_size; kh++) {
                        for (uint32_t kw = 0; kw < kernel_size; kw++) {
                            uint32_t ih = oh * stride + kh;
                            uint32_t iw = ow * stride + kw;
                            if (ih < in_h && iw < in_w) {
                                float val = input->data[((b * channels + c) * in_h + ih) * in_w + iw];
                                if (val > max_val) {
                                    max_val = val;
                                    max_ih = ih;
                                    max_iw = iw;
                                }
                            }
                        }
                    }
                    
                    input->grad[((b * channels + c) * in_h + max_ih) * in_w + max_iw] +=
                        t->grad[((b * channels + c) * out_h + oh) * out_w + ow];
                }
            }
        }
    }
}

void backward_layernorm(Tensor* t) {
    Tensor* input = t->_parents[0];
    Tensor* gamma = t->_parents[1];
    Tensor* beta = t->_parents[2];
    if (!input) return;
    
    float eps = t->_saved_scalar;
    if (eps == 0.0f) eps = 1e-5f;
    
    uint32_t outer_size = 1;
    for (uint32_t i = 0; i < input->dims - 1; i++) {
        outer_size *= input->shape[i];
    }
    uint32_t inner_size = input->shape[input->dims - 1];
    
    if (beta && beta->requires_grad) {
        #pragma omp parallel for
        for (uint32_t j = 0; j < inner_size; j++) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < outer_size; i++) {
                sum += t->grad[i * inner_size + j];
            }
            beta->grad[j] += sum;
        }
    }
    
    if (gamma && gamma->requires_grad) {
        #pragma omp parallel for
        for (uint32_t j = 0; j < inner_size; j++) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < outer_size; i++) {
                uint32_t offset = i * inner_size;
                float mean = 0.0f, sq_sum = 0.0f;
                for (uint32_t k = 0; k < inner_size; k++) {
                    mean += input->data[offset + k];
                    sq_sum += input->data[offset + k] * input->data[offset + k];
                }
                mean /= inner_size;
                float var = (sq_sum / inner_size) - (mean * mean);
                float inv_std = 1.0f / sqrtf(var + eps);
                float normalized = (input->data[offset + j] - mean) * inv_std;
                sum += t->grad[offset + j] * normalized;
            }
            gamma->grad[j] += sum;
        }
    }
    
    if (input->requires_grad) {
        #pragma omp parallel for
        for (uint32_t i = 0; i < outer_size; i++) {
            uint32_t offset = i * inner_size;
            float mean = 0.0f, sq_sum = 0.0f;
            for (uint32_t j = 0; j < inner_size; j++) {
                mean += input->data[offset + j];
                sq_sum += input->data[offset + j] * input->data[offset + j];
            }
            mean /= inner_size;
            float var = (sq_sum / inner_size) - (mean * mean);
            float inv_std = 1.0f / sqrtf(var + eps);
            
            float sum_dy_w = 0.0f;
            float sum_dy_w_y = 0.0f;
            for (uint32_t j = 0; j < inner_size; j++) {
                float normalized = (input->data[offset + j] - mean) * inv_std;
                float w = gamma ? gamma->data[j] : 1.0f;
                float dy = t->grad[offset + j];
                sum_dy_w += dy * w;
                sum_dy_w_y += dy * w * normalized;
            }
            
            float mean_dy_w = sum_dy_w / inner_size;
            float mean_dy_w_y = sum_dy_w_y / inner_size;
            
            for (uint32_t j = 0; j < inner_size; j++) {
                float normalized = (input->data[offset + j] - mean) * inv_std;
                float w = gamma ? gamma->data[j] : 1.0f;
                float dy = t->grad[offset + j];
                input->grad[offset + j] += (dy * w - mean_dy_w - normalized * mean_dy_w_y) * inv_std;
            }
        }
    }
}

void backward_batchnorm(Tensor* t) {
    Tensor* input = t->_parents[0];
    Tensor* gamma = t->_parents[1];
    Tensor* beta = t->_parents[2];
    if (!input) return;
    
    float eps = t->_saved_scalar;
    if (eps == 0.0f) eps = 1e-5f;
    
    uint32_t N = input->shape[0];
    uint32_t C = input->shape[1];
    uint32_t H = input->shape[2];
    uint32_t W = input->shape[3];
    uint32_t spatial_size = H * W;
    uint32_t M = N * spatial_size;
    
    if (beta && beta->requires_grad) {
        #pragma omp parallel for
        for (uint32_t c = 0; c < C; c++) {
            float sum = 0.0f;
            for (uint32_t n = 0; n < N; n++) {
                for (uint32_t i = 0; i < spatial_size; i++) {
                    sum += t->grad[((n * C + c) * H) * W + i];
                }
            }
            beta->grad[c] += sum;
        }
    }
    
    if (gamma && gamma->requires_grad) {
        #pragma omp parallel for
        for (uint32_t c = 0; c < C; c++) {
            float sum_x = 0.0f, sum_x2 = 0.0f;
            for (uint32_t n = 0; n < N; n++) {
                for (uint32_t i = 0; i < spatial_size; i++) {
                    float val = input->data[((n * C + c) * H) * W + i];
                    sum_x += val;
                    sum_x2 += val * val;
                }
            }
            float mean = sum_x / M;
            float var = (sum_x2 / M) - (mean * mean);
            float inv_std = 1.0f / sqrtf(var + eps);
            
            float sum_dy_norm = 0.0f;
            for (uint32_t n = 0; n < N; n++) {
                for (uint32_t i = 0; i < spatial_size; i++) {
                    uint32_t idx = ((n * C + c) * H) * W + i;
                    float normalized = (input->data[idx] - mean) * inv_std;
                    sum_dy_norm += t->grad[idx] * normalized;
                }
            }
            gamma->grad[c] += sum_dy_norm;
        }
    }
    
    if (input->requires_grad) {
        #pragma omp parallel for
        for (uint32_t c = 0; c < C; c++) {
            float sum_x = 0.0f, sum_x2 = 0.0f;
            for (uint32_t n = 0; n < N; n++) {
                for (uint32_t i = 0; i < spatial_size; i++) {
                    float val = input->data[((n * C + c) * H) * W + i];
                    sum_x += val;
                    sum_x2 += val * val;
                }
            }
            float mean = sum_x / M;
            float var = (sum_x2 / M) - (mean * mean);
            float inv_std = 1.0f / sqrtf(var + eps);
            
            float sum_dy = 0.0f;
            float sum_dy_norm = 0.0f;
            float w = gamma ? gamma->data[c] : 1.0f;
            
            for (uint32_t n = 0; n < N; n++) {
                for (uint32_t i = 0; i < spatial_size; i++) {
                    uint32_t idx = ((n * C + c) * H) * W + i;
                    float normalized = (input->data[idx] - mean) * inv_std;
                    float dy = t->grad[idx];
                    sum_dy += dy;
                    sum_dy_norm += dy * normalized;
                }
            }
            
            for (uint32_t n = 0; n < N; n++) {
                for (uint32_t i = 0; i < spatial_size; i++) {
                    uint32_t idx = ((n * C + c) * H) * W + i;
                    float normalized = (input->data[idx] - mean) * inv_std;
                    float dy = t->grad[idx];
                    input->grad[idx] += w * inv_std / M * (M * dy - sum_dy - normalized * sum_dy_norm);
                }
            }
        }
    }
}

void backward_rmsnorm(Tensor* t) {
    Tensor* input = t->_parents[0];
    Tensor* gamma = t->_parents[1];
    if (!input) return;
    
    float eps = t->_saved_scalar;
    if (eps == 0.0f) eps = 1e-5f;
    
    uint32_t outer_size = 1;
    for (uint32_t i = 0; i < input->dims - 1; i++) {
        outer_size *= input->shape[i];
    }
    uint32_t inner_size = input->shape[input->dims - 1];
    
    if (gamma && gamma->requires_grad) {
        #pragma omp parallel for
        for (uint32_t j = 0; j < inner_size; j++) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < outer_size; i++) {
                uint32_t offset = i * inner_size;
                float sq_sum = 0.0f;
                for (uint32_t k = 0; k < inner_size; k++) {
                    sq_sum += input->data[offset + k] * input->data[offset + k];
                }
                float rms = sqrtf(sq_sum / inner_size + eps);
                float normalized = input->data[offset + j] / rms;
                sum += t->grad[offset + j] * normalized;
            }
            gamma->grad[j] += sum;
        }
    }
    
    if (input->requires_grad) {
        #pragma omp parallel for
        for (uint32_t i = 0; i < outer_size; i++) {
            uint32_t offset = i * inner_size;
            float sq_sum = 0.0f;
            for (uint32_t j = 0; j < inner_size; j++) {
                sq_sum += input->data[offset + j] * input->data[offset + j];
            }
            float rms = sqrtf(sq_sum / inner_size + eps);
            
            float sum_dy_w_y = 0.0f;
            for (uint32_t j = 0; j < inner_size; j++) {
                float normalized = input->data[offset + j] / rms;
                float w = gamma ? gamma->data[j] : 1.0f;
                sum_dy_w_y += t->grad[offset + j] * w * normalized;
            }
            
            for (uint32_t j = 0; j < inner_size; j++) {
                float normalized = input->data[offset + j] / rms;
                float w = gamma ? gamma->data[j] : 1.0f;
                float dy = t->grad[offset + j];
                input->grad[offset + j] += (dy * w - normalized * sum_dy_w_y / inner_size) / rms;
            }
        }
    }
}

void backward_reshape(Tensor* t) {
    Tensor* a = t->_parents[0];
    if (!a || !a->requires_grad) return;
    #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
    for (uint32_t i = 0; i < t->size; i++) {
        a->grad[i] += t->grad[i];
    }
}

void backward_transpose(Tensor* t) {
    Tensor* a = t->_parents[0];
    if (!a || !a->requires_grad) return;
    
    if (t->dims == 2) {
        uint32_t R = t->shape[0];
        uint32_t C = t->shape[1];
        #pragma omp parallel for collapse(2)
        for (uint32_t i = 0; i < R; i++) {
            for (uint32_t j = 0; j < C; j++) {
                a->grad[j * R + i] += t->grad[i * C + j];
            }
        }
    } else {
        #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
        for (uint32_t i = 0; i < t->size; i++) {
            uint32_t temp = i;
            uint32_t coords[MAX_DIMS];
            for (int32_t d = (int32_t)t->dims - 1; d >= 0; d--) {
                coords[d] = temp % t->shape[d];
                temp /= t->shape[d];
            }
            uint32_t d0 = t->dims - 2;
            uint32_t d1 = t->dims - 1;
            uint32_t temp_c = coords[d0];
            coords[d0] = coords[d1];
            coords[d1] = temp_c;
            
            uint32_t a_idx = 0;
            for (uint32_t d = 0; d < a->dims; d++) {
                a_idx += coords[d] * a->strides[d];
            }
            a->grad[a_idx] += t->grad[i];
        }
    }
}

void backward_slice(Tensor* t) {
    Tensor* a = t->_parents[0];
    if (!a || !a->requires_grad) return;
    
    uint32_t encoded = (uint32_t)t->_saved_scalar;
    uint32_t dim = encoded & 0xFF;
    uint32_t start = (encoded >> 8) & 0xFF;
    
    #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
    for (uint32_t i = 0; i < t->size; i++) {
        uint32_t temp = i;
        uint32_t a_idx = 0;
        for (int32_t d = (int32_t)t->dims - 1; d >= 0; d--) {
            uint32_t coord = temp % t->shape[d];
            temp /= t->shape[d];
            if (d == (int32_t)dim) {
                coord += start;
            }
            a_idx += coord * a->strides[d];
        }
        a->grad[a_idx] += t->grad[i];
    }
}

// Main topological sort autograd entrypoint
void tensor_backward(Tensor* t) {
    if (!t->requires_grad) return;
    
    int stack_cap = 1024;
    StackFrame* stack = malloc(stack_cap * sizeof(StackFrame));
    int stack_size = 0;

    int topo_cap = 1024;
    Tensor** topo_order = malloc(topo_cap * sizeof(Tensor*));
    int topo_size = 0;

    stack[stack_size].node = t;
    stack[stack_size].parent_idx = 0;
    stack_size++;

    while (stack_size > 0) {
        StackFrame* frame = &stack[stack_size - 1];
        Tensor* node = frame->node;
        
        if (!node) {
            stack_size--;
            continue;
        }
        
        if (node->_visited) {
            stack_size--;
            continue;
        }
        
        if (frame->parent_idx < node->_n_parents) {
            Tensor* parent = node->_parents[frame->parent_idx];
            frame->parent_idx++;
            
            if (parent && parent->requires_grad && !parent->_visited) {
                if (stack_size >= stack_cap) {
                    stack_cap *= 2;
                    stack = realloc(stack, stack_cap * sizeof(StackFrame));
                }
                stack[stack_size].node = parent;
                stack[stack_size].parent_idx = 0;
                stack_size++;
            }
        } else {
            node->_visited = true;
            if (topo_size >= topo_cap) {
                topo_cap *= 2;
                topo_order = realloc(topo_order, topo_cap * sizeof(Tensor*));
            }
            topo_order[topo_size++] = node;
            stack_size--;
        }
    }

    free(stack);

    if (!t->grad) {
        t->grad = (float*)pool_alloc(t->_alloc_size);
    }
    for (uint32_t i = 0; i < t->size; i++) {
        t->grad[i] = 1.0f;
    }

    for (int i = topo_size - 1; i >= 0; i--) {
        Tensor* node = topo_order[i];
        if (!node->grad) continue;
        
        for (int p = 0; p < node->_n_parents; p++) {
            Tensor* parent = node->_parents[p];
            if (parent && parent->requires_grad && !parent->grad) {
                parent->grad = (float*)pool_alloc(parent->_alloc_size);
                memset(parent->grad, 0, parent->_alloc_size);
            }
        }
        
        switch (node->_op) {
            case OP_ADD: backward_add(node); break;
            case OP_SUB: backward_sub(node); break;
            case OP_MUL: backward_mul(node); break;
            case OP_DIV: backward_div(node); break;
            case OP_MATMUL: backward_matmul(node); break;
            case OP_RELU: backward_relu(node); break;
            case OP_SIGMOID: backward_sigmoid(node); break;
            case OP_TANH: backward_tanh(node); break;
            case OP_GELU: backward_gelu(node); break;
            case OP_SOFTMAX: backward_softmax(node); break;
            case OP_LEAKY_RELU: backward_leaky_relu(node); break;
            case OP_ELU: backward_elu(node); break;
            case OP_SWISH: backward_swish(node); break;
            case OP_MSE_LOSS: backward_mse(node); break;
            case OP_LOG: backward_log(node); break;
            case OP_EXP: backward_exp(node); break;
            case OP_SQRT: backward_sqrt(node); break;
            case OP_NEG: backward_neg(node); break;
            case OP_ABS: backward_abs(node); break;
            case OP_SUM: backward_sum(node); break;
            case OP_MEAN: backward_mean(node); break;
            case OP_ADD_SCALAR: backward_add_scalar(node); break;
            case OP_MUL_SCALAR: backward_mul_scalar(node); break;
            case OP_SELU: backward_selu(node); break;
            case OP_MISH: backward_mish(node); break;
            case OP_DROPOUT: backward_dropout(node); break;
            case OP_EMBEDDING: backward_embedding(node); break;
            case OP_CONV2D: backward_conv2d(node); break;
            case OP_MAXPOOL2D: backward_maxpool2d(node); break;
            case OP_LAYERNORM: backward_layernorm(node); break;
            case OP_BATCHNORM: backward_batchnorm(node); break;
            case OP_RMSNORM: backward_rmsnorm(node); break;
            case OP_RESHAPE: backward_reshape(node); break;
            case OP_TRANSPOSE: backward_transpose(node); break;
            case OP_SLICE: backward_slice(node); break;
            default:
                if (node->backward_fn) {
                    node->backward_fn(node);
                }
                break;
        }
        
        if (!node->is_leaf && node->grad) {
            pool_free(node->grad, node->_alloc_size);
            node->grad = NULL;
        }
    }

    for (int i = 0; i < topo_size; i++) {
        topo_order[i]->_visited = false;
    }

    // Release references to parents for non-leaf intermediate nodes to free graph memory
    for (int i = 0; i < topo_size; i++) {
        Tensor* node = topo_order[i];
        if (!node->is_leaf) {
            for (int p = 0; p < node->_n_parents; p++) {
                if (node->_parents[p]) {
                    tensor_release(node->_parents[p]);
                    node->_parents[p] = NULL;
                }
            }
            node->_n_parents = 0;
        }
    }

    free(topo_order);
}

void tensor_zero_grad(Tensor* t) {
    if (t && t->grad) {
        memset(t->grad, 0, t->size * sizeof(float));
    }
}

#include "rpl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// BatchNorm2D forward pass
void tensor_batchnorm2d_forward(Tensor* input, Tensor* output,
                                const float* gamma, const float* beta,
                                const float* running_mean, const float* running_var,
                                float eps, bool training, float momentum) {
    // legacy wrapper or implementation
    tensor_batchnorm2d(output, input, (float*)gamma, (float*)beta, (float*)running_mean, (float*)running_var, eps, training, momentum);
}

// LayerNorm forward pass
void tensor_layernorm_forward(Tensor* input, Tensor* output,
                              const float* gamma, const float* beta,
                              float eps) {
    // Normalize over last dimension
    uint32_t outer_size = 1;
    for (uint32_t i = 0; i < input->dims - 1; i++) {
        outer_size *= input->shape[i];
    }
    uint32_t inner_size = input->shape[input->dims - 1];
    
    #pragma omp parallel for
    for (uint32_t i = 0; i < outer_size; i++) {
        // Compute mean and variance
        float sum = 0.0f, sum_sq = 0.0f;
        uint32_t offset = i * inner_size;
        
        for (uint32_t j = 0; j < inner_size; j++) {
            float val = input->data[offset + j];
            sum += val;
            sum_sq += val * val;
        }
        
        float mean = sum / inner_size;
        float var = (sum_sq / inner_size) - (mean * mean);
        float inv_std = 1.0f / sqrtf(var + eps);
        
        // Normalize and scale
        for (uint32_t j = 0; j < inner_size; j++) {
            float normalized = (input->data[offset + j] - mean) * inv_std;
            output->data[offset + j] = gamma[j] * normalized + beta[j];
        }
    }
}

// Get min/max for calibration
void tensor_get_min_max(const Tensor* t, float* min_val, float* max_val) {
    if (t->size == 0) return;
    *min_val = t->data[0];
    *max_val = t->data[0];
    
    for (uint32_t i = 1; i < t->size; i++) {
        if (t->data[i] < *min_val) *min_val = t->data[i];
        if (t->data[i] > *max_val) *max_val = t->data[i];
    }
}

// Dequantize INT8 tensor back to FP32
Tensor* tensor_dequantize_int8(QuantizedTensor* input) {
    Tensor* output = tensor_create(input->dims, input->shape, false);
    
    #pragma omp parallel for
    for (uint32_t i = 0; i < input->size; i++) {
        output->data[i] = (input->data[i] - input->zero_point) * input->scale;
    }
    
    return output;
}

void quantized_tensor_free(QuantizedTensor* qt) {
    if (qt) {
        if (qt->data) rpitorch_aligned_free(qt->data);
        free(qt);
    }
}

// INT8 GEMM (optimized for RPi4)
void gemm_int8_fallback(const int8_t* A, const int8_t* B, int32_t* C,
               uint32_t M, uint32_t N, uint32_t K,
               float scale_a, float scale_b, float scale_c,
               int32_t zero_point_a, int32_t zero_point_b) {
    
    #pragma omp parallel for collapse(2)
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            int32_t sum = 0;
            
            #if RPITORCH_HAS_NEON
            // Use NEON SDOT instruction for 4x INT8 MAC if available,
            // or use standard NEON VMULL/VPADD
            uint32_t k = 0;
            int32x4_t vsum = vdupq_n_s32(0);
            
            for (; k <= K - 8; k += 8) {
                int8x8_t va = vld1_s8(&A[i * K + k]);
                int8x8_t vb = vld1_s8(&B[k * N + j]);
                
                int16x8_t vmul = vmull_s8(va, vb);
                vsum = vaddq_s32(vsum, vpaddlq_s16(vmul));
            }
            
            sum = vgetq_lane_s32(vsum, 0) + vgetq_lane_s32(vsum, 1) +
                  vgetq_lane_s32(vsum, 2) + vgetq_lane_s32(vsum, 3);
            
            for (; k < K; k++) {
                sum += (A[i * K + k] - zero_point_a) * (B[k * N + j] - zero_point_b);
            }
            #else
            for (uint32_t k = 0; k < K; k++) {
                sum += (int32_t)(A[i * K + k] - zero_point_a) * (int32_t)(B[k * N + j] - zero_point_b);
            }
            #endif
            
            float result = sum * scale_a * scale_b / scale_c;
            C[i * N + j] = (int32_t)roundf(result);
        }
    }
}
void gemm_int8(const int8_t* A, const int8_t* B, int32_t* C,
               uint32_t M, uint32_t N, uint32_t K,
               float scale_a, float scale_b, float scale_c,
               int32_t zero_point_a, int32_t zero_point_b) {
    gemm_int8_fallback(A, B, C, M, N, K, scale_a, scale_b, scale_c, zero_point_a, zero_point_b);
}

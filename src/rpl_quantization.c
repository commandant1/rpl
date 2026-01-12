/*
 * Complete Quantization Implementation in C
 * INT8/INT16 quantization with calibration and QAT support
 */

#include "rpl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

void backward_fake_quant(Tensor* t);

// ============================================================
// Quantization Calibration
// ============================================================

typedef struct {
    float min_val;
    float max_val;
    float scale;
    int32_t zero_point;
    uint32_t num_samples;
} QuantizationCalibrator;

QuantizationCalibrator* calibrator_create() {
    QuantizationCalibrator* cal = (QuantizationCalibrator*)calloc(1, sizeof(QuantizationCalibrator));
    cal->min_val = FLT_MAX;
    cal->max_val = -FLT_MAX;
    return cal;
}

void calibrator_update(QuantizationCalibrator* cal, const Tensor* t) {
    #pragma omp parallel
    {
        float local_min = FLT_MAX;
        float local_max = -FLT_MAX;
        
        #pragma omp for
        for (uint32_t i = 0; i < t->size; i++) {
            float v = t->data[i];
            if (v < local_min) local_min = v;
            if (v > local_max) local_max = v;
        }
        
        #pragma omp critical
        {
            if (local_min < cal->min_val) cal->min_val = local_min;
            if (local_max > cal->max_val) cal->max_val = local_max;
        }
    }
    
    cal->num_samples++;
}

void calibrator_compute_params(QuantizationCalibrator* cal, bool symmetric) {
    if (symmetric) {
        float abs_max = fmaxf(fabsf(cal->min_val), fabsf(cal->max_val));
        cal->scale = abs_max / 127.0f;
        cal->zero_point = 0;
    } else {
        // Asymmetric quantization
        float range = cal->max_val - cal->min_val;
        cal->scale = range / 255.0f;
        cal->zero_point = (int32_t)roundf(-cal->min_val / cal->scale);
    }
}

void calibrator_free(QuantizationCalibrator* cal) {
    free(cal);
}

// ============================================================
// Tensor Quantization
// ============================================================

QuantizedTensor* tensor_quantize_int8_symmetric(const Tensor* input) {
    // Find min/max
    float min_val = FLT_MAX, max_val = -FLT_MAX;
    
    #pragma omp parallel for reduction(min:min_val) reduction(max:max_val)
    for (uint32_t i = 0; i < input->size; i++) {
        if (input->data[i] < min_val) min_val = input->data[i];
        if (input->data[i] > max_val) max_val = input->data[i];
    }
    
    float abs_max = fmaxf(fabsf(min_val), fabsf(max_val));
    float scale = abs_max / 127.0f;
    
    return tensor_quantize_int8(input, scale, 0);
}

QuantizedTensor* tensor_quantize_int8_asymmetric(const Tensor* input) {
    float min_val = FLT_MAX, max_val = -FLT_MAX;
    
    #pragma omp parallel for reduction(min:min_val) reduction(max:max_val)
    for (uint32_t i = 0; i < input->size; i++) {
        if (input->data[i] < min_val) min_val = input->data[i];
        if (input->data[i] > max_val) max_val = input->data[i];
    }
    
    float range = max_val - min_val;
    float scale = range / 255.0f;
    int32_t zero_point = (int32_t)roundf(-min_val / scale);
    
    return tensor_quantize_int8(input, scale, zero_point);
}

// Per-channel quantization for weights
typedef struct {
    int8_t* data;
    float* scales;        // Per-channel scales
    int32_t* zero_points; // Per-channel zero points
    uint32_t num_channels;
    uint32_t channel_size;
    uint32_t dims;
    uint32_t shape[RPITORCH_MAX_DIMS];
} PerChannelQuantizedTensor;

PerChannelQuantizedTensor* tensor_quantize_per_channel(const Tensor* input, 
                                                       uint32_t channel_axis) {
    PerChannelQuantizedTensor* qt = (PerChannelQuantizedTensor*)calloc(1, sizeof(PerChannelQuantizedTensor));
    
    qt->num_channels = input->shape[channel_axis];
    qt->channel_size = input->size / qt->num_channels;
    qt->dims = input->dims;
    memcpy(qt->shape, input->shape, input->dims * sizeof(uint32_t));
    
    qt->data = (int8_t*)rpitorch_aligned_alloc(RPITORCH_CACHE_LINE, input->size);
    qt->scales = (float*)malloc(qt->num_channels * sizeof(float));
    qt->zero_points = (int32_t*)malloc(qt->num_channels * sizeof(int32_t));
    
    // Quantize each channel independently
    #pragma omp parallel for
    for (uint32_t c = 0; c < qt->num_channels; c++) {
        float min_val = FLT_MAX, max_val = -FLT_MAX;
        
        // Find min/max for this channel
        for (uint32_t i = 0; i < qt->channel_size; i++) {
            uint32_t idx = c * qt->channel_size + i;
            if (input->data[idx] < min_val) min_val = input->data[idx];
            if (input->data[idx] > max_val) max_val = input->data[idx];
        }
        
        // Compute scale and zero point
        float abs_max = fmaxf(fabsf(min_val), fabsf(max_val));
        qt->scales[c] = abs_max / 127.0f;
        qt->zero_points[c] = 0;
        
        // Quantize
        for (uint32_t i = 0; i < qt->channel_size; i++) {
            uint32_t idx = c * qt->channel_size + i;
            int32_t q = (int32_t)roundf(input->data[idx] / qt->scales[c]);
            q = (q < -128) ? -128 : (q > 127 ? 127 : q);
            qt->data[idx] = (int8_t)q;
        }
    }
    
    return qt;
}

void per_channel_quantized_tensor_free(PerChannelQuantizedTensor* qt) {
    if (!qt) return;
    rpitorch_aligned_free(qt->data);
    free(qt->scales);
    free(qt->zero_points);
    free(qt);
}

// ============================================================
// Quantization-Aware Training (QAT)
// ============================================================

typedef struct {
    Tensor* scale;
    Tensor* zero_point;
    bool training;
    uint32_t num_batches_tracked;
} FakeQuantize;

FakeQuantize* fake_quantize_create() {
    FakeQuantize* fq = (FakeQuantize*)calloc(1, sizeof(FakeQuantize));
    
    uint32_t shape[1] = {1};
    fq->scale = tensor_create(1, shape, true);
    fq->zero_point = tensor_create(1, shape, true);
    
    fq->scale->data[0] = 1.0f;
    fq->zero_point->data[0] = 0.0f;
    fq->training = true;
    
    return fq;
}

Tensor* fake_quantize_forward(FakeQuantize* fq, const Tensor* input) {
    Tensor* output = tensor_create(input->dims, input->shape, input->requires_grad);
    
    float scale = fq->scale->data[0];
    float zp = fq->zero_point->data[0];
    
    if (fq->training) {
        // Update scale and zero point using EMA
        float min_val = FLT_MAX, max_val = -FLT_MAX;
        
        #pragma omp parallel for reduction(min:min_val) reduction(max:max_val)
        for (uint32_t i = 0; i < input->size; i++) {
            if (input->data[i] < min_val) min_val = input->data[i];
            if (input->data[i] > max_val) max_val = input->data[i];
        }
        
        float new_scale = (max_val - min_val) / 255.0f;
        float new_zp = -min_val / new_scale;
        
        float momentum = 0.1f;
        scale = (1.0f - momentum) * scale + momentum * new_scale;
        zp = (1.0f - momentum) * zp + momentum * new_zp;
        
        fq->scale->data[0] = scale;
        fq->zero_point->data[0] = zp;
        fq->num_batches_tracked++;
    }
    
    // Simulate quantization
    float inv_scale = 1.0f / scale;
    #pragma omp parallel for
    for (uint32_t i = 0; i < input->size; i++) {
        // Quantize
        int32_t q = (int32_t)roundf(input->data[i] * inv_scale + zp);
        q = (q < -128) ? -128 : (q > 127 ? 127 : q);
        
        // Dequantize
        output->data[i] = (q - zp) * scale;
    }
    
    // Straight-Through Estimator (STE) for Backward Pass:
    // Gradient flows through as if it was identity, but clipps if outside range [-128, 127] roughly.
    // For simplicity, we just attach identity backward or custom STE if needed.
    // Here we assume simple identity STE in rpl_core's autograd if we manually set parents.
    if (output->requires_grad) {
        output->parent1 = (void*)input;
        output->backward_fn = backward_fake_quant;
        output->is_leaf = false;
    }
    
    return output;
}

// STE Backward: gradient passes through 1.0 if inside range
void backward_fake_quant(Tensor* t) {
    Tensor* input = (Tensor*)t->parent1;
    if (input && input->grad) {
        #pragma omp parallel for
        for (uint32_t i = 0; i < t->size; i++) {
            input->grad[i] += t->grad[i]; // STE: Identity gradient
        }
        if (input->backward_fn) input->backward_fn(input);
    }
}

void fake_quantize_free(FakeQuantize* fq) {
    if (!fq) return;
    tensor_free(fq->scale);
    tensor_free(fq->zero_point);
    free(fq);
}

// ============================================================
// Model Quantization
// ============================================================

typedef struct {
    Tensor** fp32_params;
    QuantizedTensor** int8_params;
    uint32_t num_params;
    bool is_quantized;
} QuantizedModel;

QuantizedModel* model_quantize(Tensor** parameters, uint32_t num_params, 
                              bool per_channel) {
    QuantizedModel* qm = (QuantizedModel*)calloc(1, sizeof(QuantizedModel));
    qm->num_params = num_params;
    qm->fp32_params = parameters;
    qm->int8_params = (QuantizedTensor**)calloc(num_params, sizeof(QuantizedTensor*));
    
    for (uint32_t i = 0; i < num_params; i++) {
        if (per_channel && parameters[i]->dims >= 2) {
            // Per-channel quantization for conv/linear weights
            // For now, use per-tensor
            qm->int8_params[i] = tensor_quantize_int8_symmetric(parameters[i]);
        } else {
            qm->int8_params[i] = tensor_quantize_int8_symmetric(parameters[i]);
        }
    }
    
    qm->is_quantized = true;
    return qm;
}

void quantized_model_free(QuantizedModel* qm) {
    if (!qm) return;
    
    for (uint32_t i = 0; i < qm->num_params; i++) {
        quantized_tensor_free(qm->int8_params[i]);
    }
    free(qm->int8_params);
    free(qm);
}

// ============================================================
// Quantized Operations
// ============================================================

// Quantized Linear Layer
void linear_int8(const QuantizedTensor* input, const QuantizedTensor* weight,
                const QuantizedTensor* bias, int32_t* output,
                uint32_t batch_size, uint32_t in_features, uint32_t out_features) {
    
    #pragma omp parallel for
    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t o = 0; o < out_features; o++) {
            int32_t sum = 0;
            
            #if RPITORCH_HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
            uint32_t i = 0;
            int32x4_t vsum = vdupq_n_s32(0);
            
            for (; i + 16 <= in_features; i += 16) {
                int8x16_t vin = vld1q_s8(&input->data[b * in_features + i]);
                int8x16_t vw = vld1q_s8(&weight->data[o * in_features + i]);
                vsum = vdotq_s32(vsum, vin, vw);
            }
            
            sum = vaddvq_s32(vsum);
            
            for (; i < in_features; i++) {
                sum += input->data[b * in_features + i] * 
                      weight->data[o * in_features + i];
            }
            #else
            for (uint32_t i = 0; i < in_features; i++) {
                sum += input->data[b * in_features + i] * 
                      weight->data[o * in_features + i];
            }
            #endif
            
            if (bias) {
                sum += bias->data[o];
            }
            
            output[b * out_features + o] = sum;
        }
    }
}

/*
 * RPL Neural Network Layers
 * Core NN building blocks
 */

#include "rpl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// ============================================================
// Linear Layer
// ============================================================

struct Linear {
    Tensor* weight;  // [out_features, in_features]
    Tensor* bias;    // [out_features]
    uint32_t in_features;
    uint32_t out_features;
};

Linear* linear_create(uint32_t in_features, uint32_t out_features) {
    Linear* layer = (Linear*)calloc(1, sizeof(Linear));
    
    layer->in_features = in_features;
    layer->out_features = out_features;
    
    uint32_t weight_shape[2] = {out_features, in_features};
    layer->weight = tensor_create(2, weight_shape, true);
    
    uint32_t bias_shape[1] = {out_features};
    layer->bias = tensor_create(1, bias_shape, true);
    
    // Xavier initialization
    float std = sqrtf(2.0f / (in_features + out_features));
    for (uint32_t i = 0; i < layer->weight->size; i++) {
        layer->weight->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std;
    }
    tensor_fill(layer->bias, 0.0f);
    
    return layer;
}

void linear_backward(Tensor* t) {
    // Linear layer passed its output 't'
    // t->parent1 is input, t->parent2 is Linear* layer (Wait, I need weight and bias)
    // This is tricky because parent1/parent2 are Tensors.
    // I'll make a more standard way.
}

// Actually, I'll just use a simpler approach in rpl_core.c 
// by making backward_matmul handle the weight orientation of Linear.

Tensor* linear_forward(Linear* layer, const Tensor* input) {
    // 1. Matmul: x * W^T
    uint32_t output_shape[2] = {input->shape[0], layer->out_features};
    Tensor* matmul_out = tensor_create(2, output_shape, input->requires_grad || layer->weight->requires_grad);
    tensor_fill(matmul_out, 0.0f);
    tensor_gemm(matmul_out, input, layer->weight, 1.0f, 0.0f, false, true);
    
    if (matmul_out->requires_grad) {
        matmul_out->parent1 = (void*)input;
        matmul_out->parent2 = (void*)layer->weight;
        matmul_out->backward_fn = backward_matmul;
        matmul_out->is_leaf = false;
    }
    
    // 2. Add bias
    Tensor* out = tensor_create(2, output_shape, matmul_out->requires_grad || layer->bias->requires_grad);
    tensor_add_out(out, matmul_out, layer->bias);
    
    // Note: In a real system, we'd manage the memory of matmul_out.
    // Here, it's part of the autograd graph, so it must stay alive until backward.
    
    return out;
}

void linear_free(Linear* layer) {
    tensor_free(layer->weight);
    tensor_free(layer->bias);
    free(layer);
}

// ============================================================
// Conv2D Layer
// ============================================================

struct Conv2dLayer {
    Tensor* weight;
    Tensor* bias;
    uint32_t in_channels;
    uint32_t out_channels;
    uint32_t kernel_size;
    uint32_t stride;
    uint32_t padding;
    uint32_t dilation;
    uint32_t groups;
};

Conv2dLayer* conv2d_create(uint32_t in_channels, uint32_t out_channels,
                           uint32_t kernel_size, uint32_t stride, uint32_t padding) {
    Conv2dLayer* layer = (Conv2dLayer*)calloc(1, sizeof(Conv2dLayer));
    
    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding = padding;
    layer->dilation = 1;
    layer->groups = 1;
    
    uint32_t weight_shape[4] = {out_channels, in_channels, kernel_size, kernel_size};
    layer->weight = tensor_create(4, weight_shape, true);
    
    uint32_t bias_shape[1] = {out_channels};
    layer->bias = tensor_create(1, bias_shape, true);
    
    float std = sqrtf(2.0f / (in_channels * kernel_size * kernel_size));
    for (uint32_t i = 0; i < layer->weight->size; i++) {
        layer->weight->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std;
    }
    tensor_fill(layer->bias, 0.0f);
    
    return layer;
}

Tensor* conv2d_forward(Conv2dLayer* layer, const Tensor* input) {
    // Simplified conv2d - assumes implementation exists in core
    // input: [batch, in_channels, height, width]
    uint32_t batch = input->shape[0];
    uint32_t in_h = input->shape[2];
    uint32_t in_w = input->shape[3];
    
    uint32_t out_h = (in_h + 2 * layer->padding - layer->kernel_size) / layer->stride + 1;
    uint32_t out_w = (in_w + 2 * layer->padding - layer->kernel_size) / layer->stride + 1;
    
    uint32_t output_shape[4] = {batch, layer->out_channels, out_h, out_w};
    Tensor* output = tensor_create(4, output_shape, input->requires_grad);
    tensor_fill(output, 0.0f);
    
    // Simplified convolution implementation
    #pragma omp parallel for collapse(4)
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t oc = 0; oc < layer->out_channels; oc++) {
            for (uint32_t oh = 0; oh < out_h; oh++) {
                for (uint32_t ow = 0; ow < out_w; ow++) {
                    float sum = layer->bias->data[oc];
                    
                    for (uint32_t ic = 0; ic < layer->in_channels; ic++) {
                        for (uint32_t kh = 0; kh < layer->kernel_size; kh++) {
                            for (uint32_t kw = 0; kw < layer->kernel_size; kw++) {
                                int32_t ih = oh * layer->stride + kh - layer->padding;
                                int32_t iw = ow * layer->stride + kw - layer->padding;
                                
                                if (ih >= 0 && ih < (int32_t)in_h && iw >= 0 && iw < (int32_t)in_w) {
                                    sum += input->data[((b * layer->in_channels + ic) * in_h + ih) * in_w + iw] *
                                          layer->weight->data[((oc * layer->in_channels + ic) * layer->kernel_size + kh) * 
                                                             layer->kernel_size + kw];
                                }
                            }
                        }
                    }
                    
                    output->data[((b * layer->out_channels + oc) * out_h + oh) * out_w + ow] = sum;
                }
            }
        }
    }
    
    return output;
}

void conv2d_free(Conv2dLayer* layer) {
    tensor_free(layer->weight);
    tensor_free(layer->bias);
    free(layer);
}

// ============================================================
// BatchNorm2D
// ============================================================

struct BatchNorm2dLayer {
    uint32_t num_features;
    Tensor* weight;
    Tensor* bias;
    Tensor* running_mean;
    Tensor* running_var;
    float momentum;
    float eps;
    bool training;
    uint32_t num_batches_tracked;
};

BatchNorm2dLayer* batchnorm2d_create(uint32_t num_features, float momentum, float eps) {
    BatchNorm2dLayer* layer = (BatchNorm2dLayer*)calloc(1, sizeof(BatchNorm2dLayer));
    
    layer->num_features = num_features;
    layer->momentum = momentum;
    layer->eps = eps;
    layer->training = true;
    
    uint32_t shape[1] = {num_features};
    layer->weight = tensor_create(1, shape, true);
    layer->bias = tensor_create(1, shape, true);
    layer->running_mean = tensor_create(1, shape, false);
    layer->running_var = tensor_create(1, shape, false);
    
    tensor_fill(layer->weight, 1.0f);
    tensor_fill(layer->bias, 0.0f);
    tensor_fill(layer->running_mean, 0.0f);
    tensor_fill(layer->running_var, 1.0f);
    
    return layer;
}

Tensor* batchnorm2d_forward(BatchNorm2dLayer* layer, const Tensor* input) {
    Tensor* output = tensor_create(input->dims, input->shape, input->requires_grad);
    
    tensor_batchnorm2d(output, input,
                      layer->weight->data, layer->bias->data,
                      layer->running_mean->data, layer->running_var->data,
                      layer->eps, layer->training, layer->momentum);
    
    if (layer->training) {
        layer->num_batches_tracked++;
    }
    
    return output;
}

void batchnorm2d_free(BatchNorm2dLayer* layer) {
    tensor_free(layer->weight);
    tensor_free(layer->bias);
    tensor_free(layer->running_mean);
    tensor_free(layer->running_var);
    free(layer);
}

// ============================================================
// LayerNorm
// ============================================================

struct LayerNormLayer {
    uint32_t normalized_shape;
    Tensor* weight;
    Tensor* bias;
    float eps;
};

LayerNormLayer* layer_norm_create(uint32_t normalized_shape, float eps) {
    LayerNormLayer* layer = (LayerNormLayer*)calloc(1, sizeof(LayerNormLayer));
    
    layer->normalized_shape = normalized_shape;
    layer->eps = eps;
    
    uint32_t shape[1] = {normalized_shape};
    layer->weight = tensor_create(1, shape, true);
    layer->bias = tensor_create(1, shape, true);
    
    tensor_fill(layer->weight, 1.0f);
    tensor_fill(layer->bias, 0.0f);
    
    return layer;
}

Tensor* layer_norm_forward(LayerNormLayer* layer, const Tensor* input) {
    Tensor* output = tensor_create(input->dims, input->shape, input->requires_grad);
    
    uint32_t batch_size = input->size / layer->normalized_shape;
    
    #pragma omp parallel for
    for (uint32_t b = 0; b < batch_size; b++) {
        float* data = &input->data[b * layer->normalized_shape];
        float* out = &output->data[b * layer->normalized_shape];
        
        float mean = 0.0f;
        for (uint32_t i = 0; i < layer->normalized_shape; i++) {
            mean += data[i];
        }
        mean /= layer->normalized_shape;
        
        float var = 0.0f;
        for (uint32_t i = 0; i < layer->normalized_shape; i++) {
            float diff = data[i] - mean;
            var += diff * diff;
        }
        var /= layer->normalized_shape;
        
        float inv_std = 1.0f / sqrtf(var + layer->eps);
        for (uint32_t i = 0; i < layer->normalized_shape; i++) {
            out[i] = (data[i] - mean) * inv_std * layer->weight->data[i] + layer->bias->data[i];
        }
    }
    
    return output;
}

void layer_norm_free(LayerNormLayer* layer) {
    tensor_free(layer->weight);
    tensor_free(layer->bias);
    free(layer);
}

// ============================================================
// Embedding
// ============================================================

struct EmbeddingLayer {
    uint32_t num_embeddings;
    uint32_t embedding_dim;
    Tensor* weight;  // [num_embeddings, embedding_dim]
};

EmbeddingLayer* embedding_create(uint32_t num_embeddings, uint32_t embedding_dim) {
    EmbeddingLayer* layer = (EmbeddingLayer*)calloc(1, sizeof(EmbeddingLayer));
    
    layer->num_embeddings = num_embeddings;
    layer->embedding_dim = embedding_dim;
    
    uint32_t weight_shape[2] = {num_embeddings, embedding_dim};
    layer->weight = tensor_create(2, weight_shape, true);
    
    // Random initialization
    for (uint32_t i = 0; i < layer->weight->size; i++) {
        layer->weight->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    return layer;
}

Tensor* embedding_forward(EmbeddingLayer* layer, const uint32_t* indices, uint32_t num_indices) {
    uint32_t output_shape[2] = {num_indices, layer->embedding_dim};
    Tensor* output = tensor_create(2, output_shape, true);
    
    for (uint32_t i = 0; i < num_indices; i++) {
        uint32_t idx = indices[i];
        if (idx < layer->num_embeddings) {
            memcpy(&output->data[i * layer->embedding_dim],
                  &layer->weight->data[idx * layer->embedding_dim],
                  layer->embedding_dim * sizeof(float));
        }
    }
    
    return output;
}

void embedding_free(EmbeddingLayer* layer) {
    tensor_free(layer->weight);
    free(layer);
}

// ============================================================
// Dropout
// ============================================================

struct DropoutLayer {
    float p;
    bool training;
};

DropoutLayer* dropout_create(float p) {
    DropoutLayer* layer = (DropoutLayer*)calloc(1, sizeof(DropoutLayer));
    layer->p = p;
    layer->training = true;
    return layer;
}

Tensor* dropout_forward(DropoutLayer* layer, const Tensor* input) {
    Tensor* output = tensor_create(input->dims, input->shape, input->requires_grad);
    tensor_dropout(output, input, layer->p, layer->training);
    return output;
}

void dropout_free(DropoutLayer* layer) {
    free(layer);
}

// ============================================================
// MaxPool2D
// ============================================================

struct MaxPool2dLayer {
    uint32_t kernel_size;
    uint32_t stride;
    uint32_t padding;
};

MaxPool2dLayer* maxpool2d_create(uint32_t kernel_size, uint32_t stride, uint32_t padding) {
    MaxPool2dLayer* layer = (MaxPool2dLayer*)calloc(1, sizeof(MaxPool2dLayer));
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding = padding;
    return layer;
}

Tensor* maxpool2d_forward(MaxPool2dLayer* layer, const Tensor* input) {
    uint32_t batch = input->shape[0];
    uint32_t channels = input->shape[1];
    uint32_t in_h = input->shape[2];
    uint32_t in_w = input->shape[3];
    
    uint32_t out_h = (in_h + 2 * layer->padding - layer->kernel_size) / layer->stride + 1;
    uint32_t out_w = (in_w + 2 * layer->padding - layer->kernel_size) / layer->stride + 1;
    
    uint32_t output_shape[4] = {batch, channels, out_h, out_w};
    Tensor* output = tensor_create(4, output_shape, input->requires_grad);
    
    #pragma omp parallel for collapse(4)
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t c = 0; c < channels; c++) {
            for (uint32_t oh = 0; oh < out_h; oh++) {
                for (uint32_t ow = 0; ow < out_w; ow++) {
                    float max_val = -FLT_MAX;
                    
                    for (uint32_t kh = 0; kh < layer->kernel_size; kh++) {
                        for (uint32_t kw = 0; kw < layer->kernel_size; kw++) {
                            int32_t ih = oh * layer->stride + kh - layer->padding;
                            int32_t iw = ow * layer->stride + kw - layer->padding;
                            
                            if (ih >= 0 && ih < (int32_t)in_h && iw >= 0 && iw < (int32_t)in_w) {
                                float val = input->data[((b * channels + c) * in_h + ih) * in_w + iw];
                                if (val > max_val) max_val = val;
                            }
                        }
                    }
                    output->data[((b * channels + c) * out_h + oh) * out_w + ow] = max_val;
                }
            }
        }
    }
    
    return output;
}

void maxpool2d_free(MaxPool2dLayer* layer) {
    free(layer);
}

Tensor* tensor_conv2d(const Tensor* input, const Tensor* kernel, uint32_t stride, uint32_t padding) {
    uint32_t batch = input->shape[0];
    uint32_t out_channels = kernel->shape[0];
    uint32_t in_channels = kernel->shape[1];
    uint32_t kh = kernel->shape[2];
    uint32_t kw = kernel->shape[3];
    uint32_t in_h = input->shape[2];
    uint32_t in_w = input->shape[3];

    uint32_t out_h = (in_h + 2 * padding - kh) / stride + 1;
    uint32_t out_w = (in_w + 2 * padding - kw) / stride + 1;

    uint32_t output_shape[4] = {batch, out_channels, out_h, out_w};
    Tensor* output = tensor_create(4, output_shape, input->requires_grad || kernel->requires_grad);

    #pragma omp parallel for collapse(4)
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t oc = 0; oc < out_channels; oc++) {
            for (uint32_t oh = 0; oh < out_h; oh++) {
                for (uint32_t ow = 0; ow < out_w; ow++) {
                    float sum = 0.0f;
                    for (uint32_t ic = 0; ic < in_channels; ic++) {
                        for (uint32_t k_h = 0; k_h < kh; k_h++) {
                            for (uint32_t k_w = 0; k_w < kw; k_w++) {
                                int32_t ih = oh * stride + k_h - padding;
                                int32_t iw = ow * stride + k_w - padding;
                                if (ih >= 0 && ih < (int32_t)in_h && iw >= 0 && iw < (int32_t)in_w) {
                                    sum += input->data[((b * in_channels + ic) * in_h + ih) * in_w + iw] *
                                           kernel->data[((oc * in_channels + ic) * kh + k_h) * kw + k_w];
                                }
                            }
                        }
                    }
                    output->data[((b * out_channels + oc) * out_h + oh) * out_w + ow] = sum;
                }
            }
        }
    }
    return output;
}

Tensor* tensor_maxpool2d(const Tensor* input, uint32_t kernel_size, uint32_t stride) {
    uint32_t batch = input->shape[0];
    uint32_t channels = input->shape[1];
    uint32_t in_h = input->shape[2];
    uint32_t in_w = input->shape[3];

    uint32_t out_h = (in_h - kernel_size) / stride + 1;
    uint32_t out_w = (in_w - kernel_size) / stride + 1;

    uint32_t output_shape[4] = {batch, channels, out_h, out_w};
    Tensor* output = tensor_create(4, output_shape, input->requires_grad);

    #pragma omp parallel for collapse(4)
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t c = 0; c < channels; c++) {
            for (uint32_t oh = 0; oh < out_h; oh++) {
                for (uint32_t ow = 0; ow < out_w; ow++) {
                    float max_val = -FLT_MAX;
                    for (uint32_t kh = 0; kh < kernel_size; kh++) {
                        for (uint32_t kw = 0; kw < kernel_size; kw++) {
                            uint32_t ih = oh * stride + kh;
                            uint32_t iw = ow * stride + kw;
                            float val = input->data[((b * channels + c) * in_h + ih) * in_w + iw];
                            if (val > max_val) max_val = val;
                        }
                    }
                    output->data[((b * channels + c) * out_h + oh) * out_w + ow] = max_val;
                }
            }
        }
    }
    return output;
}

// ============================================================
// Loss Functions
// ============================================================

float mse_loss(const Tensor* pred, const Tensor* target) {
    float loss = 0.0f;
    
    #pragma omp parallel for reduction(+:loss)
    for (uint32_t i = 0; i < pred->size; i++) {
        float diff = pred->data[i] - target->data[i];
        loss += diff * diff;
    }
    
    return loss / pred->size;
}

float cross_entropy_loss(const Tensor* pred, const Tensor* target) {
    float loss = 0.0f;
    uint32_t batch_size = pred->shape[0];
    uint32_t num_classes = pred->shape[1];
    
    #pragma omp parallel for reduction(+:loss)
    for (uint32_t b = 0; b < batch_size; b++) {
        float max_val = -FLT_MAX;
        for (uint32_t c = 0; c < num_classes; c++) {
            if (pred->data[b * num_classes + c] > max_val) {
                max_val = pred->data[b * num_classes + c];
            }
        }
        
        float sum_exp = 0.0f;
        for (uint32_t c = 0; c < num_classes; c++) {
            sum_exp += expf(pred->data[b * num_classes + c] - max_val);
        }
        
        for (uint32_t c = 0; c < num_classes; c++) {
            if (target->data[b * num_classes + c] > 0.5f) {
                float log_prob = (pred->data[b * num_classes + c] - max_val) - logf(sum_exp);
                loss -= log_prob;
            }
        }
    }
    
    return loss / batch_size;
}

float binary_cross_entropy_loss(const Tensor* pred, const Tensor* target) {
    float loss = 0.0f;
    
    #pragma omp parallel for reduction(+:loss)
    for (uint32_t i = 0; i < pred->size; i++) {
        float p = fmaxf(fminf(pred->data[i], 1.0f - 1e-7f), 1e-7f);
        loss += -(target->data[i] * logf(p) + (1.0f - target->data[i]) * logf(1.0f - p));
    }
    
    return loss / pred->size;
}

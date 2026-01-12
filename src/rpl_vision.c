/*
 * RPiTorch Vision-Specific Operations
 * Conv3D, Depthwise Separable Conv, NMS, ROI Pooling, Patch Embedding
 */

#include "rpl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// ============================================================
// Conv3D - 3D Convolution
// ============================================================

struct Conv3dLayer {
    Tensor* weight;  // [out_channels, in_channels, depth, height, width]
    Tensor* bias;
    uint32_t in_channels;
    uint32_t out_channels;
    uint32_t kernel_size[3];  // depth, height, width
    uint32_t stride[3];
    uint32_t padding[3];
};

Conv3dLayer* conv3d_create(uint32_t in_channels, uint32_t out_channels,
                           uint32_t kernel_d, uint32_t kernel_h, uint32_t kernel_w,
                           uint32_t stride_d, uint32_t stride_h, uint32_t stride_w,
                           uint32_t padding_d, uint32_t padding_h, uint32_t padding_w) {
    Conv3dLayer* layer = (Conv3dLayer*)calloc(1, sizeof(Conv3dLayer));
    
    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->kernel_size[0] = kernel_d;
    layer->kernel_size[1] = kernel_h;
    layer->kernel_size[2] = kernel_w;
    layer->stride[0] = stride_d;
    layer->stride[1] = stride_h;
    layer->stride[2] = stride_w;
    layer->padding[0] = padding_d;
    layer->padding[1] = padding_h;
    layer->padding[2] = padding_w;
    
    uint32_t weight_shape[5] = {out_channels, in_channels, kernel_d, kernel_h, kernel_w};
    layer->weight = tensor_create(5, weight_shape, true);
    
    uint32_t bias_shape[1] = {out_channels};
    layer->bias = tensor_create(1, bias_shape, true);
    
    // Kaiming initialization
    float std = sqrtf(2.0f / (in_channels * kernel_d * kernel_h * kernel_w));
    for (uint32_t i = 0; i < layer->weight->size; i++) {
        layer->weight->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std;
    }
    tensor_fill(layer->bias, 0.0f);
    
    return layer;
}

Tensor* conv3d_forward(Conv3dLayer* layer, const Tensor* input) {
    // input: [batch, in_channels, depth, height, width]
    uint32_t batch = input->shape[0];
    uint32_t in_d = input->shape[2];
    uint32_t in_h = input->shape[3];
    uint32_t in_w = input->shape[4];
    
    uint32_t out_d = (in_d + 2 * layer->padding[0] - layer->kernel_size[0]) / layer->stride[0] + 1;
    uint32_t out_h = (in_h + 2 * layer->padding[1] - layer->kernel_size[1]) / layer->stride[1] + 1;
    uint32_t out_w = (in_w + 2 * layer->padding[2] - layer->kernel_size[2]) / layer->stride[2] + 1;
    
    uint32_t output_shape[5] = {batch, layer->out_channels, out_d, out_h, out_w};
    Tensor* output = tensor_create(5, output_shape, input->requires_grad);
    
    #pragma omp parallel for collapse(4)
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t oc = 0; oc < layer->out_channels; oc++) {
            for (uint32_t od = 0; od < out_d; od++) {
                for (uint32_t oh = 0; oh < out_h; oh++) {
                    for (uint32_t ow = 0; ow < out_w; ow++) {
                        float sum = layer->bias->data[oc];
                        
                        for (uint32_t ic = 0; ic < layer->in_channels; ic++) {
                            for (uint32_t kd = 0; kd < layer->kernel_size[0]; kd++) {
                                for (uint32_t kh = 0; kh < layer->kernel_size[1]; kh++) {
                                    for (uint32_t kw = 0; kw < layer->kernel_size[2]; kw++) {
                                        int32_t id = od * layer->stride[0] + kd - layer->padding[0];
                                        int32_t ih = oh * layer->stride[1] + kh - layer->padding[1];
                                        int32_t iw = ow * layer->stride[2] + kw - layer->padding[2];
                                        
                                        if (id >= 0 && id < (int32_t)in_d &&
                                            ih >= 0 && ih < (int32_t)in_h &&
                                            iw >= 0 && iw < (int32_t)in_w) {
                                            
                                            uint32_t input_idx = ((((b * layer->in_channels + ic) * in_d + id) * 
                                                                  in_h + ih) * in_w + iw);
                                            uint32_t weight_idx = ((((oc * layer->in_channels + ic) * 
                                                                    layer->kernel_size[0] + kd) * 
                                                                   layer->kernel_size[1] + kh) * 
                                                                  layer->kernel_size[2] + kw);
                                            
                                            sum += input->data[input_idx] * layer->weight->data[weight_idx];
                                        }
                                    }
                                }
                            }
                        }
                        
                        uint32_t output_idx = ((((b * layer->out_channels + oc) * out_d + od) * 
                                               out_h + oh) * out_w + ow);
                        output->data[output_idx] = sum;
                    }
                }
            }
        }
    }
    
    return output;
}

void conv3d_free(Conv3dLayer* layer) {
    tensor_free(layer->weight);
    tensor_free(layer->bias);
    free(layer);
}

// ============================================================
// ResBlock - Residual Block
// ============================================================

struct ResBlock {
    Conv2dLayer* conv1;
    Conv2dLayer* conv2;
    BatchNorm2dLayer* bn1;
    BatchNorm2dLayer* bn2;
};

ResBlock* res_block_create(uint32_t channels) {
    ResBlock* block = (ResBlock*)calloc(1, sizeof(ResBlock));
    block->conv1 = conv2d_create(channels, channels, 3, 1, 1);
    block->conv2 = conv2d_create(channels, channels, 3, 1, 1);
    block->bn1 = batchnorm2d_create(channels, 0.1, 1e-5);
    block->bn2 = batchnorm2d_create(channels, 0.1, 1e-5);
    return block;
}

Tensor* res_block_forward(ResBlock* block, const Tensor* input) {
    Tensor* x = conv2d_forward(block->conv1, input);
    Tensor* x_bn = batchnorm2d_forward(block->bn1, x);
    tensor_relu_inplace(x_bn);
    
    Tensor* x2 = conv2d_forward(block->conv2, x_bn);
    Tensor* x2_bn = batchnorm2d_forward(block->bn2, x2);
    
    // Residual connection: x2_bn + input
    tensor_add_inplace(x2_bn, input);
    tensor_relu_inplace(x2_bn);
    
    tensor_free(x);
    tensor_free(x_bn);
    tensor_free(x2);
    
    return x2_bn;
}

void res_block_free(ResBlock* block) {
    conv2d_free(block->conv1);
    conv2d_free(block->conv2);
    batchnorm2d_free(block->bn1);
    batchnorm2d_free(block->bn2);
    free(block);
}


// ============================================================
// Depthwise Separable Convolution
// ============================================================

struct DepthwiseSeparableConv {
    Conv2dLayer* depthwise;   // Depthwise conv (groups = channels)
    Conv2dLayer* pointwise;   // 1×1 conv
};

DepthwiseSeparableConv* depthwise_separable_conv_create(uint32_t in_channels, uint32_t out_channels,
                                                         uint32_t kernel_size, uint32_t stride, uint32_t padding) {
    DepthwiseSeparableConv* layer = (DepthwiseSeparableConv*)calloc(1, sizeof(DepthwiseSeparableConv));
    
    // Depthwise: each input channel convolved separately
    layer->depthwise = conv2d_create(in_channels, in_channels, kernel_size, stride, padding);
    
    // Pointwise: 1×1 conv to combine channels
    layer->pointwise = conv2d_create(in_channels, out_channels, 1, 1, 0);
    
    return layer;
}

Tensor* depthwise_separable_conv_forward(DepthwiseSeparableConv* layer, const Tensor* input) {
    // Depthwise convolution
    Tensor* depthwise_out = conv2d_forward(layer->depthwise, input);
    
    // Pointwise convolution
    Tensor* output = conv2d_forward(layer->pointwise, depthwise_out);
    
    tensor_free(depthwise_out);
    return output;
}

// ============================================================
// Non-Maximum Suppression (NMS)
// ============================================================

float iou_boxes(const float* box1, const float* box2) {
    // box format: [x1, y1, x2, y2]
    float x1 = fmaxf(box1[0], box2[0]);
    float y1 = fmaxf(box1[1], box2[1]);
    float x2 = fminf(box1[2], box2[2]);
    float y2 = fminf(box1[3], box2[3]);
    
    float intersection = fmaxf(0.0f, x2 - x1) * fmaxf(0.0f, y2 - y1);
    
    float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    float union_area = area1 + area2 - intersection;
    
    return (union_area > 0.0f) ? (intersection / union_area) : 0.0f;
}

void nms(const float* boxes, const float* scores, uint32_t n_boxes,
         float iou_threshold, uint32_t* keep_indices, uint32_t* n_keep) {
    // Sort by scores (descending)
    uint32_t* indices = (uint32_t*)malloc(n_boxes * sizeof(uint32_t));
    for (uint32_t i = 0; i < n_boxes; i++) {
        indices[i] = i;
    }
    
    // Simple bubble sort by score
    for (uint32_t i = 0; i < n_boxes - 1; i++) {
        for (uint32_t j = i + 1; j < n_boxes; j++) {
            if (scores[indices[j]] > scores[indices[i]]) {
                uint32_t temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
    }
    
    bool* suppressed = (bool*)calloc(n_boxes, sizeof(bool));
    *n_keep = 0;
    
    for (uint32_t i = 0; i < n_boxes; i++) {
        uint32_t idx = indices[i];
        
        if (suppressed[idx]) continue;
        
        keep_indices[(*n_keep)++] = idx;
        
        // Suppress overlapping boxes
        for (uint32_t j = i + 1; j < n_boxes; j++) {
            uint32_t idx2 = indices[j];
            
            if (suppressed[idx2]) continue;
            
            float iou = iou_boxes(&boxes[idx * 4], &boxes[idx2 * 4]);
            
            if (iou > iou_threshold) {
                suppressed[idx2] = true;
            }
        }
    }
    
    free(indices);
    free(suppressed);
}

// ============================================================
// Patch Embedding (for Vision Transformers)
// ============================================================

struct PatchEmbedding {
    uint32_t patch_size;
    uint32_t embed_dim;
    Conv2dLayer* projection;  // Conv with kernel_size=patch_size, stride=patch_size
};

PatchEmbedding* patch_embedding_create(uint32_t img_channels, uint32_t patch_size, uint32_t embed_dim) {
    PatchEmbedding* layer = (PatchEmbedding*)calloc(1, sizeof(PatchEmbedding));
    
    layer->patch_size = patch_size;
    layer->embed_dim = embed_dim;
    
    // Use convolution to extract patches and project
    layer->projection = conv2d_create(img_channels, embed_dim, patch_size, patch_size, 0);
    
    return layer;
}

Tensor* patch_embedding_forward(PatchEmbedding* layer, const Tensor* input) {
    // input: [batch, channels, height, width]
    // output: [batch, num_patches, embed_dim]
    
    Tensor* patches = conv2d_forward(layer->projection, input);
    // patches: [batch, embed_dim, H/patch_size, W/patch_size]
    
    uint32_t batch = patches->shape[0];
    uint32_t embed_dim = patches->shape[1];
    uint32_t h = patches->shape[2];
    uint32_t w = patches->shape[3];
    uint32_t num_patches = h * w;
    
    // Reshape to [batch, num_patches, embed_dim]
    uint32_t output_shape[3] = {batch, num_patches, embed_dim};
    Tensor* output = tensor_create(3, output_shape, input->requires_grad);
    
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t p = 0; p < num_patches; p++) {
            uint32_t ph = p / w;
            uint32_t pw = p % w;
            
            for (uint32_t e = 0; e < embed_dim; e++) {
                output->data[(b * num_patches + p) * embed_dim + e] = 
                    patches->data[((b * embed_dim + e) * h + ph) * w + pw];
            }
        }
    }
    
    tensor_free(patches);
    return output;
}

// ============================================================
// Squeeze-and-Excitation Block
// ============================================================

struct SEBlock {
    uint32_t channels;
    uint32_t reduction;
    Tensor* fc1_weight;
    Tensor* fc1_bias;
    Tensor* fc2_weight;
    Tensor* fc2_bias;
};

SEBlock* se_block_create(uint32_t channels, uint32_t reduction) {
    SEBlock* block = (SEBlock*)calloc(1, sizeof(SEBlock));
    
    block->channels = channels;
    block->reduction = reduction;
    
    uint32_t reduced_channels = channels / reduction;
    
    uint32_t fc1_shape[2] = {reduced_channels, channels};
    uint32_t fc2_shape[2] = {channels, reduced_channels};
    uint32_t bias1_shape[1] = {reduced_channels};
    uint32_t bias2_shape[1] = {channels};
    
    block->fc1_weight = tensor_create(2, fc1_shape, true);
    block->fc1_bias = tensor_create(1, bias1_shape, true);
    block->fc2_weight = tensor_create(2, fc2_shape, true);
    block->fc2_bias = tensor_create(1, bias2_shape, true);
    
    // Xavier initialization
    float std1 = sqrtf(2.0f / (channels + reduced_channels));
    float std2 = sqrtf(2.0f / (reduced_channels + channels));
    
    for (uint32_t i = 0; i < block->fc1_weight->size; i++) {
        block->fc1_weight->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std1;
    }
    for (uint32_t i = 0; i < block->fc2_weight->size; i++) {
        block->fc2_weight->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std2;
    }
    
    tensor_fill(block->fc1_bias, 0.0f);
    tensor_fill(block->fc2_bias, 0.0f);
    
    return block;
}

Tensor* se_block_forward(SEBlock* block, const Tensor* input) {
    // input: [batch, channels, height, width]
    uint32_t batch = input->shape[0];
    uint32_t channels = input->shape[1];
    uint32_t h = input->shape[2];
    uint32_t w = input->shape[3];
    uint32_t spatial_size = h * w;
    
    // Global average pooling
    Tensor* squeeze = tensor_create(2, (uint32_t[]){batch, channels}, false);
    
    #pragma omp parallel for collapse(2)
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t c = 0; c < channels; c++) {
            float sum = 0.0f;
            for (uint32_t s = 0; s < spatial_size; s++) {
                sum += input->data[(b * channels + c) * spatial_size + s];
            }
            squeeze->data[b * channels + c] = sum / spatial_size;
        }
    }
    
    // FC1 + ReLU
    Tensor* fc1_out = tensor_matmul(squeeze, block->fc1_weight);
    tensor_add_inplace(fc1_out, block->fc1_bias);
    tensor_relu_inplace(fc1_out);
    
    // FC2 + Sigmoid
    Tensor* fc2_out = tensor_matmul(fc1_out, block->fc2_weight);
    tensor_add_inplace(fc2_out, block->fc2_bias);
    tensor_sigmoid_inplace(fc2_out);
    
    // Scale input
    Tensor* output = tensor_create(input->dims, input->shape, input->requires_grad);
    
    #pragma omp parallel for collapse(2)
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t c = 0; c < channels; c++) {
            float scale = fc2_out->data[b * channels + c];
            for (uint32_t s = 0; s < spatial_size; s++) {
                output->data[(b * channels + c) * spatial_size + s] = 
                    input->data[(b * channels + c) * spatial_size + s] * scale;
            }
        }
    }
    
    tensor_free(squeeze);
    tensor_free(fc1_out);
    tensor_free(fc2_out);
    
    return output;
}

// ============================================================
// Instance Normalization
// ============================================================

struct InstanceNorm {
    uint32_t num_features;
    Tensor* weight;
    Tensor* bias;
    float eps;
};

InstanceNorm* instance_norm_create(uint32_t num_features, float eps) {
    InstanceNorm* layer = (InstanceNorm*)calloc(1, sizeof(InstanceNorm));
    
    layer->num_features = num_features;
    layer->eps = eps;
    
    uint32_t shape[1] = {num_features};
    layer->weight = tensor_create(1, shape, true);
    layer->bias = tensor_create(1, shape, true);
    
    tensor_fill(layer->weight, 1.0f);
    tensor_fill(layer->bias, 0.0f);
    
    return layer;
}

Tensor* instance_norm_forward(InstanceNorm* layer, const Tensor* input) {
    // input: [batch, channels, height, width]
    // Normalize each instance independently
    
    Tensor* output = tensor_create(input->dims, input->shape, input->requires_grad);
    
    uint32_t batch = input->shape[0];
    uint32_t channels = input->shape[1];
    uint32_t spatial_size = input->shape[2] * input->shape[3];
    
    #pragma omp parallel for collapse(2)
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t c = 0; c < channels; c++) {
            float* data = (float*)&input->data[(b * channels + c) * spatial_size];
            float* out = &output->data[(b * channels + c) * spatial_size];
            
            // Compute mean
            float mean = 0.0f;
            for (uint32_t s = 0; s < spatial_size; s++) {
                mean += data[s];
            }
            mean /= spatial_size;
            
            // Compute variance
            float var = 0.0f;
            for (uint32_t s = 0; s < spatial_size; s++) {
                float diff = data[s] - mean;
                var += diff * diff;
            }
            var /= spatial_size;
            
            // Normalize and scale
            float inv_std = 1.0f / sqrtf(var + layer->eps);
            for (uint32_t s = 0; s < spatial_size; s++) {
                out[s] = (data[s] - mean) * inv_std * layer->weight->data[c] + layer->bias->data[c];
            }
        }
    }
    
    return output;
}

// ============================================================
// Dilated/Atrous Convolution
// ============================================================

struct DilatedConv2d {
    Tensor* weight;
    Tensor* bias;
    uint32_t in_channels;
    uint32_t out_channels;
    uint32_t kernel_size;
    uint32_t stride;
    uint32_t padding;
    uint32_t dilation;
};

DilatedConv2d* dilated_conv_create(uint32_t in_channels, uint32_t out_channels,
                                   uint32_t kernel_size, uint32_t stride,
                                   uint32_t padding, uint32_t dilation) {
    DilatedConv2d* layer = (DilatedConv2d*)calloc(1, sizeof(DilatedConv2d));
    
    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding = padding;
    layer->dilation = dilation;
    
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

Tensor* dilated_conv_forward(DilatedConv2d* layer, const Tensor* input) {
    uint32_t batch = input->shape[0];
    uint32_t in_h = input->shape[2];
    uint32_t in_w = input->shape[3];
    
    // Effective kernel size with dilation
    uint32_t eff_kernel = (layer->kernel_size - 1) * layer->dilation + 1;
    uint32_t out_h = (in_h + 2 * layer->padding - eff_kernel) / layer->stride + 1;
    uint32_t out_w = (in_w + 2 * layer->padding - eff_kernel) / layer->stride + 1;
    
    uint32_t output_shape[4] = {batch, layer->out_channels, out_h, out_w};
    Tensor* output = tensor_create(4, output_shape, input->requires_grad);
    
    #pragma omp parallel for collapse(4)
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t oc = 0; oc < layer->out_channels; oc++) {
            for (uint32_t oh = 0; oh < out_h; oh++) {
                for (uint32_t ow = 0; ow < out_w; ow++) {
                    float sum = layer->bias->data[oc];
                    
                    for (uint32_t ic = 0; ic < layer->in_channels; ic++) {
                        for (uint32_t kh = 0; kh < layer->kernel_size; kh++) {
                            for (uint32_t kw = 0; kw < layer->kernel_size; kw++) {
                                int32_t ih = oh * layer->stride + kh * layer->dilation - layer->padding;
                                int32_t iw = ow * layer->stride + kw * layer->dilation - layer->padding;
                                
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

// ============================================================
// Spatial Pyramid Pooling (SPP)
// ============================================================

struct SPPLayer {
    uint32_t* pool_sizes;  // e.g., [1, 2, 4]
    uint32_t num_levels;
};

SPPLayer* spp_create(uint32_t* pool_sizes, uint32_t num_levels) {
    SPPLayer* layer = (SPPLayer*)calloc(1, sizeof(SPPLayer));
    layer->num_levels = num_levels;
    layer->pool_sizes = (uint32_t*)malloc(num_levels * sizeof(uint32_t));
    memcpy(layer->pool_sizes, pool_sizes, num_levels * sizeof(uint32_t));
    return layer;
}

Tensor* spp_forward(SPPLayer* layer, const Tensor* input) {
    // input: [batch, channels, height, width]
    uint32_t batch = input->shape[0];
    uint32_t channels = input->shape[1];
    uint32_t h = input->shape[2];
    uint32_t w = input->shape[3];
    
    // Calculate total output size
    uint32_t total_bins = 0;
    for (uint32_t i = 0; i < layer->num_levels; i++) {
        total_bins += layer->pool_sizes[i] * layer->pool_sizes[i];
    }
    
    uint32_t output_shape[2] = {batch, channels * total_bins};
    Tensor* output = tensor_create(2, output_shape, input->requires_grad);
    
    #pragma omp parallel for collapse(2)
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t c = 0; c < channels; c++) {
            uint32_t out_offset = 0;
            
            for (uint32_t level = 0; level < layer->num_levels; level++) {
                uint32_t pool_size = layer->pool_sizes[level];
                uint32_t bin_h = h / pool_size;
                uint32_t bin_w = w / pool_size;
                
                for (uint32_t ph = 0; ph < pool_size; ph++) {
                    for (uint32_t pw = 0; pw < pool_size; pw++) {
                        float max_val = -FLT_MAX;
                        
                        uint32_t h_start = ph * bin_h;
                        uint32_t h_end = (ph == pool_size - 1) ? h : (ph + 1) * bin_h;
                        uint32_t w_start = pw * bin_w;
                        uint32_t w_end = (pw == pool_size - 1) ? w : (pw + 1) * bin_w;
                        
                        for (uint32_t ih = h_start; ih < h_end; ih++) {
                            for (uint32_t iw = w_start; iw < w_end; iw++) {
                                float val = input->data[((b * channels + c) * h + ih) * w + iw];
                                if (val > max_val) max_val = val;
                            }
                        }
                        
                        output->data[b * (channels * total_bins) + c * total_bins + out_offset] = max_val;
                        out_offset++;
                    }
                }
            }
        }
    }
    
    return output;
}

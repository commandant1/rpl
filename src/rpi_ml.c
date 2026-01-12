#include "rpi_ml.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

// --- Forward Declarations for Autograd ---
void backward_add(Tensor* t);
void backward_mul(Tensor* t);
void backward_matmul(Tensor* t);
void backward_relu(Tensor* t);

// --- Fast GEMM Kernel ---
void parallel_gemm(const float* A, const float* B, float* C, uint32_t M, uint32_t K, uint32_t N) {
    #pragma omp parallel for collapse(2)
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            float sum = 0;
            #if defined(__ARM_NEON)
            // NEON block
            uint32_t k = 0;
            float32x4_t vsum = vdupq_n_f32(0.0f);
            for (; k <= K - 4; k += 4) {
                float32x4_t va = vld1q_f32(&A[i * K + k]);
                float32x4_t vb = vld1q_f32(&B[k * N + j]); // Note: This is not optimal for row-major B
                vsum = vmlaq_f32(vsum, va, vb);
            }
            sum = vgetq_lane_f32(vsum, 0) + vgetq_lane_f32(vsum, 1) + vgetq_lane_f32(vsum, 2) + vgetq_lane_f32(vsum, 3);
            for (; k < K; k++) sum += A[i * K + k] * B[k * N + j];
            #else
            // Generic fallback
            for (uint32_t k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            #endif
            C[i * N + j] = sum;
        }
    }
}

// --- Tensor Engine Logic ---

Tensor* tensor_create(uint32_t dims, const uint32_t* shape, bool requires_grad) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->dims = dims;
    t->size = 1;
    for (uint32_t i = 0; i < dims; i++) {
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }
    t->strides[dims - 1] = 1;
    for (int i = (int)dims - 2; i >= 0; i--) {
        t->strides[i] = t->strides[i + 1] * t->shape[i + 1];
    }
    t->data = (float*)aligned_alloc(64, t->size * sizeof(float));
    t->grad = NULL;
    t->requires_grad = requires_grad;
    if (requires_grad) {
        t->grad = (float*)aligned_alloc(64, t->size * sizeof(float));
        memset(t->grad, 0, t->size * sizeof(float));
    }
    t->parent1 = NULL; t->parent2 = NULL; t->backward_fn = NULL;
    return t;
}

void tensor_zero_grad(Tensor* t) {
    if (t->grad) {
        memset(t->grad, 0, t->size * sizeof(float));
    }
}

Tensor* tensor_create_2d(uint32_t rows, uint32_t cols, bool requires_grad) {
    uint32_t shape[2] = {rows, cols};
    return tensor_create(2, shape, requires_grad);
}

void tensor_free(Tensor* t) {
    if (t) {
        if (t->data) free(t->data);
        if (t->grad) free(t->grad);
        free(t);
    }
}

void tensor_fill(Tensor* t, float value) {
    #pragma omp parallel for
    for (uint32_t i = 0; i < t->size; i++) t->data[i] = value;
}

void tensor_randomize(Tensor* t) {
    for (uint32_t i = 0; i < t->size; i++) {
        t->data[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
}

// --- Autograd Backwards ---

void backward_add(Tensor* t) {
    #pragma omp parallel for
    for (uint32_t i = 0; i < t->size; i++) {
        if (t->parent1->grad) t->parent1->grad[i] += t->grad[i];
        if (t->parent2->grad) t->parent2->grad[i] += t->grad[i];
    }
}

void backward_mul(Tensor* t) {
    #pragma omp parallel for
    for (uint32_t i = 0; i < t->size; i++) {
        if (t->parent1->grad) t->parent1->grad[i] += t->grad[i] * t->parent2->data[i];
        if (t->parent2->grad) t->parent2->grad[i] += t->grad[i] * t->parent1->data[i];
    }
}

void backward_matmul(Tensor* t) {
    uint32_t M = t->parent1->shape[0];
    uint32_t K = t->parent1->shape[1];
    uint32_t N = t->parent2->shape[1];
    if (t->parent1->grad) {
        for (uint32_t i = 0; i < M; i++) {
            for (uint32_t k = 0; k < K; k++) {
                for (uint32_t j = 0; j < N; j++) {
                    t->parent1->grad[i * K + k] += t->grad[i * N + j] * t->parent2->data[k * N + j];
                }
            }
        }
    }
    if (t->parent2->grad) {
        for (uint32_t k = 0; k < K; k++) {
            for (uint32_t j = 0; j < N; j++) {
                for (uint32_t i = 0; i < M; i++) {
                    t->parent2->grad[k * N + j] += t->parent1->data[i * K + k] * t->grad[i * N + j];
                }
            }
        }
    }
}

void backward_relu(Tensor* t) {
    #pragma omp parallel for
    for (uint32_t i = 0; i < t->size; i++) {
        if (t->parent1->grad) {
            t->parent1->grad[i] += (t->parent1->data[i] > 0) ? t->grad[i] : 0;
        }
    }
}

// --- Ops ---

Tensor* tensor_add(Tensor* a, Tensor* b) {
    Tensor* res = tensor_create(a->dims, a->shape, a->requires_grad || b->requires_grad);
    #pragma omp parallel for
    for (uint32_t i = 0; i < a->size; i++) res->data[i] = a->data[i] + b->data[i];
    if (res->requires_grad) { res->parent1 = a; res->parent2 = b; res->backward_fn = backward_add; }
    return res;
}

Tensor* tensor_mul(Tensor* a, Tensor* b) {
    Tensor* res = tensor_create(a->dims, a->shape, a->requires_grad || b->requires_grad);
    #pragma omp parallel for
    for (uint32_t i = 0; i < a->size; i++) res->data[i] = a->data[i] * b->data[i];
    if (res->requires_grad) { res->parent1 = a; res->parent2 = b; res->backward_fn = backward_mul; }
    return res;
}

Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    uint32_t shape[2] = {a->shape[0], b->shape[1]};
    Tensor* res = tensor_create(2, shape, a->requires_grad || b->requires_grad);
    parallel_gemm(a->data, b->data, res->data, a->shape[0], a->shape[1], b->shape[1]);
    if (res->requires_grad) { res->parent1 = a; res->parent2 = b; res->backward_fn = backward_matmul; }
    return res;
}

Tensor* tensor_relu(Tensor* t) {
    Tensor* res = tensor_create(t->dims, t->shape, t->requires_grad);
    #pragma omp parallel for
    for (uint32_t i = 0; i < t->size; i++) res->data[i] = t->data[i] > 0 ? t->data[i] : 0;
    if (res->requires_grad) { res->parent1 = t; res->backward_fn = backward_relu; }
    return res;
}

void tensor_backward(Tensor* t) {
    if (!t->requires_grad) return;
    if (!t->grad) {
        t->grad = (float*)aligned_alloc(64, t->size * sizeof(float));
        for (uint32_t i = 0; i < t->size; i++) t->grad[i] = 1.0f;
    }
    if (t->backward_fn) t->backward_fn(t);
    if (t->parent1) tensor_backward(t->parent1);
    if (t->parent2) tensor_backward(t->parent2);
}

void im2col(const float* data_im, uint32_t channels, uint32_t height, uint32_t width,
            uint32_t ksize, uint32_t stride, uint32_t pad, float* data_col) {
    uint32_t out_h = (height + 2 * pad - ksize) / stride + 1;
    uint32_t out_w = (width + 2 * pad - ksize) / stride + 1;
    uint32_t channel_size = height * width;
    #pragma omp parallel for collapse(3)
    for (uint32_t c = 0; c < channels; c++) {
        for (uint32_t h = 0; h < out_h; h++) {
            for (uint32_t w = 0; w < out_w; w++) {
                uint32_t col_row = h * out_w + w;
                for (uint32_t kh = 0; kh < ksize; kh++) {
                    for (uint32_t kw = 0; kw < ksize; kw++) {
                        int im_h = h * stride + kh - pad;
                        int im_w = w * stride + kw - pad;
                        uint32_t col_idx = ((c * ksize * ksize + kh * ksize + kw) * out_h * out_w) + col_row;
                        if (im_h >= 0 && im_h < height && im_w >= 0 && im_w < width) {
                            data_col[col_idx] = data_im[c * channel_size + im_h * width + im_w];
                        } else { data_col[col_idx] = 0; }
                    }
                }
            }
        }
    }
}

Tensor* tensor_conv2d(Tensor* input, Tensor* kernel, uint32_t stride, uint32_t padding) {
    uint32_t in_c = input->shape[0], in_h = input->shape[1], in_w = input->shape[2];
    uint32_t out_c = kernel->shape[0], kh = kernel->shape[2], kw = kernel->shape[3];
    uint32_t out_h = (in_h + 2 * padding - kh) / stride + 1, out_w = (in_w + 2 * padding - kw) / stride + 1;
    uint32_t col_rows = in_c * kh * kw, col_cols = out_h * out_w;
    float* data_col = (float*)aligned_alloc(64, col_rows * col_cols * sizeof(float));
    im2col(input->data, in_c, in_h, in_w, kh, stride, padding, data_col);
    uint32_t shape_out[3] = {out_c, out_h, out_w};
    Tensor* res = tensor_create(3, shape_out, input->requires_grad || kernel->requires_grad);
    parallel_gemm(kernel->data, data_col, res->data, out_c, col_rows, col_cols);
    free(data_col);
    return res;
}

Tensor* tensor_maxpool2d(Tensor* input, uint32_t kernel_size, uint32_t stride) {
    uint32_t in_c = input->shape[0], in_h = input->shape[1], in_w = input->shape[2];
    uint32_t out_h = (in_h - kernel_size) / stride + 1, out_w = (in_w - kernel_size) / stride + 1;
    uint32_t shape_out[3] = {in_c, out_h, out_w};
    Tensor* res = tensor_create(3, shape_out, input->requires_grad);
    #pragma omp parallel for collapse(3)
    for (uint32_t c = 0; c < in_c; c++) {
        for (uint32_t oh = 0; oh < out_h; oh++) {
            for (uint32_t ow = 0; ow < out_w; ow++) {
                float max_val = -INFINITY;
                for (uint32_t kh = 0; kh < kernel_size; kh++) {
                    for (uint32_t kw = 0; kw < kernel_size; kw++) {
                        float val = input->data[c * in_h * in_w + (oh * stride + kh) * in_w + (ow * stride + kw)];
                        if (val > max_val) max_val = val;
                    }
                }
                res->data[c * out_h * out_w + oh * out_w + ow] = max_val;
            }
        }
    }
    return res;
}

Tensor* tensor_sigmoid(Tensor* t) {
    Tensor* res = tensor_create(t->dims, t->shape, t->requires_grad);
    #pragma omp parallel for
    for (uint32_t i = 0; i < t->size; i++) res->data[i] = 1.0f / (1.0f + expf(-t->data[i]));
    return res;
}

Tensor* tensor_tanh(Tensor* t) {
    Tensor* res = tensor_create(t->dims, t->shape, t->requires_grad);
    #pragma omp parallel for
    for (uint32_t i = 0; i < t->size; i++) res->data[i] = tanhf(t->data[i]);
    return res;
}

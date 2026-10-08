#include "rpl.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EPSILON 1e-4

void assert_float_eq(float expected, float actual, const char* name) {
    if (fabsf(expected - actual) > EPSILON) {
        printf("[FAIL] %s: Expected %f, got %f\n", name, expected, actual);
        exit(1);
    }
}

void test_conv_transpose2d() {
    // 1-channel, 2x2 input
    uint32_t in_shape[] = {1, 1, 2, 2};
    Tensor* in = tensor_create(4, in_shape, false);
    for (uint32_t i = 0; i < 4; i++) in->data[i] = 1.0f;
    
    // Deconv to 4x4 (using kernel=2, stride=2, padding=0)
    ConvTranspose2dLayer* layer = conv_transpose2d_create(1, 1, 2, 2, 2, 2, 0, 0, 0, 0);
    tensor_fill(layer->weight, 2.0f);
    tensor_fill(layer->bias, 0.5f);
    
    Tensor* out = conv_transpose2d_forward(layer, in);
    
    // out shape should be [1, 1, 4, 4]
    if (out->shape[2] != 4 || out->shape[3] != 4) {
        printf("[FAIL] ConvTranspose2d shape mismatch\n");
        exit(1);
    }
    
    // value should be input * weight + bias = 1 * 2 + 0.5 = 2.5
    assert_float_eq(2.5f, out->data[0], "ConvTranspose2d data");
    
    printf("[PASS] ConvTranspose2d\n");
    tensor_free(in);
    tensor_free(out);
    conv_transpose2d_free(layer);
}

void test_group_norm() {
    uint32_t shape[] = {1, 4, 2, 2};
    Tensor* in = tensor_create(4, shape, false);
    
    // Group 1: channels 0, 1 -> values [0, 2], mean=1, var=1
    // Group 2: channels 2, 3 -> values [4, 6], mean=5, var=1
    for (uint32_t s = 0; s < 4; s++) in->data[0 * 4 + s] = 0.0f;
    for (uint32_t s = 0; s < 4; s++) in->data[1 * 4 + s] = 2.0f;
    for (uint32_t s = 0; s < 4; s++) in->data[2 * 4 + s] = 4.0f;
    for (uint32_t s = 0; s < 4; s++) in->data[3 * 4 + s] = 6.0f;
    
    GroupNormLayer* layer = group_norm_create(2, 4, 1e-5f); // 2 groups, 4 channels
    Tensor* out = group_norm_forward(layer, in);
    
    // Normalization logic: Group 1 (-1, 1) and Group 2 (-1, 1) rescaled
    // Value 0.0 mapped to -1.0
    // Value 2.0 mapped to 1.0
    assert_float_eq(-1.0f, out->data[0], "GroupNorm G1 C0");
    assert_float_eq(1.0f, out->data[4], "GroupNorm G1 C1");
    assert_float_eq(-1.0f, out->data[8], "GroupNorm G2 C0");
    
    printf("[PASS] GroupNorm\n");
    tensor_free(in);
    tensor_free(out);
    group_norm_free(layer);
}

void test_huber_loss() {
    uint32_t shape[] = {2};
    Tensor* pred = tensor_create(1, shape, false);
    Tensor* tgt = tensor_create(1, shape, false);
    
    pred->data[0] = 1.0f; tgt->data[0] = 1.5f; // diff = 0.5 (<= 1.0, so 0.5 * 0.5^2 = 0.125)
    pred->data[1] = 1.0f; tgt->data[1] = 4.0f; // diff = 3.0 (> 1.0, so 1.0 * (3.0 - 0.5) = 2.5)
    // mean = (0.125 + 2.5) / 2 = 1.3125
    
    float loss = huber_loss(pred, tgt, 1.0f);
    assert_float_eq(1.3125f, loss, "HuberLoss");
    
    printf("[PASS] HuberLoss\n");
    tensor_free(pred);
    tensor_free(tgt);
}

void test_lstm() {
    // seq=2, batch=1, input=3
    uint32_t in_shape[] = {2, 1, 3};
    Tensor* in = tensor_zeros(3, in_shape);
    
    LSTMLayer* lstm = lstm_create(3, 2);
    
    uint32_t h_shape[] = {2, 1, 2};
    Tensor* out_h = tensor_zeros(3, h_shape);
    Tensor* out_c = tensor_zeros(3, h_shape);
    
    // Quick smoke validation
    tensor_lstm_forward(lstm, in, NULL, NULL, out_h, out_c);
    
    printf("[PASS] LSTMLayer\n");
    tensor_free(in);
    tensor_free(out_h);
    tensor_free(out_c);
    lstm_free(lstm);
}

int main() {
    test_conv_transpose2d();
    test_group_norm();
    test_huber_loss();
    test_lstm();
    return 0;
}

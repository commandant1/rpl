#include "rpl.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Software implementation of FP32 <-> FP16 conversion
// Based on public domain impls or IEEE 754 standard
// Since we don't assume __fp16 support on x86, we do bit manipulation.

static uint16_t float_to_half_impl(float x) {
    uint32_t f;
    memcpy(&f, &x, 4);

    uint32_t sign = (f >> 31) & 0x01;
    uint32_t exp = (f >> 23) & 0xFF;
    uint32_t mant = f & 0x7FFFFF;

    uint16_t h_sign = sign << 15;
    uint16_t h_exp = 0;
    uint16_t h_mant = 0;

    if (exp == 255) { // Inf or NaN
        h_exp = 31;
        h_mant = mant ? 0x200 : 0; // Simple NaN/Inf
    } else if (exp == 0) { // Zero or Denormal
        h_exp = 0; // maintain zero
    } else {
        int new_exp = (int)exp - 127 + 15;
        if (new_exp >= 31) { // Overflow
            h_exp = 31;
        } else if (new_exp <= 0) { // Underflow -> Denormal or Zero
             // Simplified: flush to zero for speed
             h_exp = 0;
        } else {
            h_exp = new_exp;
            h_mant = mant >> 13;
        }
    }

    return h_sign | (h_exp << 10) | h_mant;
}

static float half_to_float_impl(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x01;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    uint32_t f_sign = sign << 31;
    uint32_t f_exp;
    uint32_t f_mant;

    if (exp == 31) {
        f_exp = 255;
        f_mant = mant ? 0x7FFFFF : 0;
    } else if (exp == 0) {
        if (mant == 0) {
            f_exp = 0;
            f_mant = 0;
        } else {
            // Denormal half - simplified
            f_exp = 0;
            f_mant = 0;
        }
    } else {
        f_exp = exp + 127 - 15;
        f_mant = mant << 13;
    }

    uint32_t f = f_sign | (f_exp << 23) | f_mant;
    float res;
    memcpy(&res, &f, 4);
    return res;
}

HalfTensor* tensor_to_half(const Tensor* t) {
    HalfTensor* ht = (HalfTensor*)calloc(1, sizeof(HalfTensor));
    ht->dims = t->dims;
    ht->size = t->size;
    memcpy(ht->shape, t->shape, sizeof(t->shape));
    memcpy(ht->strides, t->strides, sizeof(t->strides));
    ht->device = t->device;

    ht->_allocation = rpitorch_aligned_alloc(64, ht->size * sizeof(rpl_half));
    ht->data = (rpl_half*)ht->_allocation;

    if (t->device == DEVICE_CPU) {
        #pragma omp parallel for
        for (uint32_t i = 0; i < t->size; i++) {
            ht->data[i] = float_to_half_impl(t->data[i]);
        }
    }
    // GPU handling would differ (copy buffer?)
    return ht;
}

Tensor* tensor_from_half(const HalfTensor* ht) {
    Tensor* t = tensor_create(ht->dims, ht->shape, false);
    t->device = ht->device;
    
    if (ht->device == DEVICE_CPU) {
        #pragma omp parallel for
        for (uint32_t i = 0; i < ht->size; i++) {
            t->data[i] = half_to_float_impl(ht->data[i]);
        }
    }
    return t;
}

void half_tensor_free(HalfTensor* t) {
    if (!t) return;
    if (t->_allocation) rpitorch_aligned_free(t->_allocation);
    free(t);
}

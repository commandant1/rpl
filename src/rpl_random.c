/*
 * RPL Random — tensor creation and random distributions
 */
#include "rpl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static uint64_t rpl_rng_state = 0;
static bool rpl_rng_initialized = false;

static void ensure_rng(void) {
    if (!rpl_rng_initialized) { rpl_rng_state = (uint64_t)time(NULL); rpl_rng_initialized = true; }
}

// xorshift64
static uint64_t xorshift64(void) {
    rpl_rng_state ^= rpl_rng_state << 13;
    rpl_rng_state ^= rpl_rng_state >> 7;
    rpl_rng_state ^= rpl_rng_state << 17;
    return rpl_rng_state;
}

static float rand_uniform(void) { return (float)(xorshift64() & 0xFFFFFFFF) / 4294967296.0f; }

// Box-Muller for normal distribution
static float rand_normal(void) {
    float u1 = rand_uniform(), u2 = rand_uniform();
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

void rpl_manual_seed(uint64_t seed) { rpl_rng_state = seed; rpl_rng_initialized = true; }
void rpl_seed(void) { rpl_rng_state = (uint64_t)time(NULL); rpl_rng_initialized = true; }

// Tensor creation
Tensor* tensor_zeros(uint32_t dims, const uint32_t* shape) {
    Tensor* t = tensor_create(dims, shape, false);
    tensor_fill(t, 0); return t;
}

Tensor* tensor_ones(uint32_t dims, const uint32_t* shape) {
    Tensor* t = tensor_create(dims, shape, false);
    tensor_fill(t, 1.0f); return t;
}

Tensor* tensor_full(uint32_t dims, const uint32_t* shape, float value) {
    Tensor* t = tensor_create(dims, shape, false);
    tensor_fill(t, value); return t;
}

Tensor* tensor_zeros_like(const Tensor* t) { return tensor_zeros(t->dims, t->shape); }
Tensor* tensor_ones_like(const Tensor* t) { return tensor_ones(t->dims, t->shape); }
Tensor* tensor_full_like(const Tensor* t, float value) { return tensor_full(t->dims, t->shape, value); }
Tensor* tensor_empty(uint32_t dims, const uint32_t* shape) { return tensor_create(dims, shape, false); }
Tensor* tensor_empty_like(const Tensor* t) { return tensor_create(t->dims, t->shape, false); }

// Arange
Tensor* tensor_arange(float start, float end, float step) {
    uint32_t n = (uint32_t)ceilf((end - start) / step);
    uint32_t s[1] = {n};
    Tensor* t = tensor_create(1, s, false);
    for (uint32_t i = 0; i < n; i++) t->data[i] = start + i * step;
    return t;
}

// Linspace
Tensor* tensor_linspace(float start, float end, uint32_t steps) {
    uint32_t s[1] = {steps};
    Tensor* t = tensor_create(1, s, false);
    if (steps == 1) { t->data[0] = start; return t; }
    float step = (end - start) / (steps - 1);
    for (uint32_t i = 0; i < steps; i++) t->data[i] = start + i * step;
    return t;
}

// Logspace
Tensor* tensor_logspace(float start, float end, uint32_t steps, float base) {
    Tensor* lin = tensor_linspace(start, end, steps);
    for (uint32_t i = 0; i < lin->size; i++) lin->data[i] = powf(base, lin->data[i]);
    return lin;
}

// Random tensors
Tensor* tensor_rand(uint32_t dims, const uint32_t* shape) {
    ensure_rng();
    Tensor* t = tensor_create(dims, shape, false);
    for (uint32_t i = 0; i < t->size; i++) t->data[i] = rand_uniform();
    return t;
}

Tensor* tensor_randn(uint32_t dims, const uint32_t* shape) {
    ensure_rng();
    Tensor* t = tensor_create(dims, shape, false);
    for (uint32_t i = 0; i < t->size; i++) t->data[i] = rand_normal();
    return t;
}

Tensor* tensor_randint(int32_t low, int32_t high, uint32_t dims, const uint32_t* shape) {
    ensure_rng();
    Tensor* t = tensor_create(dims, shape, false);
    int32_t range = high - low;
    for (uint32_t i = 0; i < t->size; i++) t->data[i] = (float)(low + (int32_t)(xorshift64() % range));
    return t;
}

Tensor* tensor_randperm(uint32_t n) {
    uint32_t s[1] = {n};
    Tensor* t = tensor_create(1, s, false);
    ensure_rng();
    for (uint32_t i = 0; i < n; i++) t->data[i] = (float)i;
    // Fisher-Yates shuffle
    for (uint32_t i = n-1; i > 0; i--) {
        uint32_t j = xorshift64() % (i+1);
        float tmp = t->data[i]; t->data[i] = t->data[j]; t->data[j] = tmp;
    }
    return t;
}

Tensor* tensor_rand_like(const Tensor* t) { return tensor_rand(t->dims, t->shape); }
Tensor* tensor_randn_like(const Tensor* t) { return tensor_randn(t->dims, t->shape); }

// Distributions
Tensor* tensor_bernoulli(const Tensor* probs) {
    ensure_rng();
    Tensor* out = tensor_create(probs->dims, probs->shape, false);
    for (uint32_t i = 0; i < probs->size; i++)
        out->data[i] = (rand_uniform() < probs->data[i]) ? 1.0f : 0.0f;
    return out;
}

Tensor* tensor_normal(float mean, float std, uint32_t dims, const uint32_t* shape) {
    ensure_rng();
    Tensor* t = tensor_create(dims, shape, false);
    for (uint32_t i = 0; i < t->size; i++) t->data[i] = mean + std * rand_normal();
    return t;
}

Tensor* tensor_poisson_sample(const Tensor* rates) {
    ensure_rng();
    Tensor* out = tensor_create(rates->dims, rates->shape, false);
    for (uint32_t i = 0; i < rates->size; i++) {
        float L = expf(-rates->data[i]);
        float p = 1.0f; int32_t k = 0;
        do { k++; p *= rand_uniform(); } while (p > L);
        out->data[i] = (float)(k - 1);
    }
    return out;
}

Tensor* tensor_multinomial(const Tensor* probs, uint32_t num_samples, bool replacement) {
    ensure_rng();
    uint32_t n = probs->size;
    uint32_t s[1] = {num_samples};
    Tensor* out = tensor_create(1, s, false);
    
    float* cumprobs = (float*)malloc(n * sizeof(float));
    cumprobs[0] = probs->data[0];
    for (uint32_t i = 1; i < n; i++) cumprobs[i] = cumprobs[i-1] + probs->data[i];
    float total = cumprobs[n-1];
    
    for (uint32_t s_idx = 0; s_idx < num_samples; s_idx++) {
        float r = rand_uniform() * total;
        uint32_t idx = 0;
        while (idx < n-1 && cumprobs[idx] < r) idx++;
        out->data[s_idx] = (float)idx;
    }
    free(cumprobs);
    return out;
}

// Meshgrid
void tensor_meshgrid(const Tensor** inputs, uint32_t n_inputs, Tensor** outputs) {
    uint32_t shape[MAX_DIMS];
    for (uint32_t i = 0; i < n_inputs; i++) shape[i] = inputs[i]->size;
    
    uint32_t total = 1;
    for (uint32_t i = 0; i < n_inputs; i++) total *= shape[i];
    
    for (uint32_t g = 0; g < n_inputs; g++) {
        outputs[g] = tensor_create(n_inputs, shape, false);
        for (uint32_t idx = 0; idx < total; idx++) {
            uint32_t coords[MAX_DIMS], tmp = idx;
            for (int d = n_inputs-1; d >= 0; d--) { coords[d] = tmp % shape[d]; tmp /= shape[d]; }
            outputs[g]->data[idx] = inputs[g]->data[coords[g]];
        }
    }
}

/*
 * RPL Utilities — tensor info, broadcast, window functions, etc.
 */
#include "rpl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Info
uint32_t tensor_numel(const Tensor* t) { return t->size; }
bool tensor_is_floating_point(const Tensor* t) { return true; /* RPL only has float */ }
bool tensor_is_nonzero(const Tensor* t) { return t->size == 1 && t->data[0] != 0.0f; }

// Contiguous (RPL is always contiguous, just clone)
Tensor* tensor_contiguous(const Tensor* t) {
    Tensor* out = tensor_create(t->dims, t->shape, t->requires_grad);
    memcpy(out->data, t->data, t->size * sizeof(float));
    return out;
}

// Broadcast
Tensor* tensor_broadcast_to(const Tensor* t, uint32_t dims, const uint32_t* shape) {
    Tensor* out = tensor_create(dims, shape, false);
    uint32_t total = 1;
    for (uint32_t i = 0; i < dims; i++) total *= shape[i];
    
    #pragma omp parallel for
    for (uint32_t i = 0; i < total; i++) {
        uint32_t coords[MAX_DIMS], tmp = i;
        for (int d = dims-1; d >= 0; d--) { coords[d] = tmp % shape[d]; tmp /= shape[d]; }
        uint32_t si = 0, stride = 1;
        for (int d = t->dims-1; d >= 0; d--) {
            uint32_t c = coords[d + (dims - t->dims)] % t->shape[d];
            si += c * stride; stride *= t->shape[d];
        }
        out->data[i] = t->data[si];
    }
    return out;
}

// Atleast
Tensor* tensor_atleast_1d(const Tensor* t) {
    if (t->dims >= 1) { Tensor* o = tensor_create(t->dims, t->shape, false); memcpy(o->data, t->data, t->size*sizeof(float)); return o; }
    uint32_t s[1] = {1}; Tensor* o = tensor_create(1, s, false); o->data[0] = t->data[0]; return o;
}
Tensor* tensor_atleast_2d(const Tensor* t) {
    if (t->dims >= 2) { Tensor* o = tensor_create(t->dims, t->shape, false); memcpy(o->data, t->data, t->size*sizeof(float)); return o; }
    if (t->dims == 1) { uint32_t s[2] = {1, t->shape[0]}; Tensor* o = tensor_create(2, s, false); memcpy(o->data, t->data, t->size*sizeof(float)); return o; }
    uint32_t s[2] = {1, 1}; Tensor* o = tensor_create(2, s, false); o->data[0] = t->data[0]; return o;
}
Tensor* tensor_atleast_3d(const Tensor* t) {
    Tensor* t2 = tensor_atleast_2d(t);
    if (t2->dims >= 3) return t2;
    uint32_t s[3] = {t2->shape[0], t2->shape[1], 1};
    Tensor* o = tensor_create(3, s, false); memcpy(o->data, t2->data, t2->size*sizeof(float));
    tensor_free(t2); return o;
}

// Block diagonal
Tensor* tensor_block_diag(const Tensor** tensors, uint32_t num) {
    uint32_t total_r = 0, total_c = 0;
    for (uint32_t i = 0; i < num; i++) { total_r += tensors[i]->shape[0]; total_c += tensors[i]->shape[1]; }
    uint32_t s[2] = {total_r, total_c};
    Tensor* out = tensor_create(2, s, false);
    tensor_fill(out, 0);
    uint32_t ro = 0, co = 0;
    for (uint32_t n = 0; n < num; n++) {
        uint32_t r = tensors[n]->shape[0], c = tensors[n]->shape[1];
        for (uint32_t i = 0; i < r; i++)
            memcpy(&out->data[(ro+i)*total_c+co], &tensors[n]->data[i*c], c*sizeof(float));
        ro += r; co += c;
    }
    return out;
}

// Vander matrix
Tensor* tensor_vander(const Tensor* x, uint32_t N, bool increasing) {
    uint32_t n = x->size;
    uint32_t s[2] = {n, N};
    Tensor* out = tensor_create(2, s, false);
    for (uint32_t i = 0; i < n; i++)
        for (uint32_t j = 0; j < N; j++)
            out->data[i*N+j] = powf(x->data[i], increasing ? (float)j : (float)(N-1-j));
    return out;
}

// Window functions
Tensor* tensor_hann_window(uint32_t size) {
    uint32_t s[1] = {size};
    Tensor* t = tensor_create(1, s, false);
    for (uint32_t i = 0; i < size; i++)
        t->data[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (size - 1)));
    return t;
}

Tensor* tensor_hamming_window(uint32_t size) {
    uint32_t s[1] = {size};
    Tensor* t = tensor_create(1, s, false);
    for (uint32_t i = 0; i < size; i++)
        t->data[i] = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * i / (size - 1));
    return t;
}

Tensor* tensor_blackman_window(uint32_t size) {
    uint32_t s[1] = {size};
    Tensor* t = tensor_create(1, s, false);
    for (uint32_t i = 0; i < size; i++)
        t->data[i] = 0.42f - 0.5f * cosf(2.0f*(float)M_PI*i/(size-1)) + 0.08f * cosf(4.0f*(float)M_PI*i/(size-1));
    return t;
}

Tensor* tensor_bartlett_window(uint32_t size) {
    uint32_t s[1] = {size};
    Tensor* t = tensor_create(1, s, false);
    float half = (size - 1) / 2.0f;
    for (uint32_t i = 0; i < size; i++)
        t->data[i] = 1.0f - fabsf(((float)i - half) / half);
    return t;
}

Tensor* tensor_kaiser_window(uint32_t size, float beta) {
    uint32_t s[1] = {size};
    Tensor* t = tensor_create(1, s, false);
    // Approximate I0 using polynomial
    float i0_beta = 1.0f;
    { float x = beta; float ax = fabsf(x);
      if (ax < 3.75f) { float tt = (x/3.75f); tt*=tt; i0_beta = 1.0f+tt*(3.5156229f+tt*(3.0899424f+tt*(1.2067492f+tt*(0.2659732f+tt*(0.0360768f+tt*0.0045813f))))); }
      else { float tt = 3.75f/ax; i0_beta = (expf(ax)/sqrtf(ax))*(0.39894228f+tt*(0.01328592f+tt*(0.00225319f+tt*(-0.00157565f+tt*(0.00916281f+tt*(-0.02057706f+tt*(0.02635537f+tt*(-0.01647633f+tt*0.00392377f)))))))); }
    }
    float half = (size - 1) / 2.0f;
    for (uint32_t i = 0; i < size; i++) {
        float r = (i - half) / half;
        float arg = beta * sqrtf(1.0f - r*r);
        float i0_arg = 1.0f;
        { float x = arg; float ax = fabsf(x);
          if (ax < 3.75f) { float tt=(x/3.75f); tt*=tt; i0_arg=1.0f+tt*(3.5156229f+tt*(3.0899424f+tt*(1.2067492f+tt*(0.2659732f+tt*(0.0360768f+tt*0.0045813f))))); }
          else { float tt=3.75f/ax; i0_arg=(expf(ax)/sqrtf(ax))*(0.39894228f+tt*(0.01328592f+tt*(0.00225319f+tt*(-0.00157565f+tt*(0.00916281f+tt*(-0.02057706f+tt*(0.02635537f+tt*(-0.01647633f+tt*0.00392377f)))))))); }
        }
        t->data[i] = i0_arg / i0_beta;
    }
    return t;
}

// Convolve (1D, full mode)
Tensor* tensor_convolve(const Tensor* a, const Tensor* v) {
    uint32_t len = a->size + v->size - 1;
    uint32_t s[1] = {len};
    Tensor* out = tensor_create(1, s, false);
    tensor_fill(out, 0);
    for (uint32_t i = 0; i < a->size; i++)
        for (uint32_t j = 0; j < v->size; j++)
            out->data[i+j] += a->data[i] * v->data[j];
    return out;
}

// Interp (1D linear interpolation)
Tensor* tensor_interp(const Tensor* x, const Tensor* xp, const Tensor* fp) {
    Tensor* out = tensor_create(x->dims, x->shape, false);
    uint32_t n = xp->size;
    for (uint32_t i = 0; i < x->size; i++) {
        float xi = x->data[i];
        if (xi <= xp->data[0]) { out->data[i] = fp->data[0]; continue; }
        if (xi >= xp->data[n-1]) { out->data[i] = fp->data[n-1]; continue; }
        // Binary search
        uint32_t lo = 0, hi = n-1;
        while (lo < hi-1) { uint32_t mid = (lo+hi)/2; if (xp->data[mid] <= xi) lo = mid; else hi = mid; }
        float t = (xi - xp->data[lo]) / (xp->data[hi] - xp->data[lo]);
        out->data[i] = fp->data[lo] + t * (fp->data[hi] - fp->data[lo]);
    }
    return out;
}

// Bincount
Tensor* tensor_bincount(const Tensor* t, uint32_t minlength) {
    uint32_t mx = minlength;
    for (uint32_t i = 0; i < t->size; i++) { uint32_t v = (uint32_t)t->data[i]; if (v >= mx) mx = v+1; }
    uint32_t s[1] = {mx};
    Tensor* out = tensor_create(1, s, false);
    tensor_fill(out, 0);
    for (uint32_t i = 0; i < t->size; i++) out->data[(uint32_t)t->data[i]] += 1.0f;
    return out;
}

// Histogram
Tensor* tensor_histc(const Tensor* t, uint32_t bins, float min_val, float max_val) {
    if (min_val == max_val) { min_val = t->data[0]; max_val = t->data[0];
        for (uint32_t i=1;i<t->size;i++){if(t->data[i]<min_val)min_val=t->data[i]; if(t->data[i]>max_val)max_val=t->data[i];}
    }
    uint32_t s[1] = {bins};
    Tensor* out = tensor_create(1, s, false);
    tensor_fill(out, 0);
    float bw = (max_val - min_val) / bins;
    for (uint32_t i = 0; i < t->size; i++) {
        int32_t b = (int32_t)((t->data[i] - min_val) / bw);
        if (b < 0) b = 0; if ((uint32_t)b >= bins) b = bins-1;
        out->data[b] += 1.0f;
    }
    return out;
}

// Trapezoid integration
float tensor_trapezoid(const Tensor* y, float dx) {
    float s = 0;
    for (uint32_t i = 1; i < y->size; i++) s += (y->data[i-1] + y->data[i]) * 0.5f * dx;
    return s;
}

// Corrcoef
Tensor* tensor_corrcoef(const Tensor* t) {
    // t is (N, M): N variables, M observations
    uint32_t N = t->shape[0], M = t->shape[1];
    uint32_t s[2] = {N, N};
    Tensor* out = tensor_create(2, s, false);
    
    float* means = (float*)malloc(N*sizeof(float));
    float* stds = (float*)malloc(N*sizeof(float));
    for (uint32_t i = 0; i < N; i++) {
        float m = 0; for (uint32_t j = 0; j < M; j++) m += t->data[i*M+j]; m /= M; means[i] = m;
        float v = 0; for (uint32_t j = 0; j < M; j++) { float d = t->data[i*M+j]-m; v += d*d; } stds[i] = sqrtf(v/(M-1));
    }
    for (uint32_t i = 0; i < N; i++)
        for (uint32_t j = 0; j < N; j++) {
            float cov = 0;
            for (uint32_t k = 0; k < M; k++) cov += (t->data[i*M+k]-means[i])*(t->data[j*M+k]-means[j]);
            cov /= (M-1);
            out->data[i*N+j] = cov / (stds[i]*stds[j]);
        }
    free(means); free(stds);
    return out;
}

// Cdist (pairwise distances)
Tensor* tensor_cdist(const Tensor* x1, const Tensor* x2, float p) {
    uint32_t N = x1->shape[0], M = x2->shape[0], D = x1->shape[1];
    uint32_t s[2] = {N, M};
    Tensor* out = tensor_create(2, s, false);
    #pragma omp parallel for collapse(2)
    for (uint32_t i = 0; i < N; i++)
        for (uint32_t j = 0; j < M; j++) {
            float d = 0;
            for (uint32_t k = 0; k < D; k++) d += powf(fabsf(x1->data[i*D+k]-x2->data[j*D+k]), p);
            out->data[i*M+j] = powf(d, 1.0f/p);
        }
    return out;
}

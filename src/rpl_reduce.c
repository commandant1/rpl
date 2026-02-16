/*
 * RPL Reduction Operations — sum, mean, var, argmax, cumsum, diff, etc.
 */
#include "rpl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>

// ============================================================
// Full reductions — NEON-optimized for Cortex-A72
// 16 floats/iter = 1 cache line, 4 accumulators hide FP latency
// ============================================================

float tensor_sum_all(const Tensor* t) {
#if RPITORCH_HAS_NEON
    const float* d = t->data;
    float32x4_t v0 = vdupq_n_f32(0), v1 = v0, v2 = v0, v3 = v0;
    uint32_t i = 0;
    for (; i + 16 <= t->size; i += 16) {
        __builtin_prefetch(&d[i + 64], 0, 1);  // prefetch 4 cache lines ahead
        v0 = vaddq_f32(v0, vld1q_f32(&d[i]));
        v1 = vaddq_f32(v1, vld1q_f32(&d[i + 4]));
        v2 = vaddq_f32(v2, vld1q_f32(&d[i + 8]));
        v3 = vaddq_f32(v3, vld1q_f32(&d[i + 12]));
    }
    v0 = vaddq_f32(vaddq_f32(v0, v1), vaddq_f32(v2, v3));
    float s = vaddvq_f32(v0);
    for (; i < t->size; i++) s += d[i];
    return s;
#else
    float s = 0; for (uint32_t i = 0; i < t->size; i++) s += t->data[i]; return s;
#endif
}

float tensor_prod_all(const Tensor* t) {
    float p = 1; for (uint32_t i = 0; i < t->size; i++) p *= t->data[i]; return p;
}

float tensor_mean_all(const Tensor* t) { return tensor_sum_all(t) / t->size; }

float tensor_var_all(const Tensor* t, bool unbiased) {
    float m = tensor_mean_all(t);
#if RPITORCH_HAS_NEON
    const float* d = t->data;
    float32x4_t vm = vdupq_n_f32(m);
    float32x4_t v0 = vdupq_n_f32(0), v1 = v0, v2 = v0, v3 = v0;
    uint32_t i = 0;
    for (; i + 16 <= t->size; i += 16) {
        __builtin_prefetch(&d[i + 64], 0, 1);
        float32x4_t d0 = vsubq_f32(vld1q_f32(&d[i]),     vm);
        float32x4_t d1 = vsubq_f32(vld1q_f32(&d[i + 4]), vm);
        float32x4_t d2 = vsubq_f32(vld1q_f32(&d[i + 8]), vm);
        float32x4_t d3 = vsubq_f32(vld1q_f32(&d[i + 12]),vm);
        v0 = vfmaq_f32(v0, d0, d0);
        v1 = vfmaq_f32(v1, d1, d1);
        v2 = vfmaq_f32(v2, d2, d2);
        v3 = vfmaq_f32(v3, d3, d3);
    }
    v0 = vaddq_f32(vaddq_f32(v0, v1), vaddq_f32(v2, v3));
    float s = vaddvq_f32(v0);
    for (; i < t->size; i++) { float dd = d[i] - m; s += dd * dd; }
#else
    float s = 0;
    for (uint32_t i = 0; i < t->size; i++) { float dd = t->data[i] - m; s += dd * dd; }
#endif
    return s / (unbiased ? (t->size - 1) : t->size);
}

float tensor_std_all(const Tensor* t, bool unbiased) { return sqrtf(tensor_var_all(t, unbiased)); }

float tensor_max_all(const Tensor* t) {
#if RPITORCH_HAS_NEON
    const float* d = t->data;
    if (t->size >= 16) {
        float32x4_t v0 = vld1q_f32(&d[0]), v1 = vld1q_f32(&d[4]);
        float32x4_t v2 = vld1q_f32(&d[8]), v3 = vld1q_f32(&d[12]);
        uint32_t i = 16;
        for (; i + 16 <= t->size; i += 16) {
            __builtin_prefetch(&d[i + 64], 0, 1);
            v0 = vmaxq_f32(v0, vld1q_f32(&d[i]));
            v1 = vmaxq_f32(v1, vld1q_f32(&d[i + 4]));
            v2 = vmaxq_f32(v2, vld1q_f32(&d[i + 8]));
            v3 = vmaxq_f32(v3, vld1q_f32(&d[i + 12]));
        }
        v0 = vmaxq_f32(vmaxq_f32(v0, v1), vmaxq_f32(v2, v3));
        float m = vmaxvq_f32(v0);
        for (; i < t->size; i++) if (d[i] > m) m = d[i];
        return m;
    }
#endif
    float m = -FLT_MAX; for (uint32_t i = 0; i < t->size; i++) if (t->data[i] > m) m = t->data[i]; return m;
}

float tensor_min_all(const Tensor* t) {
#if RPITORCH_HAS_NEON
    const float* d = t->data;
    if (t->size >= 16) {
        float32x4_t v0 = vld1q_f32(&d[0]), v1 = vld1q_f32(&d[4]);
        float32x4_t v2 = vld1q_f32(&d[8]), v3 = vld1q_f32(&d[12]);
        uint32_t i = 16;
        for (; i + 16 <= t->size; i += 16) {
            __builtin_prefetch(&d[i + 64], 0, 1);
            v0 = vminq_f32(v0, vld1q_f32(&d[i]));
            v1 = vminq_f32(v1, vld1q_f32(&d[i + 4]));
            v2 = vminq_f32(v2, vld1q_f32(&d[i + 8]));
            v3 = vminq_f32(v3, vld1q_f32(&d[i + 12]));
        }
        v0 = vminq_f32(vminq_f32(v0, v1), vminq_f32(v2, v3));
        float m = vminvq_f32(v0);
        for (; i < t->size; i++) if (d[i] < m) m = d[i];
        return m;
    }
#endif
    float m = FLT_MAX; for (uint32_t i = 0; i < t->size; i++) if (t->data[i] < m) m = t->data[i]; return m;
}

uint32_t tensor_argmax_all(const Tensor* t) { float m = -FLT_MAX; uint32_t idx = 0; for (uint32_t i = 0; i < t->size; i++) if (t->data[i] > m) { m = t->data[i]; idx = i; } return idx; }
uint32_t tensor_argmin_all(const Tensor* t) { float m = FLT_MAX; uint32_t idx = 0; for (uint32_t i = 0; i < t->size; i++) if (t->data[i] < m) { m = t->data[i]; idx = i; } return idx; }

float tensor_norm_all(const Tensor* t, float p) {
    if (p == 0) { float c = 0; for (uint32_t i = 0; i < t->size; i++) if (t->data[i] != 0) c++; return c; }
    if (p == INFINITY) { float m = 0; for (uint32_t i = 0; i < t->size; i++) { float a = fabsf(t->data[i]); if (a > m) m = a; } return m; }
    // L2 fast path with NEON
    if (p == 2.0f) {
#if RPITORCH_HAS_NEON
        const float* d = t->data;
        float32x4_t v0 = vdupq_n_f32(0), v1 = v0, v2 = v0, v3 = v0;
        uint32_t i = 0;
        for (; i + 16 <= t->size; i += 16) {
            __builtin_prefetch(&d[i + 64], 0, 1);
            float32x4_t a0 = vld1q_f32(&d[i]),     a1 = vld1q_f32(&d[i + 4]);
            float32x4_t a2 = vld1q_f32(&d[i + 8]), a3 = vld1q_f32(&d[i + 12]);
            v0 = vfmaq_f32(v0, a0, a0);
            v1 = vfmaq_f32(v1, a1, a1);
            v2 = vfmaq_f32(v2, a2, a2);
            v3 = vfmaq_f32(v3, a3, a3);
        }
        v0 = vaddq_f32(vaddq_f32(v0, v1), vaddq_f32(v2, v3));
        float s = vaddvq_f32(v0);
        for (; i < t->size; i++) s += d[i] * d[i];
        return sqrtf(s);
#endif
    }
    float s = 0; for (uint32_t i = 0; i < t->size; i++) s += powf(fabsf(t->data[i]), p); return powf(s, 1.0f / p);
}
float tensor_logsumexp_all(const Tensor* t) { float m=tensor_max_all(t),s=0; for(uint32_t i=0;i<t->size;i++) s+=expf(t->data[i]-m); return m+logf(s); }
uint32_t tensor_count_nonzero_all(const Tensor* t) { uint32_t c=0; for(uint32_t i=0;i<t->size;i++) if(t->data[i]!=0) c++; return c; }

// Along-axis reduction helper
static void reduce_axis_info(const Tensor* t, int32_t dim, uint32_t* outer, uint32_t* ds, uint32_t* inner) {
    if (dim < 0) dim += t->dims;
    *outer = 1; *inner = 1; *ds = t->shape[dim];
    for (int32_t d = 0; d < dim; d++) *outer *= t->shape[d];
    for (uint32_t d = dim+1; d < t->dims; d++) *inner *= t->shape[d];
}

static Tensor* make_reduced(const Tensor* t, int32_t dim) {
    if (dim < 0) dim += t->dims;
    uint32_t nd = t->dims - 1, s[MAX_DIMS], j = 0;
    for (uint32_t i = 0; i < t->dims; i++) if (i != (uint32_t)dim) s[j++] = t->shape[i];
    if (nd == 0) { s[0] = 1; nd = 1; }
    return tensor_create(nd, s, false);
}

Tensor* tensor_sum(const Tensor* t, int32_t dim) {
    uint32_t outer, ds, inner;
    reduce_axis_info(t, dim, &outer, &ds, &inner);
    Tensor* out = make_reduced(t, dim);
    #pragma omp parallel for
    for (uint32_t o = 0; o < outer; o++)
        for (uint32_t i = 0; i < inner; i++) {
            float s = 0;
            for (uint32_t d = 0; d < ds; d++) s += t->data[(o*ds+d)*inner+i];
            out->data[o*inner+i] = s;
        }
    return out;
}

Tensor* tensor_prod(const Tensor* t, int32_t dim) {
    uint32_t outer, ds, inner; reduce_axis_info(t, dim, &outer, &ds, &inner);
    Tensor* out = make_reduced(t, dim);
    #pragma omp parallel for
    for (uint32_t o = 0; o < outer; o++)
        for (uint32_t i = 0; i < inner; i++) {
            float p = 1;
            for (uint32_t d = 0; d < ds; d++) p *= t->data[(o*ds+d)*inner+i];
            out->data[o*inner+i] = p;
        }
    return out;
}

Tensor* tensor_mean(const Tensor* t, int32_t dim) {
    Tensor* s = tensor_sum(t, dim);
    uint32_t ds = t->shape[dim < 0 ? dim + t->dims : dim];
    float inv = 1.0f / ds;
    for (uint32_t i = 0; i < s->size; i++) s->data[i] *= inv;
    return s;
}

Tensor* tensor_var(const Tensor* t, int32_t dim, bool unbiased) {
    uint32_t outer, ds, inner; reduce_axis_info(t, dim, &outer, &ds, &inner);
    Tensor* out = make_reduced(t, dim);
    float denom = unbiased ? (ds-1) : ds;
    #pragma omp parallel for
    for (uint32_t o = 0; o < outer; o++)
        for (uint32_t i = 0; i < inner; i++) {
            float m = 0;
            for (uint32_t d = 0; d < ds; d++) m += t->data[(o*ds+d)*inner+i];
            m /= ds;
            float v = 0;
            for (uint32_t d = 0; d < ds; d++) { float x = t->data[(o*ds+d)*inner+i]-m; v += x*x; }
            out->data[o*inner+i] = v / denom;
        }
    return out;
}

Tensor* tensor_std(const Tensor* t, int32_t dim, bool unbiased) {
    Tensor* v = tensor_var(t, dim, unbiased);
    for (uint32_t i = 0; i < v->size; i++) v->data[i] = sqrtf(v->data[i]);
    return v;
}

Tensor* tensor_max_dim(const Tensor* t, int32_t dim) {
    uint32_t outer, ds, inner; reduce_axis_info(t, dim, &outer, &ds, &inner);
    Tensor* out = make_reduced(t, dim);
    #pragma omp parallel for
    for (uint32_t o = 0; o < outer; o++)
        for (uint32_t i = 0; i < inner; i++) {
            float m = -FLT_MAX;
            for (uint32_t d = 0; d < ds; d++) { float v = t->data[(o*ds+d)*inner+i]; if(v>m) m=v; }
            out->data[o*inner+i] = m;
        }
    return out;
}

Tensor* tensor_min_dim(const Tensor* t, int32_t dim) {
    uint32_t outer, ds, inner; reduce_axis_info(t, dim, &outer, &ds, &inner);
    Tensor* out = make_reduced(t, dim);
    #pragma omp parallel for
    for (uint32_t o = 0; o < outer; o++)
        for (uint32_t i = 0; i < inner; i++) {
            float m = FLT_MAX;
            for (uint32_t d = 0; d < ds; d++) { float v = t->data[(o*ds+d)*inner+i]; if(v<m) m=v; }
            out->data[o*inner+i] = m;
        }
    return out;
}

Tensor* tensor_argmax_dim(const Tensor* t, int32_t dim) {
    uint32_t outer, ds, inner; reduce_axis_info(t, dim, &outer, &ds, &inner);
    Tensor* out = make_reduced(t, dim);
    #pragma omp parallel for
    for (uint32_t o = 0; o < outer; o++)
        for (uint32_t i = 0; i < inner; i++) {
            float m = -FLT_MAX; uint32_t mi = 0;
            for (uint32_t d = 0; d < ds; d++) { float v = t->data[(o*ds+d)*inner+i]; if(v>m){m=v;mi=d;} }
            out->data[o*inner+i] = (float)mi;
        }
    return out;
}

Tensor* tensor_argmin_dim(const Tensor* t, int32_t dim) {
    uint32_t outer, ds, inner; reduce_axis_info(t, dim, &outer, &ds, &inner);
    Tensor* out = make_reduced(t, dim);
    #pragma omp parallel for
    for (uint32_t o = 0; o < outer; o++)
        for (uint32_t i = 0; i < inner; i++) {
            float m = FLT_MAX; uint32_t mi = 0;
            for (uint32_t d = 0; d < ds; d++) { float v = t->data[(o*ds+d)*inner+i]; if(v<m){m=v;mi=d;} }
            out->data[o*inner+i] = (float)mi;
        }
    return out;
}

// Cumulative
Tensor* tensor_cumsum(const Tensor* t, int32_t dim) {
    uint32_t outer, ds, inner; reduce_axis_info(t, dim, &outer, &ds, &inner);
    Tensor* out = tensor_create(t->dims, t->shape, false);
    for (uint32_t o = 0; o < outer; o++)
        for (uint32_t i = 0; i < inner; i++) {
            float s = 0;
            for (uint32_t d = 0; d < ds; d++) { s += t->data[(o*ds+d)*inner+i]; out->data[(o*ds+d)*inner+i] = s; }
        }
    return out;
}

Tensor* tensor_cumprod(const Tensor* t, int32_t dim) {
    uint32_t outer, ds, inner; reduce_axis_info(t, dim, &outer, &ds, &inner);
    Tensor* out = tensor_create(t->dims, t->shape, false);
    for (uint32_t o = 0; o < outer; o++)
        for (uint32_t i = 0; i < inner; i++) {
            float p = 1;
            for (uint32_t d = 0; d < ds; d++) { p *= t->data[(o*ds+d)*inner+i]; out->data[(o*ds+d)*inner+i] = p; }
        }
    return out;
}

Tensor* tensor_cummax(const Tensor* t, int32_t dim) {
    uint32_t outer, ds, inner; reduce_axis_info(t, dim, &outer, &ds, &inner);
    Tensor* out = tensor_create(t->dims, t->shape, false);
    for (uint32_t o = 0; o < outer; o++)
        for (uint32_t i = 0; i < inner; i++) {
            float m = -FLT_MAX;
            for (uint32_t d = 0; d < ds; d++) { float v=t->data[(o*ds+d)*inner+i]; if(v>m)m=v; out->data[(o*ds+d)*inner+i]=m; }
        }
    return out;
}

Tensor* tensor_cummin(const Tensor* t, int32_t dim) {
    uint32_t outer, ds, inner; reduce_axis_info(t, dim, &outer, &ds, &inner);
    Tensor* out = tensor_create(t->dims, t->shape, false);
    for (uint32_t o = 0; o < outer; o++)
        for (uint32_t i = 0; i < inner; i++) {
            float m = FLT_MAX;
            for (uint32_t d = 0; d < ds; d++) { float v=t->data[(o*ds+d)*inner+i]; if(v<m)m=v; out->data[(o*ds+d)*inner+i]=m; }
        }
    return out;
}

// NaN-safe
float tensor_nansum_all(const Tensor* t) { float s=0; for(uint32_t i=0;i<t->size;i++) if(!isnan(t->data[i])) s+=t->data[i]; return s; }
float tensor_nanmean_all(const Tensor* t) { float s=0; uint32_t c=0; for(uint32_t i=0;i<t->size;i++) if(!isnan(t->data[i])){s+=t->data[i];c++;} return c?s/c:0; }
float tensor_nanprod_all(const Tensor* t) { float p=1; for(uint32_t i=0;i<t->size;i++) if(!isnan(t->data[i])) p*=t->data[i]; return p; }
float tensor_nanmax_all(const Tensor* t) { float m=-FLT_MAX; for(uint32_t i=0;i<t->size;i++) if(!isnan(t->data[i])&&t->data[i]>m) m=t->data[i]; return m; }
float tensor_nanmin_all(const Tensor* t) { float m=FLT_MAX; for(uint32_t i=0;i<t->size;i++) if(!isnan(t->data[i])&&t->data[i]<m) m=t->data[i]; return m; }

// Median (full)
float tensor_median_all(const Tensor* t) {
    float* tmp = (float*)malloc(t->size * sizeof(float));
    memcpy(tmp, t->data, t->size * sizeof(float));
    // Simple insertion sort for small, qsort compare for large
    for (uint32_t i = 1; i < t->size; i++) {
        float key = tmp[i]; int32_t j = i-1;
        while (j >= 0 && tmp[j] > key) { tmp[j+1] = tmp[j]; j--; }
        tmp[j+1] = key;
    }
    float m = (t->size % 2) ? tmp[t->size/2] : (tmp[t->size/2-1]+tmp[t->size/2])*0.5f;
    free(tmp); return m;
}

// Diff
Tensor* tensor_diff(const Tensor* t, int32_t dim) {
    uint32_t outer, ds, inner; reduce_axis_info(t, dim, &outer, &ds, &inner);
    uint32_t ns[MAX_DIMS]; memcpy(ns, t->shape, t->dims*sizeof(uint32_t));
    ns[dim < 0 ? dim+t->dims : dim] = ds - 1;
    Tensor* out = tensor_create(t->dims, ns, false);
    #pragma omp parallel for
    for (uint32_t o = 0; o < outer; o++)
        for (uint32_t i = 0; i < inner; i++)
            for (uint32_t d = 0; d < ds-1; d++)
                out->data[(o*(ds-1)+d)*inner+i] = t->data[(o*ds+d+1)*inner+i] - t->data[(o*ds+d)*inner+i];
    return out;
}

// All / Any
bool tensor_all(const Tensor* t) { for(uint32_t i=0;i<t->size;i++) if(t->data[i]==0) return false; return true; }
bool tensor_any(const Tensor* t) { for(uint32_t i=0;i<t->size;i++) if(t->data[i]!=0) return true; return false; }

// Dist (Lp distance)
float tensor_dist(const Tensor* a, const Tensor* b, float p) {
    float s = 0;
    for (uint32_t i = 0; i < a->size; i++) s += powf(fabsf(a->data[i]-b->data[i%b->size]), p);
    return powf(s, 1.0f/p);
}

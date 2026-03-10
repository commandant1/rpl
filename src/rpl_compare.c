/*
 * RPL Compare & Logic — eq, lt, gt, sort, topk, logical, test ops
 */
#include "rpl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>

static inline Tensor* _like(const Tensor* t) { return tensor_create(t->dims, t->shape, false); }

// Comparison — NEON-optimized for same-size tensors
// vceqq/vcltq/etc return uint32 masks; AND with 1.0f bit-pattern to get float 1.0/0.0
#if RPITORCH_HAS_NEON
static inline void _cmp_neon_eq(float* out, const float* a, const float* b, uint32_t n) {
    uint32x4_t vone = vreinterpretq_u32_f32(vdupq_n_f32(1.0f));
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_f32(&out[i], vreinterpretq_f32_u32(vandq_u32(vceqq_f32(vld1q_f32(&a[i]), vld1q_f32(&b[i])), vone)));
    for (; i < n; i++) out[i] = (a[i] == b[i]) ? 1.0f : 0.0f;
}
/* added NEON not-equal helper */
static inline void _cmp_neon_ne(float* out, const float* a, const float* b, uint32_t n) {
    uint32x4_t vone = vreinterpretq_u32_f32(vdupq_n_f32(1.0f));
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_f32(&out[i], vreinterpretq_f32_u32(vandq_u32(vmvnq_u32(vceqq_f32(vld1q_f32(&a[i]), vld1q_f32(&b[i]))), vone)));
    for (; i < n; i++) out[i] = (a[i] != b[i]) ? 1.0f : 0.0f;
}
static inline void _cmp_neon_lt(float* out, const float* a, const float* b, uint32_t n) {
    uint32x4_t vone = vreinterpretq_u32_f32(vdupq_n_f32(1.0f));
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_f32(&out[i], vreinterpretq_f32_u32(vandq_u32(vcltq_f32(vld1q_f32(&a[i]), vld1q_f32(&b[i])), vone)));
    for (; i < n; i++) out[i] = (a[i] < b[i]) ? 1.0f : 0.0f;
}
static inline void _cmp_neon_gt(float* out, const float* a, const float* b, uint32_t n) {
    uint32x4_t vone = vreinterpretq_u32_f32(vdupq_n_f32(1.0f));
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_f32(&out[i], vreinterpretq_f32_u32(vandq_u32(vcgtq_f32(vld1q_f32(&a[i]), vld1q_f32(&b[i])), vone)));
    for (; i < n; i++) out[i] = (a[i] > b[i]) ? 1.0f : 0.0f;
}
static inline void _cmp_neon_le(float* out, const float* a, const float* b, uint32_t n) {
    uint32x4_t vone = vreinterpretq_u32_f32(vdupq_n_f32(1.0f));
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_f32(&out[i], vreinterpretq_f32_u32(vandq_u32(vcleq_f32(vld1q_f32(&a[i]), vld1q_f32(&b[i])), vone)));
    for (; i < n; i++) out[i] = (a[i] <= b[i]) ? 1.0f : 0.0f;
}
static inline void _cmp_neon_ge(float* out, const float* a, const float* b, uint32_t n) {
    uint32x4_t vone = vreinterpretq_u32_f32(vdupq_n_f32(1.0f));
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_f32(&out[i], vreinterpretq_f32_u32(vandq_u32(vcgeq_f32(vld1q_f32(&a[i]), vld1q_f32(&b[i])), vone)));
    for (; i < n; i++) out[i] = (a[i] >= b[i]) ? 1.0f : 0.0f;
}
#endif

#define DEF_CMP_NEON(name, neon_fn, op) \
Tensor* tensor_##name(const Tensor* a, const Tensor* b) { \
    Tensor* out = _like(a); \
    if (a->size == b->size) { \
        RPITORCH_HAS_NEON_COND(neon_fn(out->data, a->data, b->data, a->size), \
            for (uint32_t i = 0; i < a->size; i++) out->data[i] = (a->data[i] op b->data[i]) ? 1.0f : 0.0f); \
    } else { \
        for (uint32_t i = 0; i < a->size; i++) out->data[i] = (a->data[i] op b->data[i%b->size]) ? 1.0f : 0.0f; \
    } \
    return out; \
}

// Need a helper macro for conditional NEON
#if RPITORCH_HAS_NEON
#define RPITORCH_HAS_NEON_COND(neon_code, scalar_code) neon_code
#else
#define RPITORCH_HAS_NEON_COND(neon_code, scalar_code) scalar_code
#endif

DEF_CMP_NEON(eq, _cmp_neon_eq, ==)
DEF_CMP_NEON(ne, _cmp_neon_ne, !=)  // use dedicated NEON 'not-equal' helper
DEF_CMP_NEON(lt, _cmp_neon_lt, <)
DEF_CMP_NEON(le, _cmp_neon_le, <=)
DEF_CMP_NEON(gt, _cmp_neon_gt, >)
DEF_CMP_NEON(ge, _cmp_neon_ge, >=)

bool tensor_equal(const Tensor* a, const Tensor* b) {
    if (a->size != b->size || a->dims != b->dims) return false;
    for (uint32_t i = 0; i < a->dims; i++) if (a->shape[i] != b->shape[i]) return false;
    for (uint32_t i = 0; i < a->size; i++) if (a->data[i] != b->data[i]) return false;
    return true;
}

bool tensor_allclose(const Tensor* a, const Tensor* b, float rtol, float atol) {
    if (a->size != b->size) return false;
    for (uint32_t i = 0; i < a->size; i++)
        if (fabsf(a->data[i]-b->data[i]) > atol + rtol * fabsf(b->data[i])) return false;
    return true;
}

Tensor* tensor_isclose(const Tensor* a, const Tensor* b, float rtol, float atol) {
    Tensor* out = _like(a);
    for (uint32_t i = 0; i < a->size; i++)
        out->data[i] = (fabsf(a->data[i]-b->data[i%b->size]) <= atol + rtol*fabsf(b->data[i%b->size])) ? 1.0f : 0.0f;
    return out;
}

// Logical
Tensor* tensor_logical_and(const Tensor* a, const Tensor* b) {
    Tensor* out = _like(a);
    for (uint32_t i = 0; i < a->size; i++) out->data[i] = (a->data[i]!=0 && b->data[i%b->size]!=0) ? 1.0f : 0.0f;
    return out;
}
Tensor* tensor_logical_or(const Tensor* a, const Tensor* b) {
    Tensor* out = _like(a);
    for (uint32_t i = 0; i < a->size; i++) out->data[i] = (a->data[i]!=0 || b->data[i%b->size]!=0) ? 1.0f : 0.0f;
    return out;
}
Tensor* tensor_logical_not(const Tensor* t) {
    Tensor* out = _like(t);
    for (uint32_t i = 0; i < t->size; i++) out->data[i] = (t->data[i]==0) ? 1.0f : 0.0f;
    return out;
}
Tensor* tensor_logical_xor(const Tensor* a, const Tensor* b) {
    Tensor* out = _like(a);
    for (uint32_t i = 0; i < a->size; i++) out->data[i] = ((a->data[i]!=0) != (b->data[i%b->size]!=0)) ? 1.0f : 0.0f;
    return out;
}

// Test functions
Tensor* tensor_isnan_op(const Tensor* t) { Tensor* o=_like(t); for(uint32_t i=0;i<t->size;i++) o->data[i]=isnan(t->data[i])?1.0f:0.0f; return o; }
Tensor* tensor_isinf_op(const Tensor* t) { Tensor* o=_like(t); for(uint32_t i=0;i<t->size;i++) o->data[i]=isinf(t->data[i])?1.0f:0.0f; return o; }
Tensor* tensor_isfinite_op(const Tensor* t) { Tensor* o=_like(t); for(uint32_t i=0;i<t->size;i++) o->data[i]=isfinite(t->data[i])?1.0f:0.0f; return o; }
Tensor* tensor_isposinf(const Tensor* t) { Tensor* o=_like(t); for(uint32_t i=0;i<t->size;i++) o->data[i]=(isinf(t->data[i])&&t->data[i]>0)?1.0f:0.0f; return o; }
Tensor* tensor_isneginf(const Tensor* t) { Tensor* o=_like(t); for(uint32_t i=0;i<t->size;i++) o->data[i]=(isinf(t->data[i])&&t->data[i]<0)?1.0f:0.0f; return o; }

// Element-wise max/min
Tensor* tensor_maximum(const Tensor* a, const Tensor* b) {
    Tensor* out = _like(a);
#if RPITORCH_HAS_NEON
    uint32_t i = 0;
    for (; i+4 <= a->size; i += 4)
        vst1q_f32(&out->data[i], vmaxq_f32(vld1q_f32(&a->data[i]), vld1q_f32(&b->data[i])));
    for (; i < a->size; i++) out->data[i] = fmaxf(a->data[i], b->data[i%b->size]);
#else
    for (uint32_t i = 0; i < a->size; i++) out->data[i] = fmaxf(a->data[i], b->data[i%b->size]);
#endif
    return out;
}
Tensor* tensor_minimum(const Tensor* a, const Tensor* b) {
    Tensor* out = _like(a);
#if RPITORCH_HAS_NEON
    uint32_t i = 0;
    for (; i+4 <= a->size; i += 4)
        vst1q_f32(&out->data[i], vminq_f32(vld1q_f32(&a->data[i]), vld1q_f32(&b->data[i])));
    for (; i < a->size; i++) out->data[i] = fminf(a->data[i], b->data[i%b->size]);
#else
    for (uint32_t i = 0; i < a->size; i++) out->data[i] = fminf(a->data[i], b->data[i%b->size]);
#endif
    return out;
}
// fmax/fmin (NaN-propagating variants — fmax ignores NaN)
Tensor* tensor_fmax(const Tensor* a, const Tensor* b) {
    Tensor* o=_like(a); for(uint32_t i=0;i<a->size;i++) o->data[i]=fmaxf(a->data[i],b->data[i%b->size]); return o;
}
Tensor* tensor_fmin(const Tensor* a, const Tensor* b) {
    Tensor* o=_like(a); for(uint32_t i=0;i<a->size;i++) o->data[i]=fminf(a->data[i],b->data[i%b->size]); return o;
}

// Sort (simple — copies data, sorts with qsort)
static int cmp_asc(const void* a, const void* b) { float d = *(const float*)a - *(const float*)b; return (d>0)-(d<0); }
static int cmp_desc(const void* a, const void* b) { float d = *(const float*)b - *(const float*)a; return (d>0)-(d<0); }

Tensor* tensor_sort_op(const Tensor* t, int32_t dim, bool descending, Tensor** indices_out) {
    if (dim < 0) dim += t->dims;
    Tensor* out = tensor_create(t->dims, t->shape, false);
    memcpy(out->data, t->data, t->size * sizeof(float));
    Tensor* idx = tensor_create(t->dims, t->shape, false);
    
    uint32_t outer=1, inner=1, ds=t->shape[dim];
    for (int32_t d=0; d<dim; d++) outer *= t->shape[d];
    for (uint32_t d=dim+1; d<t->dims; d++) inner *= t->shape[d];
    
    float* buf = (float*)malloc(ds * sizeof(float));
    uint32_t* ibuf = (uint32_t*)malloc(ds * sizeof(uint32_t));
    
    for (uint32_t o = 0; o < outer; o++) {
        for (uint32_t i = 0; i < inner; i++) {
            // Extract slice
            for (uint32_t d = 0; d < ds; d++) { buf[d] = t->data[(o*ds+d)*inner+i]; ibuf[d] = d; }
            // Insertion sort with index tracking
            for (uint32_t j = 1; j < ds; j++) {
                float key = buf[j]; uint32_t ki = ibuf[j]; int32_t k = j-1;
                if (descending) { while (k>=0 && buf[k]<key) { buf[k+1]=buf[k]; ibuf[k+1]=ibuf[k]; k--; } }
                else { while (k>=0 && buf[k]>key) { buf[k+1]=buf[k]; ibuf[k+1]=ibuf[k]; k--; } }
                buf[k+1]=key; ibuf[k+1]=ki;
            }
            for (uint32_t d = 0; d < ds; d++) { out->data[(o*ds+d)*inner+i] = buf[d]; idx->data[(o*ds+d)*inner+i] = (float)ibuf[d]; }
        }
    }
    free(buf); free(ibuf);
    if (indices_out) *indices_out = idx; else tensor_free(idx);
    return out;
}

Tensor* tensor_argsort(const Tensor* t, int32_t dim, bool descending) {
    Tensor* idx;
    Tensor* sorted = tensor_sort_op(t, dim, descending, &idx);
    tensor_free(sorted);
    return idx;
}

Tensor* tensor_topk(const Tensor* t, uint32_t k, int32_t dim, bool largest) {
    Tensor* idx;
    Tensor* sorted = tensor_sort_op(t, dim, largest, &idx);
    if (dim < 0) dim += t->dims;
    // Narrow to k
    uint32_t ns[MAX_DIMS]; memcpy(ns, t->shape, t->dims*sizeof(uint32_t)); ns[dim] = k;
    Tensor* out = tensor_create(t->dims, ns, false);
    
    uint32_t outer=1, inner=1, ds=t->shape[dim];
    for (int32_t d=0; d<dim; d++) outer *= t->shape[d];
    for (uint32_t d=dim+1; d<t->dims; d++) inner *= t->shape[d];
    for (uint32_t o=0; o<outer; o++)
        for (uint32_t i=0; i<inner; i++)
            for (uint32_t d=0; d<k; d++) out->data[(o*k+d)*inner+i] = sorted->data[(o*ds+d)*inner+i];
    
    tensor_free(sorted); tensor_free(idx);
    return out;
}

Tensor* tensor_unique(const Tensor* t, uint32_t* out_count) {
    float* tmp = (float*)malloc(t->size * sizeof(float));
    memcpy(tmp, t->data, t->size * sizeof(float));
    qsort(tmp, t->size, sizeof(float), cmp_asc);
    uint32_t c = 1;
    for (uint32_t i = 1; i < t->size; i++) if (tmp[i] != tmp[i-1]) c++;
    *out_count = c;
    uint32_t s[1] = {c};
    Tensor* out = tensor_create(1, s, false);
    out->data[0] = tmp[0]; uint32_t j = 1;
    for (uint32_t i = 1; i < t->size; i++) if (tmp[i] != tmp[i-1]) out->data[j++] = tmp[i];
    free(tmp);
    return out;
}

// isin
Tensor* tensor_isin(const Tensor* elements, const Tensor* test) {
    Tensor* out = _like(elements);
    for (uint32_t i = 0; i < elements->size; i++) {
        float v = elements->data[i]; bool found = false;
        for (uint32_t j = 0; j < test->size; j++) if (test->data[j] == v) { found = true; break; }
        out->data[i] = found ? 1.0f : 0.0f;
    }
    return out;
}

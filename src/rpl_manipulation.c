/*
 * RPL Tensor Manipulation — reshape, cat, split, index, flip, permute
 */
#include "rpl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

Tensor* tensor_reshape(const Tensor* t, uint32_t nd, const uint32_t* ns) {
    Tensor* out = tensor_create(nd, ns, t->requires_grad);
    memcpy(out->data, t->data, t->size * sizeof(float));
    return out;
}

Tensor* tensor_squeeze(const Tensor* t) {
    uint32_t s[MAX_DIMS]; uint32_t nd = 0;
    for (uint32_t i = 0; i < t->dims; i++) if (t->shape[i] != 1) s[nd++] = t->shape[i];
    if (nd == 0) { s[0] = 1; nd = 1; }
    return tensor_reshape(t, nd, s);
}

Tensor* tensor_unsqueeze(const Tensor* t, int32_t dim) {
    if (dim < 0) dim += t->dims + 1;
    uint32_t s[MAX_DIMS], nd = t->dims + 1;
    for (uint32_t i = 0, j = 0; i < nd; i++) s[i] = (i == (uint32_t)dim) ? 1 : t->shape[j++];
    return tensor_reshape(t, nd, s);
}

Tensor* tensor_flatten(const Tensor* t, int32_t sd, int32_t ed) {
    if (sd < 0) sd += t->dims; if (ed < 0) ed += t->dims;
    uint32_t s[MAX_DIMS], nd = 0;
    for (int32_t i = 0; i < sd; i++) s[nd++] = t->shape[i];
    uint32_t f = 1; for (int32_t i = sd; i <= ed; i++) f *= t->shape[i];
    s[nd++] = f;
    for (uint32_t i = ed + 1; i < t->dims; i++) s[nd++] = t->shape[i];
    return tensor_reshape(t, nd, s);
}

Tensor* tensor_ravel(const Tensor* t) { uint32_t s[1] = {t->size}; return tensor_reshape(t, 1, s); }

Tensor* tensor_t_op(const Tensor* t) {
    if (t->dims != 2) return NULL;
    uint32_t s[2] = {t->shape[1], t->shape[0]};
    Tensor* out = tensor_create(2, s, t->requires_grad);
    for (uint32_t i = 0; i < t->shape[0]; i++)
        for (uint32_t j = 0; j < t->shape[1]; j++)
            out->data[j * t->shape[0] + i] = t->data[i * t->shape[1] + j];
    return out;
}

Tensor* tensor_transpose(const Tensor* t, int32_t d0, int32_t d1) {
    if (d0 < 0) d0 += t->dims; if (d1 < 0) d1 += t->dims;
    uint32_t ns[MAX_DIMS]; memcpy(ns, t->shape, t->dims * sizeof(uint32_t));
    ns[d0] = t->shape[d1]; ns[d1] = t->shape[d0];
    Tensor* out = tensor_create(t->dims, ns, t->requires_grad);
    #pragma omp parallel for
    for (uint32_t idx = 0; idx < t->size; idx++) {
        uint32_t c[MAX_DIMS], tmp = idx;
        for (int d = t->dims-1; d >= 0; d--) { c[d] = tmp % t->shape[d]; tmp /= t->shape[d]; }
        uint32_t t0 = c[d0]; c[d0] = c[d1]; c[d1] = t0;
        uint32_t di = 0, st = 1;
        for (int d = t->dims-1; d >= 0; d--) { di += c[d]*st; st *= ns[d]; }
        out->data[di] = t->data[idx];
    }
    return out;
}

Tensor* tensor_permute(const Tensor* t, const uint32_t* perm) {
    uint32_t ns[MAX_DIMS];
    for (uint32_t i = 0; i < t->dims; i++) ns[i] = t->shape[perm[i]];
    Tensor* out = tensor_create(t->dims, ns, t->requires_grad);
    #pragma omp parallel for
    for (uint32_t idx = 0; idx < t->size; idx++) {
        uint32_t c[MAX_DIMS], tmp = idx;
        for (int d = t->dims-1; d >= 0; d--) { c[d] = tmp % t->shape[d]; tmp /= t->shape[d]; }
        uint32_t di = 0, st = 1;
        for (int d = t->dims-1; d >= 0; d--) { di += c[perm[d]]*st; st *= ns[d]; }
        out->data[di] = t->data[idx];
    }
    return out;
}

Tensor* tensor_movedim(const Tensor* t, int32_t src, int32_t dst) {
    if (src < 0) src += t->dims; if (dst < 0) dst += t->dims;
    uint32_t p[MAX_DIMS], j = 0;
    for (uint32_t i = 0; i < t->dims; i++) if (i != (uint32_t)src) p[j++] = i;
    for (int i = j; i > dst; i--) p[i] = p[i-1];
    p[dst] = src;
    return tensor_permute(t, p);
}

Tensor* tensor_swapaxes(const Tensor* t, int32_t a, int32_t b) { return tensor_transpose(t, a, b); }

Tensor* tensor_cat(const Tensor** ts, uint32_t num, int32_t dim) {
    if (!num) return NULL;
    if (dim < 0) dim += ts[0]->dims;
    uint32_t ns[MAX_DIMS]; memcpy(ns, ts[0]->shape, ts[0]->dims * sizeof(uint32_t));
    uint32_t td = 0; for (uint32_t n = 0; n < num; n++) td += ts[n]->shape[dim];
    ns[dim] = td;
    Tensor* out = tensor_create(ts[0]->dims, ns, false);
    uint32_t outer = 1, inner = 1;
    for (int32_t d = 0; d < dim; d++) outer *= ns[d];
    for (uint32_t d = dim+1; d < ts[0]->dims; d++) inner *= ns[d];
    uint32_t off = 0;
    for (uint32_t n = 0; n < num; n++) {
        uint32_t ds = ts[n]->shape[dim];
        for (uint32_t o = 0; o < outer; o++)
            memcpy(&out->data[(o*td+off)*inner], &ts[n]->data[o*ds*inner], ds*inner*sizeof(float));
        off += ds;
    }
    return out;
}

Tensor* tensor_stack(const Tensor** ts, uint32_t num, int32_t dim) {
    Tensor** us = (Tensor**)malloc(num * sizeof(Tensor*));
    for (uint32_t i = 0; i < num; i++) us[i] = tensor_unsqueeze(ts[i], dim);
    Tensor* out = tensor_cat((const Tensor**)us, num, dim);
    for (uint32_t i = 0; i < num; i++) tensor_free(us[i]);
    free(us); return out;
}

Tensor* tensor_hstack(const Tensor** ts, uint32_t n) { return tensor_cat(ts, n, ts[0]->dims >= 2 ? 1 : 0); }

Tensor* tensor_vstack(const Tensor** ts, uint32_t n) {
    if (ts[0]->dims == 1) {
        Tensor** us = (Tensor**)malloc(n * sizeof(Tensor*));
        for (uint32_t i = 0; i < n; i++) us[i] = tensor_unsqueeze(ts[i], 0);
        Tensor* out = tensor_cat((const Tensor**)us, n, 0);
        for (uint32_t i = 0; i < n; i++) tensor_free(us[i]);
        free(us); return out;
    }
    return tensor_cat(ts, n, 0);
}

Tensor** tensor_chunk(const Tensor* t, uint32_t chunks, int32_t dim, uint32_t* on) {
    if (dim < 0) dim += t->dims;
    uint32_t ds = t->shape[dim], cs = (ds+chunks-1)/chunks, actual = (ds+cs-1)/cs;
    *on = actual;
    Tensor** r = (Tensor**)malloc(actual * sizeof(Tensor*));
    uint32_t outer = 1, inner = 1;
    for (int32_t d = 0; d < dim; d++) outer *= t->shape[d];
    for (uint32_t d = dim+1; d < t->dims; d++) inner *= t->shape[d];
    uint32_t off = 0;
    for (uint32_t c = 0; c < actual; c++) {
        uint32_t sz = (off+cs<=ds)?cs:(ds-off);
        uint32_t s[MAX_DIMS]; memcpy(s, t->shape, t->dims*sizeof(uint32_t)); s[dim] = sz;
        r[c] = tensor_create(t->dims, s, false);
        for (uint32_t o = 0; o < outer; o++)
            memcpy(&r[c]->data[o*sz*inner], &t->data[(o*ds+off)*inner], sz*inner*sizeof(float));
        off += sz;
    }
    return r;
}

Tensor** tensor_split(const Tensor* t, uint32_t sec, int32_t dim, uint32_t* on) {
    return tensor_chunk(t, sec, dim, on);
}

Tensor* tensor_index_select(const Tensor* t, int32_t dim, const uint32_t* idx, uint32_t ni) {
    if (dim < 0) dim += t->dims;
    uint32_t ns[MAX_DIMS]; memcpy(ns, t->shape, t->dims*sizeof(uint32_t)); ns[dim] = ni;
    Tensor* out = tensor_create(t->dims, ns, false);
    uint32_t outer = 1, inner = 1;
    for (int32_t d = 0; d < dim; d++) outer *= t->shape[d];
    for (uint32_t d = dim+1; d < t->dims; d++) inner *= t->shape[d];
    for (uint32_t o = 0; o < outer; o++)
        for (uint32_t i = 0; i < ni; i++)
            memcpy(&out->data[(o*ni+i)*inner], &t->data[(o*t->shape[dim]+idx[i])*inner], inner*sizeof(float));
    return out;
}

Tensor* tensor_gather(const Tensor* t, int32_t dim, const Tensor* index) {
    if (dim < 0) dim += t->dims;
    Tensor* out = tensor_create(index->dims, index->shape, false);
    uint32_t ss[MAX_DIMS]; ss[t->dims-1] = 1;
    for (int i = t->dims-2; i >= 0; i--) ss[i] = ss[i+1]*t->shape[i+1];
    #pragma omp parallel for
    for (uint32_t i = 0; i < index->size; i++) {
        uint32_t c[MAX_DIMS], tmp = i;
        for (int d = index->dims-1; d >= 0; d--) { c[d] = tmp % index->shape[d]; tmp /= index->shape[d]; }
        c[dim] = (uint32_t)index->data[i];
        uint32_t si = 0; for (uint32_t d = 0; d < t->dims; d++) si += c[d]*ss[d];
        out->data[i] = t->data[si];
    }
    return out;
}

Tensor* tensor_where_cond(const Tensor* cond, const Tensor* x, const Tensor* y) {
    Tensor* out = tensor_create(x->dims, x->shape, false);
    #pragma omp parallel for
    for (uint32_t i = 0; i < x->size; i++)
        out->data[i] = (cond->data[i] != 0.0f) ? x->data[i] : y->data[i % y->size];
    return out;
}

Tensor* tensor_masked_select(const Tensor* t, const Tensor* mask, uint32_t* os) {
    uint32_t c = 0;
    for (uint32_t i = 0; i < t->size; i++) if (mask->data[i] != 0.0f) c++;
    *os = c; uint32_t s[1] = {c};
    Tensor* out = tensor_create(1, s, false);
    uint32_t j = 0;
    for (uint32_t i = 0; i < t->size; i++) if (mask->data[i] != 0.0f) out->data[j++] = t->data[i];
    return out;
}

Tensor* tensor_nonzero_indices(const Tensor* t, uint32_t* count) {
    uint32_t cnt = 0;
    for (uint32_t i = 0; i < t->size; i++) if (t->data[i] != 0.0f) cnt++;
    *count = cnt;
    uint32_t s[2] = {cnt, t->dims};
    Tensor* out = tensor_create(2, s, false);
    uint32_t idx = 0;
    for (uint32_t i = 0; i < t->size; i++) {
        if (t->data[i] != 0.0f) {
            uint32_t tmp = i;
            for (int d = t->dims-1; d >= 0; d--) { out->data[idx*t->dims+d] = (float)(tmp%t->shape[d]); tmp /= t->shape[d]; }
            idx++;
        }
    }
    return out;
}

Tensor* tensor_flip(const Tensor* t, const int32_t* dims, uint32_t nd) {
    Tensor* out = tensor_create(t->dims, t->shape, false);
    #pragma omp parallel for
    for (uint32_t i = 0; i < t->size; i++) {
        uint32_t c[MAX_DIMS], tmp = i;
        for (int d = t->dims-1; d >= 0; d--) { c[d] = tmp%t->shape[d]; tmp /= t->shape[d]; }
        for (uint32_t f = 0; f < nd; f++) { int32_t fd = dims[f]<0?dims[f]+t->dims:dims[f]; c[fd] = t->shape[fd]-1-c[fd]; }
        uint32_t di = 0, st = 1;
        for (int d = t->dims-1; d >= 0; d--) { di += c[d]*st; st *= t->shape[d]; }
        out->data[di] = t->data[i];
    }
    return out;
}

Tensor* tensor_fliplr(const Tensor* t) { int32_t d=1; return tensor_flip(t,&d,1); }
Tensor* tensor_flipud(const Tensor* t) { int32_t d=0; return tensor_flip(t,&d,1); }

Tensor* tensor_roll(const Tensor* t, int32_t shift, int32_t dim) {
    Tensor* out = tensor_create(t->dims, t->shape, false);
    if (dim < 0) dim += t->dims;
    uint32_t outer=1, inner=1;
    for (int32_t d = 0; d < dim; d++) outer *= t->shape[d];
    for (uint32_t d = dim+1; d < t->dims; d++) inner *= t->shape[d];
    uint32_t ds = t->shape[dim];
    for (uint32_t o = 0; o < outer; o++)
        for (uint32_t i = 0; i < ds; i++) {
            uint32_t si = (uint32_t)(((int32_t)i - shift) % (int32_t)ds + (int32_t)ds) % ds;
            memcpy(&out->data[(o*ds+i)*inner], &t->data[(o*ds+si)*inner], inner*sizeof(float));
        }
    return out;
}

Tensor* tensor_clone(const Tensor* t) {
    Tensor* out = tensor_create(t->dims, t->shape, t->requires_grad);
    memcpy(out->data, t->data, t->size * sizeof(float));
    return out;
}

Tensor* tensor_tile(const Tensor* t, const uint32_t* reps, uint32_t nr) {
    uint32_t ns[MAX_DIMS];
    for (uint32_t i = 0; i < t->dims; i++) ns[i] = t->shape[i] * (i < nr ? reps[i] : 1);
    Tensor* out = tensor_create(t->dims, ns, false);
    #pragma omp parallel for
    for (uint32_t i = 0; i < out->size; i++) {
        uint32_t c[MAX_DIMS], tmp = i;
        for (int d = out->dims-1; d >= 0; d--) { c[d] = tmp % ns[d]; tmp /= ns[d]; }
        uint32_t si = 0, st = 1;
        for (int d = t->dims-1; d >= 0; d--) { si += (c[d]%t->shape[d])*st; st *= t->shape[d]; }
        out->data[i] = t->data[si];
    }
    return out;
}

Tensor* tensor_narrow(const Tensor* t, int32_t dim, uint32_t start, uint32_t len) {
    if (dim < 0) dim += t->dims;
    uint32_t ns[MAX_DIMS]; memcpy(ns, t->shape, t->dims*sizeof(uint32_t)); ns[dim] = len;
    Tensor* out = tensor_create(t->dims, ns, false);
    uint32_t outer=1, inner=1;
    for (int32_t d=0; d<dim; d++) outer *= t->shape[d];
    for (uint32_t d=dim+1; d<t->dims; d++) inner *= t->shape[d];
    for (uint32_t o = 0; o < outer; o++)
        memcpy(&out->data[o*len*inner], &t->data[(o*t->shape[dim]+start)*inner], len*inner*sizeof(float));
    return out;
}

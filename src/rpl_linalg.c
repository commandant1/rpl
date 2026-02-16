/*
 * RPL Linear Algebra — dot, mm, bmm, cholesky, svd, det, etc.
 */
#include "rpl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

static inline Tensor* _like(const Tensor* t) { return tensor_create(t->dims, t->shape, false); }

// Dot product (1D)
float tensor_dot(const Tensor* a, const Tensor* b) {
    float s = 0;
#if RPITORCH_HAS_NEON
    float32x4_t vs = vdupq_n_f32(0);
    uint32_t i = 0;
    for (; i+4 <= a->size; i += 4)
        vs = vfmaq_f32(vs, vld1q_f32(&a->data[i]), vld1q_f32(&b->data[i]));
    s = vgetq_lane_f32(vs,0)+vgetq_lane_f32(vs,1)+vgetq_lane_f32(vs,2)+vgetq_lane_f32(vs,3);
    for (; i < a->size; i++) s += a->data[i]*b->data[i];
#else
    for (uint32_t i = 0; i < a->size; i++) s += a->data[i]*b->data[i];
#endif
    return s;
}

float tensor_vdot(const Tensor* a, const Tensor* b) { return tensor_dot(a, b); }

// Inner product
Tensor* tensor_inner(const Tensor* a, const Tensor* b) {
    if (a->dims == 1 && b->dims == 1) {
        uint32_t s[1] = {1};
        Tensor* out = tensor_create(1, s, false);
        out->data[0] = tensor_dot(a, b);
        return out;
    }
    // General: contract last dims
    return tensor_matmul(a, b);
}

// Outer product
Tensor* tensor_outer(const Tensor* a, const Tensor* b) {
    uint32_t s[2] = {a->size, b->size};
    Tensor* out = tensor_create(2, s, false);
    #pragma omp parallel for
    for (uint32_t i = 0; i < a->size; i++)
        for (uint32_t j = 0; j < b->size; j++)
            out->data[i*b->size+j] = a->data[i]*b->data[j];
    return out;
}

// mm (2D matmul)
Tensor* tensor_mm(const Tensor* a, const Tensor* b) { return tensor_matmul(a, b); }

// mv (matrix-vector)
Tensor* tensor_mv(const Tensor* mat, const Tensor* vec) {
    uint32_t M = mat->shape[0], K = mat->shape[1], s[1] = {M};
    Tensor* out = tensor_create(1, s, false);
    #pragma omp parallel for
    for (uint32_t i = 0; i < M; i++) {
        float sum = 0;
        for (uint32_t k = 0; k < K; k++) sum += mat->data[i*K+k]*vec->data[k];
        out->data[i] = sum;
    }
    return out;
}

// bmm (batched matmul)
Tensor* tensor_bmm(const Tensor* a, const Tensor* b) {
    uint32_t B = a->shape[0], M = a->shape[1], K = a->shape[2], N = b->shape[2];
    uint32_t s[3] = {B, M, N};
    Tensor* out = tensor_create(3, s, false);
    #pragma omp parallel for
    for (uint32_t batch = 0; batch < B; batch++) {
        const float* A = &a->data[batch*M*K];
        const float* Bp = &b->data[batch*K*N];
        float* C = &out->data[batch*M*N];
        for (uint32_t i = 0; i < M; i++)
            for (uint32_t j = 0; j < N; j++) {
                float sum = 0;
                for (uint32_t k = 0; k < K; k++) sum += A[i*K+k]*Bp[k*N+j];
                C[i*N+j] = sum;
            }
    }
    return out;
}

// addmm: beta*input + alpha*(mat1 @ mat2)
Tensor* tensor_addmm(const Tensor* input, const Tensor* m1, const Tensor* m2, float beta, float alpha) {
    Tensor* mm = tensor_matmul(m1, m2);
    Tensor* out = tensor_create(mm->dims, mm->shape, false);
    for (uint32_t i = 0; i < mm->size; i++)
        out->data[i] = beta * input->data[i % input->size] + alpha * mm->data[i];
    tensor_free(mm);
    return out;
}

// addr: beta*input + alpha*(vec1 outer vec2)
Tensor* tensor_addr(const Tensor* input, const Tensor* v1, const Tensor* v2, float beta, float alpha) {
    Tensor* op = tensor_outer(v1, v2);
    Tensor* out = tensor_create(op->dims, op->shape, false);
    for (uint32_t i = 0; i < op->size; i++)
        out->data[i] = beta * input->data[i % input->size] + alpha * op->data[i];
    tensor_free(op);
    return out;
}

// Trace
float tensor_trace(const Tensor* t) {
    uint32_t n = (t->shape[0] < t->shape[1]) ? t->shape[0] : t->shape[1];
    float s = 0;
    for (uint32_t i = 0; i < n; i++) s += t->data[i*t->shape[1]+i];
    return s;
}

// Diag — extract diagonal or create diagonal matrix
Tensor* tensor_diag(const Tensor* t, int32_t diagonal) {
    if (t->dims == 1) {
        uint32_t n = t->size + (diagonal < 0 ? -diagonal : diagonal);
        uint32_t s[2] = {n, n};
        Tensor* out = tensor_create(2, s, false);
        tensor_fill(out, 0);
        uint32_t off = diagonal >= 0 ? diagonal : (-diagonal) * n;
        for (uint32_t i = 0; i < t->size; i++) out->data[off + i*(n+1)] = t->data[i];
        return out;
    } else {
        uint32_t r = t->shape[0], c = t->shape[1];
        int32_t d = diagonal;
        uint32_t len = 0;
        if (d >= 0) len = ((r < c-d) ? r : c-d);
        else len = ((r+d < c) ? r+d : c);
        uint32_t s[1] = {len};
        Tensor* out = tensor_create(1, s, false);
        for (uint32_t i = 0; i < len; i++) {
            uint32_t ri = (d >= 0) ? i : i - d;
            uint32_t ci = (d >= 0) ? i + d : i;
            out->data[i] = t->data[ri*c + ci];
        }
        return out;
    }
}

// Tril / Triu
Tensor* tensor_tril(const Tensor* t, int32_t diagonal) {
    Tensor* out = _like(t);
    uint32_t r = t->shape[0], c = t->shape[1];
    for (uint32_t i = 0; i < r; i++)
        for (uint32_t j = 0; j < c; j++)
            out->data[i*c+j] = ((int32_t)j <= (int32_t)i + diagonal) ? t->data[i*c+j] : 0.0f;
    return out;
}

Tensor* tensor_triu(const Tensor* t, int32_t diagonal) {
    Tensor* out = _like(t);
    uint32_t r = t->shape[0], c = t->shape[1];
    for (uint32_t i = 0; i < r; i++)
        for (uint32_t j = 0; j < c; j++)
            out->data[i*c+j] = ((int32_t)j >= (int32_t)i + diagonal) ? t->data[i*c+j] : 0.0f;
    return out;
}

// Eye (identity matrix)
Tensor* tensor_eye(uint32_t n) {
    uint32_t s[2] = {n, n};
    Tensor* out = tensor_create(2, s, false);
    tensor_fill(out, 0);
    for (uint32_t i = 0; i < n; i++) out->data[i*n+i] = 1.0f;
    return out;
}

// Cross product (3D vectors)
Tensor* tensor_cross(const Tensor* a, const Tensor* b) {
    uint32_t s[1] = {3};
    Tensor* out = tensor_create(1, s, false);
    out->data[0] = a->data[1]*b->data[2] - a->data[2]*b->data[1];
    out->data[1] = a->data[2]*b->data[0] - a->data[0]*b->data[2];
    out->data[2] = a->data[0]*b->data[1] - a->data[1]*b->data[0];
    return out;
}

// Determinant (2x2 and 3x3, LU for larger)
float tensor_det(const Tensor* t) {
    uint32_t n = t->shape[0];
    if (n == 1) return t->data[0];
    if (n == 2) return t->data[0]*t->data[3] - t->data[1]*t->data[2];
    if (n == 3) {
        const float *d = t->data;
        return d[0]*(d[4]*d[8]-d[5]*d[7]) - d[1]*(d[3]*d[8]-d[5]*d[6]) + d[2]*(d[3]*d[7]-d[4]*d[6]);
    }
    // LU decomposition for general case
    float* lu = (float*)malloc(n*n*sizeof(float));
    memcpy(lu, t->data, n*n*sizeof(float));
    float det = 1.0f;
    for (uint32_t i = 0; i < n; i++) {
        // Partial pivoting
        float max_val = fabsf(lu[i*n+i]);
        uint32_t max_row = i;
        for (uint32_t k = i+1; k < n; k++) if (fabsf(lu[k*n+i]) > max_val) { max_val = fabsf(lu[k*n+i]); max_row = k; }
        if (max_row != i) {
            for (uint32_t j = 0; j < n; j++) { float tmp = lu[i*n+j]; lu[i*n+j] = lu[max_row*n+j]; lu[max_row*n+j] = tmp; }
            det = -det;
        }
        if (fabsf(lu[i*n+i]) < 1e-12f) { free(lu); return 0.0f; }
        det *= lu[i*n+i];
        for (uint32_t k = i+1; k < n; k++) {
            lu[k*n+i] /= lu[i*n+i];
            for (uint32_t j = i+1; j < n; j++) lu[k*n+j] -= lu[k*n+i]*lu[i*n+j];
        }
    }
    free(lu);
    return det;
}

// Inverse (Gauss-Jordan)
Tensor* tensor_inverse(const Tensor* t) {
    uint32_t n = t->shape[0];
    float* aug = (float*)calloc(n*2*n, sizeof(float));
    for (uint32_t i = 0; i < n; i++) {
        memcpy(&aug[i*2*n], &t->data[i*n], n*sizeof(float));
        aug[i*2*n+n+i] = 1.0f;
    }
    for (uint32_t i = 0; i < n; i++) {
        // Pivot
        uint32_t mr = i; float mv = fabsf(aug[i*2*n+i]);
        for (uint32_t k=i+1;k<n;k++) if(fabsf(aug[k*2*n+i])>mv){mv=fabsf(aug[k*2*n+i]);mr=k;}
        if (mr != i) for (uint32_t j=0;j<2*n;j++){float t2=aug[i*2*n+j];aug[i*2*n+j]=aug[mr*2*n+j];aug[mr*2*n+j]=t2;}
        float piv = aug[i*2*n+i];
        for (uint32_t j=0;j<2*n;j++) aug[i*2*n+j] /= piv;
        for (uint32_t k=0;k<n;k++) if(k!=i){float f=aug[k*2*n+i]; for(uint32_t j=0;j<2*n;j++) aug[k*2*n+j]-=f*aug[i*2*n+j];}
    }
    uint32_t s[2] = {n, n};
    Tensor* out = tensor_create(2, s, false);
    for (uint32_t i = 0; i < n; i++) memcpy(&out->data[i*n], &aug[i*2*n+n], n*sizeof(float));
    free(aug);
    return out;
}

// Cholesky decomposition (lower triangular)
Tensor* tensor_cholesky(const Tensor* t) {
    uint32_t n = t->shape[0];
    uint32_t s[2] = {n, n};
    Tensor* out = tensor_create(2, s, false);
    tensor_fill(out, 0);
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j <= i; j++) {
            float sum = 0;
            for (uint32_t k = 0; k < j; k++) sum += out->data[i*n+k]*out->data[j*n+k];
            if (i == j) out->data[i*n+j] = sqrtf(t->data[i*n+i] - sum);
            else out->data[i*n+j] = (t->data[i*n+j] - sum) / out->data[j*n+j];
        }
    }
    return out;
}

// Matrix power
Tensor* tensor_matrix_power(const Tensor* t, int32_t n) {
    uint32_t sz = t->shape[0];
    Tensor* result = tensor_eye(sz);
    if (n == 0) return result;
    
    Tensor* base;
    if (n < 0) { base = tensor_inverse(t); n = -n; }
    else { base = tensor_create(t->dims, t->shape, false); memcpy(base->data, t->data, t->size*sizeof(float)); }
    
    for (int32_t i = 0; i < n; i++) {
        Tensor* tmp = tensor_matmul(result, base);
        tensor_free(result); result = tmp;
    }
    tensor_free(base);
    return result;
}

// Kron product
Tensor* tensor_kron(const Tensor* a, const Tensor* b) {
    uint32_t ar = a->shape[0], ac = a->shape[1], br = b->shape[0], bc = b->shape[1];
    uint32_t s[2] = {ar*br, ac*bc};
    Tensor* out = tensor_create(2, s, false);
    for (uint32_t i = 0; i < ar; i++)
        for (uint32_t j = 0; j < ac; j++)
            for (uint32_t k = 0; k < br; k++)
                for (uint32_t l = 0; l < bc; l++)
                    out->data[(i*br+k)*s[1]+(j*bc+l)] = a->data[i*ac+j]*b->data[k*bc+l];
    return out;
}

// Tensordot
Tensor* tensor_tensordot(const Tensor* a, const Tensor* b, uint32_t dims_to_contract) {
    // Simple: contract last `dims_to_contract` of a with first `dims_to_contract` of b
    uint32_t K = 1;
    for (uint32_t i = 0; i < dims_to_contract; i++) K *= a->shape[a->dims-1-i];
    uint32_t M = a->size / K, N = b->size / K;
    uint32_t s[2] = {M, N};
    Tensor* out = tensor_create(2, s, false);
    #pragma omp parallel for
    for (uint32_t i = 0; i < M; i++)
        for (uint32_t j = 0; j < N; j++) {
            float sum = 0;
            for (uint32_t k = 0; k < K; k++) sum += a->data[i*K+k]*b->data[k*N+j];
            out->data[i*N+j] = sum;
        }
    return out;
}

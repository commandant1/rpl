/*
 * RPiTorch Core Implementation
 * Highly optimized for Cortex-A72 with OpenBLAS, OpenMP, NEON
 */

#include "rpl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <float.h>
#include <assert.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Forward declarations
void tensor_gemm_large(Tensor* C, const Tensor* A, const Tensor* B);
void tensor_gemm_small(Tensor* C, const Tensor* A, const Tensor* B);
void backward_add(Tensor* t);
void backward_mul(Tensor* t);
void backward_matmul(Tensor* t);
void backward_relu(Tensor* t);
void backward_sigmoid(Tensor* t);
void backward_mse(Tensor* t);
void parallel_gemm_optimized(const float* A, const float* B, float* C, uint32_t M, uint32_t N, uint32_t K);
void parallel_gemm_optimized_trans(const float* A, const float* B, float* C, uint32_t M, uint32_t N, uint32_t K, bool trans_a, bool trans_b);
void conv2d_winograd_3x3(const float* input, const float* kernel, float* output,
                          int in_channels, int out_channels,
                          int height, int width, int stride, int padding);

// ============================================================
// Memory Management — Thread-Local Slab Pool
// ============================================================
// Bins: 64B, 128B, 256B, 512B, 1KB, 2KB, 4KB, 8KB, 16KB, 32KB, 64KB
// Above 64KB: fall through to posix_memalign (rare for ML workloads)

void* rpitorch_aligned_alloc(size_t alignment, size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) return NULL;
    return ptr;
}

void rpitorch_aligned_free(void* ptr) {
    free(ptr);
}

#define POOL_NUM_BINS 11
#define POOL_MAX_CACHED 64   // max slabs per bin per thread
#define POOL_MIN_SIZE 64     // 64B = 1 cache line
#define POOL_MAX_SIZE 65536  // 64KB

typedef struct {
    void* slabs[POOL_MAX_CACHED];
    int count;
    size_t slab_size;
} PoolBin;

static __thread PoolBin pool_bins[POOL_NUM_BINS];
static __thread int pool_initialized = 0;

static void pool_init(void) {
    size_t sz = POOL_MIN_SIZE;
    for (int i = 0; i < POOL_NUM_BINS; i++) {
        pool_bins[i].count = 0;
        pool_bins[i].slab_size = sz;
        sz <<= 1;
    }
    pool_initialized = 1;
}

// Find bin index for a given size (or -1 if too large)
static inline int pool_bin_index(size_t size) {
    if (size > POOL_MAX_SIZE) return -1;
    size_t s = POOL_MIN_SIZE;
    for (int i = 0; i < POOL_NUM_BINS; i++) {
        if (size <= s) return i;
        s <<= 1;
    }
    return -1;
}

void* pool_alloc(size_t size) {
    if (!pool_initialized) pool_init();
    int bin = pool_bin_index(size);
    if (bin >= 0 && pool_bins[bin].count > 0) {
        // Hot path: pop from free list (no syscall)
        return pool_bins[bin].slabs[--pool_bins[bin].count];
    }
    // Cold path: allocate from system
    size_t actual = (bin >= 0) ? pool_bins[bin].slab_size : size;
    return rpitorch_aligned_alloc(64, actual);
}

void pool_free(void* ptr, size_t size) {
    if (!ptr) return;
    if (!pool_initialized) { free(ptr); return; }
    int bin = pool_bin_index(size);
    if (bin >= 0 && pool_bins[bin].count < POOL_MAX_CACHED) {
        // Hot path: return to free list (no syscall)
        pool_bins[bin].slabs[pool_bins[bin].count++] = ptr;
        return;
    }
    free(ptr);
}

Tensor* tensor_create(uint32_t dims, const uint32_t* shape, bool requires_grad) {
    Tensor* t = (Tensor*)calloc(1, sizeof(Tensor));
    if (!t) return NULL;
    
    t->dims = dims;
    t->size = 1;
    for (uint32_t i = 0; i < dims; i++) {
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }
    
    // Compute strides
    t->strides[dims-1] = 1;
    for (int i = dims-2; i >= 0; i--) t->strides[i] = t->strides[i+1] * t->shape[i+1];
    
    // Round up to cache line (64B)
    size_t alloc_size = (t->size * sizeof(float) + 63) & ~(size_t)63;
    t->_alloc_size = alloc_size;
    t->_allocation = pool_alloc(alloc_size);
    if (!t->_allocation) {
        free(t);
        return NULL;
    }
    t->data = (float*)t->_allocation;
    
    if (requires_grad) {
        t->grad = (float*)pool_alloc(alloc_size);
        if (t->grad) {
            memset(t->grad, 0, alloc_size);
        }
    }
    
    t->requires_grad = requires_grad;
    t->is_leaf = true;
    t->_refcount = 1;
    t->_op = OP_NONE;
    return t;
}

void tensor_free(Tensor* t) {
    if (!t) return;
#ifdef USE_GPU
    tensor_free_gpu(t);
#endif
    for (int i = 0; i < t->_n_parents; i++) {
        if (t->_parents[i]) {
            tensor_release(t->_parents[i]);
            t->_parents[i] = NULL;
        }
    }
    size_t alloc_size = t->_alloc_size;
    if (t->_allocation) pool_free(t->_allocation, alloc_size);
    if (t->grad) pool_free(t->grad, alloc_size);
    free(t);
}

// ============================================================
// Basic Operations
// ============================================================

void tensor_fill(Tensor* t, float value) {
#if RPITORCH_HAS_NEON
    float32x4_t vval = vdupq_n_f32(value);
    uint32_t i = 0;
    for (; i + 16 <= t->size; i += 16) {
        vst1q_f32(&t->data[i], vval);
        vst1q_f32(&t->data[i+4], vval);
        vst1q_f32(&t->data[i+8], vval);
        vst1q_f32(&t->data[i+12], vval);
    }
    for (; i + 4 <= t->size; i += 4) vst1q_f32(&t->data[i], vval);
    for (; i < t->size; i++) t->data[i] = value;
#else
    #pragma omp parallel for
    for (uint32_t i = 0; i < t->size; i++) t->data[i] = value;
#endif
}

void tensor_randomize(Tensor* t) {
    #pragma omp parallel for if(t->size >= RPL_OMP_THRESHOLD)
    for (uint32_t i = 0; i < t->size; i++) {
        t->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
}

void tensor_add_out(Tensor* out, const Tensor* a, const Tensor* b) {
    if (a->size == b->size && out->size == a->size) {
#ifdef USE_GPU
        if ((a->device == DEVICE_GPU || b->device == DEVICE_GPU || out->device == DEVICE_GPU)
                && RPL_GPU_PREFERABLE(out->size)) {
            tensor_add_gpu(out, a, b);
            goto finalize;
        }
        // Below threshold: pull data back to CPU
        if (a->device == DEVICE_GPU) tensor_from_gpu((Tensor*)a);
        if (b->device == DEVICE_GPU) tensor_from_gpu((Tensor*)b);
        if (out->device == DEVICE_GPU) tensor_from_gpu(out);
#endif

#if RPITORCH_HAS_NEON
        const float* restrict pa = a->data;
        const float* restrict pb = b->data;
        float* restrict po = out->data;
        
        #pragma omp parallel for if(out->size >= RPL_OMP_THRESHOLD)
        for (uint32_t base = 0; base < out->size; base += 512) {
            uint32_t end = (base + 512 < out->size) ? base + 512 : out->size;
            uint32_t i = base;
            
            // Prefetch next block
            __builtin_prefetch(&pa[base + 512], 0, 0);
            __builtin_prefetch(&pb[base + 512], 0, 0);
            
            // 16-wide processing
            for (; i + 16 <= end; i += 16) {
                float32x4_t a0 = vld1q_f32(&pa[i]);
                float32x4_t a1 = vld1q_f32(&pa[i+4]);
                float32x4_t a2 = vld1q_f32(&pa[i+8]);
                float32x4_t a3 = vld1q_f32(&pa[i+12]);
                float32x4_t b0 = vld1q_f32(&pb[i]);
                float32x4_t b1 = vld1q_f32(&pb[i+4]);
                float32x4_t b2 = vld1q_f32(&pb[i+8]);
                float32x4_t b3 = vld1q_f32(&pb[i+12]);
                vst1q_f32(&po[i], vaddq_f32(a0, b0));
                vst1q_f32(&po[i+4], vaddq_f32(a1, b1));
                vst1q_f32(&po[i+8], vaddq_f32(a2, b2));
                vst1q_f32(&po[i+12], vaddq_f32(a3, b3));
            }
            for (; i + 4 <= end; i += 4) {
                vst1q_f32(&po[i], vaddq_f32(vld1q_f32(&pa[i]), vld1q_f32(&pb[i])));
            }
            for (; i < end; i++) po[i] = pa[i] + pb[i];
        }
#else
        #pragma omp parallel for if(out->size >= RPL_OMP_THRESHOLD)
        for (uint32_t i = 0; i < out->size; i++) out->data[i] = a->data[i] + b->data[i];
#endif
    } else {
#ifdef USE_GPU
        tensor_from_gpu((Tensor*)a);
        tensor_from_gpu((Tensor*)b);
        tensor_from_gpu(out);
#endif
        #pragma omp parallel for
        for (uint32_t i = 0; i < out->size; i++) {
            out->data[i] = a->data[i] + b->data[i % b->size];
        }
    }

finalize:

    if (rpl_is_grad_enabled() && out->requires_grad) {
        out->parent1 = (void*)a;
        out->parent2 = (void*)b;
        out->backward_fn = backward_add;
        out->is_leaf = false;
        out->_op = OP_ADD;
        out->_parents[0] = (Tensor*)a;
        out->_parents[1] = (Tensor*)b;
        out->_n_parents = 2;
        tensor_retain((Tensor*)a);
        tensor_retain((Tensor*)b);
    } else {
        out->requires_grad = false;
    }
}

Tensor* tensor_add(const Tensor* a, const Tensor* b) {
    Tensor* out = tensor_create(a->dims, a->shape, a->requires_grad || b->requires_grad);
    tensor_add_out(out, a, b);
    return out;
}

void tensor_mul_out(Tensor* out, const Tensor* a, const Tensor* b) {
    if (a->size == b->size && out->size == a->size) {
#ifdef USE_GPU
        if ((a->device == DEVICE_GPU || b->device == DEVICE_GPU || out->device == DEVICE_GPU)
                && RPL_GPU_PREFERABLE(out->size)) {
            tensor_mul_gpu(out, a, b);
            goto finalize;
        }
        // Below threshold: pull data back to CPU
        if (a->device == DEVICE_GPU) tensor_from_gpu((Tensor*)a);
        if (b->device == DEVICE_GPU) tensor_from_gpu((Tensor*)b);
        if (out->device == DEVICE_GPU) tensor_from_gpu(out);
#endif

#if RPITORCH_HAS_NEON
        const float* restrict pa = a->data;
        const float* restrict pb = b->data;
        float* restrict po = out->data;
        
        #pragma omp parallel for if(out->size >= RPL_OMP_THRESHOLD)
        for (uint32_t base = 0; base < out->size; base += 512) {
            uint32_t end = (base + 512 < out->size) ? base + 512 : out->size;
            uint32_t i = base;
            
            __builtin_prefetch(&pa[base + 512], 0, 0);
            __builtin_prefetch(&pb[base + 512], 0, 0);
            
            for (; i + 16 <= end; i += 16) {
                float32x4_t a0 = vld1q_f32(&pa[i]);
                float32x4_t a1 = vld1q_f32(&pa[i+4]);
                float32x4_t a2 = vld1q_f32(&pa[i+8]);
                float32x4_t a3 = vld1q_f32(&pa[i+12]);
                float32x4_t b0 = vld1q_f32(&pb[i]);
                float32x4_t b1 = vld1q_f32(&pb[i+4]);
                float32x4_t b2 = vld1q_f32(&pb[i+8]);
                float32x4_t b3 = vld1q_f32(&pb[i+12]);
                vst1q_f32(&po[i], vmulq_f32(a0, b0));
                vst1q_f32(&po[i+4], vmulq_f32(a1, b1));
                vst1q_f32(&po[i+8], vmulq_f32(a2, b2));
                vst1q_f32(&po[i+12], vmulq_f32(a3, b3));
            }
            for (; i + 4 <= end; i += 4) {
                vst1q_f32(&po[i], vmulq_f32(vld1q_f32(&pa[i]), vld1q_f32(&pb[i])));
            }
            for (; i < end; i++) po[i] = pa[i] * pb[i];
        }
#else
        #pragma omp parallel for
        for (uint32_t i = 0; i < out->size; i++) out->data[i] = a->data[i] * b->data[i];
#endif
    } else {
#ifdef USE_GPU
        tensor_from_gpu((Tensor*)a);
        tensor_from_gpu((Tensor*)b);
        tensor_from_gpu(out);
#endif
        #pragma omp parallel for
        for (uint32_t i = 0; i < out->size; i++) {
            out->data[i] = a->data[i] * b->data[i % b->size];
        }
    }

finalize:

    if (rpl_is_grad_enabled() && out->requires_grad) {
        out->parent1 = (void*)a;
        out->parent2 = (void*)b;
        out->backward_fn = backward_mul;
        out->is_leaf = false;
        out->_op = OP_MUL;
        out->_parents[0] = (Tensor*)a;
        out->_parents[1] = (Tensor*)b;
        out->_n_parents = 2;
        tensor_retain((Tensor*)a);
        tensor_retain((Tensor*)b);
    } else {
        out->requires_grad = false;
    }
}

Tensor* tensor_mul(const Tensor* a, const Tensor* b) {
    Tensor* out = tensor_create(a->dims, a->shape, a->requires_grad || b->requires_grad);
    tensor_mul_out(out, a, b);
    return out;
}

Tensor* tensor_matmul(const Tensor* a, const Tensor* b) {
    uint32_t M = a->shape[a->dims-2];
    uint32_t N = b->shape[b->dims-1];
    uint32_t shape[2] = {M, N};
    Tensor* out = tensor_create(2, shape, a->requires_grad || b->requires_grad);
    tensor_fill(out, 0.0f);
    
    tensor_gemm(out, a, b, 1.0f, 0.0f, false, false);
    
    if (rpl_is_grad_enabled() && out->requires_grad) {
        out->parent1 = (void*)a;
        out->parent2 = (void*)b;
        out->backward_fn = backward_matmul;
        out->is_leaf = false;
        out->_op = OP_MATMUL;
        out->_parents[0] = (Tensor*)a;
        out->_parents[1] = (Tensor*)b;
        out->_n_parents = 2;
        tensor_retain((Tensor*)a);
        tensor_retain((Tensor*)b);
    } else {
        out->requires_grad = false;
    }
    return out;
}

// ============================================================
// Optimized GEMM
// ============================================================

void tensor_gemm(Tensor* C, const Tensor* A, const Tensor* B,
                float alpha, float beta, bool trans_a, bool trans_b) {
    uint32_t M = trans_a ? A->shape[1] : A->shape[0];
    uint32_t K = trans_a ? A->shape[0] : A->shape[1];
    uint32_t N = trans_b ? B->shape[0] : B->shape[1];

#ifdef USE_GPU
    // FLOPs-based threshold: GPU wins only when M*N*K exceeds the kernel-launch
    // overhead (~100-500µs on VideoCore VI).  Handles ALL trans/alpha/beta variants.
    if (RPL_GPU_GEMM_PREFERABLE(M, N, K) &&
        (A->device == DEVICE_GPU || B->device == DEVICE_GPU || C->device == DEVICE_GPU)) {
        tensor_gemm_gpu(C, A, B, M, N, K, alpha, beta, trans_a, trans_b);
        return;
    }
    // Below FLOPs threshold or no GPU tensor — pull data to CPU.
    if (A->device == DEVICE_GPU) tensor_from_gpu((Tensor*)A);
    if (B->device == DEVICE_GPU) tensor_from_gpu((Tensor*)B);
    if (C->device == DEVICE_GPU) tensor_from_gpu(C);
#endif

    // General path: apply beta scaling to C first, then accumulate alpha*A@B.
    if (beta == 0.0f) {
        tensor_fill(C, 0.0f);
    } else if (beta != 1.0f) {
        for (uint32_t i = 0; i < C->size; i++) C->data[i] *= beta;
    }

    // Call optimized multi-threaded NEON Cortex-A72 GEMM with zero-allocation transposed packing
    if (alpha == 1.0f) {
        parallel_gemm_optimized_trans(A->data, B->data, C->data, M, N, K, trans_a, trans_b);
    } else {
        if (beta == 0.0f) {
            parallel_gemm_optimized_trans(A->data, B->data, C->data, M, N, K, trans_a, trans_b);
            for (uint32_t i = 0; i < C->size; i++) C->data[i] *= alpha;
        } else {
            float* tmp = (float*)rpitorch_aligned_alloc(64, (size_t)M * N * sizeof(float));
            if (tmp) {
                memset(tmp, 0, (size_t)M * N * sizeof(float));
                parallel_gemm_optimized_trans(A->data, B->data, tmp, M, N, K, trans_a, trans_b);
                #pragma omp parallel for if(C->size >= RPL_OMP_THRESHOLD)
                for (uint32_t i = 0; i < C->size; i++) C->data[i] += alpha * tmp[i];
                rpitorch_aligned_free(tmp);
            }
        }
    }
}

// ============================================================
// Activations
// ============================================================

void tensor_relu_inplace(Tensor* t) {
#ifdef USE_GPU
    if (t->device == DEVICE_GPU && RPL_GPU_PREFERABLE(t->size)) {
        tensor_relu_inplace_gpu(t);
        return;
    }
    if (t->device == DEVICE_GPU) tensor_from_gpu(t);
#endif
#if RPITORCH_HAS_NEON
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    
    #pragma omp parallel for
    for (uint32_t base = 0; base < t->size; base += 512) {
        uint32_t end = (base + 512 < t->size) ? base + 512 : t->size;
        uint32_t k = base;
        
        // Process 16 elements per iteration (4 registers)
        for (; k + 16 <= end; k += 16) {
            float32x4_t v0 = vmaxq_f32(vld1q_f32(&t->data[k]), vzero);
            float32x4_t v1 = vmaxq_f32(vld1q_f32(&t->data[k+4]), vzero);
            float32x4_t v2 = vmaxq_f32(vld1q_f32(&t->data[k+8]), vzero);
            float32x4_t v3 = vmaxq_f32(vld1q_f32(&t->data[k+12]), vzero);
            vst1q_f32(&t->data[k], v0);
            vst1q_f32(&t->data[k+4], v1);
            vst1q_f32(&t->data[k+8], v2);
            vst1q_f32(&t->data[k+12], v3);
        }
        // Tail: 4 elements
        for (; k + 4 <= end; k += 4) {
            vst1q_f32(&t->data[k], vmaxq_f32(vld1q_f32(&t->data[k]), vzero));
        }
        // Scalar tail
        for (; k < end; k++) if (t->data[k] < 0) t->data[k] = 0;
    }
#else
    #pragma omp parallel for
    for (uint32_t i = 0; i < t->size; i++) if (t->data[i] < 0) t->data[i] = 0;
#endif
    
    if (rpl_is_grad_enabled() && t->requires_grad) {
        t->parent1 = NULL;
        t->backward_fn = NULL;
        t->_op = OP_NONE;
        t->_n_parents = 0;
    } else {
        t->requires_grad = false;
    }
}

void tensor_sigmoid_inplace(Tensor* t) {
#ifdef USE_GPU
    if (t->device == DEVICE_GPU && RPL_GPU_PREFERABLE(t->size)) {
        tensor_sigmoid_gpu(t, t);
        return;
    }
    if (t->device == DEVICE_GPU) tensor_from_gpu(t);
#endif
    #pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < t->size; i++) {
        t->data[i] = 1.0f / (1.0f + expf(-t->data[i]));
    }
    
    if (rpl_is_grad_enabled() && t->requires_grad) {
        t->parent1 = NULL;
        t->backward_fn = NULL;
        t->_op = OP_NONE;
        t->_n_parents = 0;
    } else {
        t->requires_grad = false;
    }
}

Tensor* tensor_relu(const Tensor* t) {
    Tensor* out = tensor_create(t->dims, t->shape, t->requires_grad);
#ifdef USE_GPU
    if ((t->device == DEVICE_GPU || out->device == DEVICE_GPU)
            && RPL_GPU_PREFERABLE(t->size)) {
        tensor_relu_gpu(out, t);
        goto finalize;
    }
    if (t->device == DEVICE_GPU) tensor_from_gpu((Tensor*)t);
#endif
    
    #if RPITORCH_HAS_NEON
        #pragma omp parallel for
        for (uint32_t base = 0; base < t->size; base += 1024) {
            uint32_t end = (base + 1024 < t->size) ? base + 1024 : t->size;
            float32x4_t vzero = vdupq_n_f32(0.0f);
            uint32_t k = base;
            for (; k + 4 <= end; k += 4) {
                 float32x4_t val = vld1q_f32(&t->data[k]);
                 vst1q_f32(&out->data[k], vmaxq_f32(val, vzero));
            }
            for (; k < end; k++) out->data[k] = (t->data[k] > 0) ? t->data[k] : 0;
        }
    #else
        #pragma omp parallel for
        for (uint32_t i = 0; i < t->size; i++) out->data[i] = (t->data[i] > 0) ? t->data[i] : 0;
    #endif
finalize:
    if (rpl_is_grad_enabled() && out->requires_grad) {
        out->parent1 = (void*)t;
        out->backward_fn = backward_relu;
        out->is_leaf = false;
        out->_op = OP_RELU;
        out->_parents[0] = (Tensor*)t;
        out->_n_parents = 1;
        tensor_retain((Tensor*)t);
    } else {
        out->requires_grad = false;
    }
    return out;
}

Tensor* tensor_sigmoid(const Tensor* t) {
    Tensor* out = tensor_create(t->dims, t->shape, t->requires_grad);
#ifdef USE_GPU
    if ((t->device == DEVICE_GPU || out->device == DEVICE_GPU)
            && RPL_GPU_PREFERABLE(t->size)) {
        tensor_sigmoid_gpu(out, t);
        goto finalize;
    }
    if (t->device == DEVICE_GPU) tensor_from_gpu((Tensor*)t);
#endif
    
    #pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < t->size; i++) {
        out->data[i] = 1.0f / (1.0f + expf(-t->data[i]));
    }
finalize:
    if (rpl_is_grad_enabled() && out->requires_grad) {
        out->parent1 = (void*)t;
        out->backward_fn = backward_sigmoid;
        out->is_leaf = false;
        out->_op = OP_SIGMOID;
        out->_parents[0] = (Tensor*)t;
        out->_n_parents = 1;
        tensor_retain((Tensor*)t);
    } else {
        out->requires_grad = false;
    }
    return out;
}

// ============================================================
// Autograd Implementation
Tensor* tensor_mse_loss(const Tensor* pred, const Tensor* target) {
    uint32_t shape[1] = {1};
    Tensor* out = tensor_create(1, shape, pred->requires_grad);
    float loss = 0;
    for (uint32_t i = 0; i < pred->size; i++) {
        float d = pred->data[i] - target->data[i];
        loss += d * d;
    }
    out->data[0] = loss / pred->size;
    if (rpl_is_grad_enabled() && out->requires_grad) {
        out->parent1 = (void*)pred;
        out->parent2 = (void*)target;
        out->backward_fn = backward_mse;
        out->is_leaf = false;
        out->_op = OP_MSE_LOSS;
        out->_parents[0] = (Tensor*)pred;
        out->_parents[1] = (Tensor*)target;
        out->_n_parents = 2;
        tensor_retain((Tensor*)pred);
        tensor_retain((Tensor*)target);
    } else {
        out->requires_grad = false;
    }
    return out;
}

// Placeholders and in-place routines
void tensor_add_inplace(Tensor* a, const Tensor* b) { tensor_add_out(a, a, b); }

void tensor_mul_inplace(Tensor* a, float scalar) {
#ifdef USE_GPU
    if (a->device == DEVICE_GPU) { tensor_scale_gpu(a, scalar); return; }
#endif
#if RPITORCH_HAS_NEON
    float32x4_t vs = vdupq_n_f32(scalar);
    #pragma omp parallel for schedule(static) if(a->size >= 4096)
    for (uint32_t base = 0; base < a->size; base += 1024) {
        uint32_t end = (base + 1024 < a->size) ? base + 1024 : a->size;
        uint32_t idx = base;
        for (; idx + 16 <= end; idx += 16) {
            __builtin_prefetch(&a->data[idx + 64], 1, 1);
            float32x4_t v0 = vld1q_f32(&a->data[idx]);
            float32x4_t v1 = vld1q_f32(&a->data[idx + 4]);
            float32x4_t v2 = vld1q_f32(&a->data[idx + 8]);
            float32x4_t v3 = vld1q_f32(&a->data[idx + 12]);
            vst1q_f32(&a->data[idx],      vmulq_f32(v0, vs));
            vst1q_f32(&a->data[idx + 4],  vmulq_f32(v1, vs));
            vst1q_f32(&a->data[idx + 8],  vmulq_f32(v2, vs));
            vst1q_f32(&a->data[idx + 12], vmulq_f32(v3, vs));
        }
        for (; idx + 4 <= end; idx += 4) {
            vst1q_f32(&a->data[idx], vmulq_f32(vld1q_f32(&a->data[idx]), vs));
        }
        for (; idx < end; idx++) {
            a->data[idx] *= scalar;
        }
    }
#else
    #pragma omp parallel for schedule(static) if(a->size >= 4096)
    for (uint32_t i = 0; i < a->size; i++) a->data[i] *= scalar;
#endif
}

void tensor_fill_buffer(float* buffer, float value, uint32_t size) {
#if RPITORCH_HAS_NEON
    float32x4_t val_vec = vdupq_n_f32(value);
    uint32_t i = 0;
    for (; i + 16 <= size; i += 16) {
        vst1q_f32(&buffer[i], val_vec);
        vst1q_f32(&buffer[i + 4], val_vec);
        vst1q_f32(&buffer[i + 8], val_vec);
        vst1q_f32(&buffer[i + 12], val_vec);
    }
    for (; i < size; i++) buffer[i] = value;
#else
    for (uint32_t i = 0; i < size; i++) buffer[i] = value;
#endif
}

#if RPITORCH_HAS_NEON
static inline float32x4_t core_fast_exp_neon(float32x4_t x) {
    const float32x4_t LOG2E = vdupq_n_f32(1.442695040f);
    const float32x4_t C1 = vdupq_n_f32(0.0136779459f);
    const float32x4_t C2 = vdupq_n_f32(0.0517869298f);
    const float32x4_t C3 = vdupq_n_f32(0.2413797378f);
    const float32x4_t C4 = vdupq_n_f32(0.6930230856f);
    const float32x4_t ONE = vdupq_n_f32(1.0f);
    
    x = vmaxq_f32(vminq_f32(x, vdupq_n_f32(87.0f)), vdupq_n_f32(-87.0f));
    float32x4_t t = vmulq_f32(x, LOG2E);
    float32x4_t k = vrndmq_f32(t);
    float32x4_t f = vsubq_f32(t, k);
    
    float32x4_t exp_f = vfmaq_f32(C2, f, C1);
    exp_f = vfmaq_f32(C3, f, exp_f);
    exp_f = vfmaq_f32(C4, f, exp_f);
    exp_f = vfmaq_f32(ONE, f, exp_f);
    
    int32x4_t k_int = vaddq_s32(vcvtq_s32_f32(k), vdupq_n_s32(127));
    float32x4_t exp_k = vreinterpretq_f32_s32(vshlq_n_s32(k_int, 23));
    return vmulq_f32(exp_k, exp_f);
}

static inline float32x4_t core_fast_sigmoid_neon(float32x4_t x) {
    float32x4_t exp_neg = core_fast_exp_neon(vnegq_f32(x));
    float32x4_t denom = vaddq_f32(vdupq_n_f32(1.0f), exp_neg);
    float32x4_t recip = vrecpeq_f32(denom);
    recip = vmulq_f32(recip, vrecpsq_f32(denom, recip));
    recip = vmulq_f32(recip, vrecpsq_f32(denom, recip));
    return recip;
}

static inline float32x4_t core_fast_tanh_neon(float32x4_t x) {
    const float32x4_t TWO = vdupq_n_f32(2.0f);
    return vsubq_f32(vmulq_f32(TWO, core_fast_sigmoid_neon(vmulq_f32(TWO, x))), vdupq_n_f32(1.0f));
}
#endif

void tensor_tanh_inplace(Tensor* t) {
#ifdef USE_GPU
    if (t->device == DEVICE_GPU) {
        tensor_tanh_gpu(t, t);
        return;
    }
#endif
#if RPITORCH_HAS_NEON
    #pragma omp parallel for schedule(static) if(t->size >= 2048)
    for (uint32_t base = 0; base < t->size; base += 1024) {
        uint32_t end = (base + 1024 < t->size) ? base + 1024 : t->size;
        uint32_t i = base;
        for (; i + 8 <= end; i += 8) {
            __builtin_prefetch(&t->data[i + 32], 1, 1);
            float32x4_t v0 = vld1q_f32(&t->data[i]);
            float32x4_t v1 = vld1q_f32(&t->data[i + 4]);
            vst1q_f32(&t->data[i], core_fast_tanh_neon(v0));
            vst1q_f32(&t->data[i + 4], core_fast_tanh_neon(v1));
        }
        for (; i + 4 <= end; i += 4) {
            vst1q_f32(&t->data[i], core_fast_tanh_neon(vld1q_f32(&t->data[i])));
        }
        for (; i < end; i++) {
            t->data[i] = tanhf(t->data[i]);
        }
    }
#else
    #pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < t->size; i++) {
        t->data[i] = tanhf(t->data[i]);
    }
#endif
}

void tensor_gelu_inplace(Tensor* t) {
    tensor_gelu(t, t);
}

void tensor_softmax_inplace(Tensor* t) {
#ifdef USE_GPU
    if (t->device == DEVICE_GPU) {
        tensor_softmax_gpu(t, t, t->dims - 1);
        return;
    }
#endif
    uint32_t last_dim = t->shape[t->dims - 1];
    uint32_t num_rows = t->size / last_dim;
    
    #pragma omp parallel for schedule(static)
    for (uint32_t r = 0; r < num_rows; r++) {
        float* row = &t->data[r * last_dim];
#if RPITORCH_HAS_NEON
        float32x4_t vmax = vdupq_n_f32(-FLT_MAX);
        uint32_t i = 0;
        for (; i + 4 <= last_dim; i += 4) {
            vmax = vmaxq_f32(vmax, vld1q_f32(&row[i]));
        }
        float max_val = -FLT_MAX;
        for (int k = 0; k < 4; k++) {
            float v = vgetq_lane_f32(vmax, k);
            if (v > max_val) max_val = v;
        }
        for (; i < last_dim; i++) {
            if (row[i] > max_val) max_val = row[i];
        }
        vmax = vdupq_n_f32(max_val);

        float32x4_t vsum = vdupq_n_f32(0.0f);
        i = 0;
        for (; i + 4 <= last_dim; i += 4) {
            float32x4_t e = core_fast_exp_neon(vsubq_f32(vld1q_f32(&row[i]), vmax));
            vst1q_f32(&row[i], e);
            vsum = vaddq_f32(vsum, e);
        }
        float sum_exp = vgetq_lane_f32(vsum, 0) + vgetq_lane_f32(vsum, 1) +
                        vgetq_lane_f32(vsum, 2) + vgetq_lane_f32(vsum, 3);
        for (; i < last_dim; i++) {
            row[i] = expf(row[i] - max_val);
            sum_exp += row[i];
        }

        float inv_sum = 1.0f / (sum_exp > 0.0f ? sum_exp : 1e-12f);
        float32x4_t vinv = vdupq_n_f32(inv_sum);
        i = 0;
        for (; i + 4 <= last_dim; i += 4) {
            vst1q_f32(&row[i], vmulq_f32(vld1q_f32(&row[i]), vinv));
        }
        for (; i < last_dim; i++) {
            row[i] *= inv_sum;
        }
#else
        float max_val = -FLT_MAX;
        for (uint32_t i = 0; i < last_dim; i++) {
            if (row[i] > max_val) max_val = row[i];
        }
        float sum_exp = 0.0f;
        for (uint32_t i = 0; i < last_dim; i++) {
            row[i] = expf(row[i] - max_val);
            sum_exp += row[i];
        }
        float inv_sum = 1.0f / (sum_exp > 0.0f ? sum_exp : 1e-12f);
        for (uint32_t i = 0; i < last_dim; i++) {
            row[i] *= inv_sum;
        }
#endif
    }
}

QuantizedTensor* tensor_quantize_int8(const Tensor* input, float scale, int32_t zero_point) {
#ifdef USE_GPU
    if (input->device == DEVICE_GPU) {
        tensor_from_gpu((Tensor*)input);
    }
#endif
    QuantizedTensor* qt = (QuantizedTensor*)malloc(sizeof(QuantizedTensor));
    if (!qt) return NULL;
    qt->size = input->size;
    qt->dims = input->dims;
    memcpy(qt->shape, input->shape, input->dims * sizeof(uint32_t));
    qt->scale = scale;
    qt->zero_point = zero_point;
    
    qt->data = (int8_t*)rpitorch_aligned_alloc(64, qt->size);
    if (!qt->data) {
        free(qt);
        return NULL;
    }
    
    float inv_scale = 1.0f / (scale != 0.0f ? scale : 1e-7f);
    
#if RPITORCH_HAS_NEON
    float32x4_t vinv = vdupq_n_f32(inv_scale);
    int32x4_t vzp = vdupq_n_s32(zero_point);
    
    #pragma omp parallel for schedule(static) if(input->size >= 4096)
    for (uint32_t base = 0; base < input->size; base += 1024) {
        uint32_t end = (base + 1024 < input->size) ? base + 1024 : input->size;
        uint32_t i = base;
        for (; i + 16 <= end; i += 16) {
            __builtin_prefetch(&input->data[i + 64], 0, 1);
            float32x4_t f0 = vmulq_f32(vld1q_f32(&input->data[i]), vinv);
            float32x4_t f1 = vmulq_f32(vld1q_f32(&input->data[i + 4]), vinv);
            float32x4_t f2 = vmulq_f32(vld1q_f32(&input->data[i + 8]), vinv);
            float32x4_t f3 = vmulq_f32(vld1q_f32(&input->data[i + 12]), vinv);
            
            int32x4_t q0 = vaddq_s32(vcvtnq_s32_f32(f0), vzp);
            int32x4_t q1 = vaddq_s32(vcvtnq_s32_f32(f1), vzp);
            int32x4_t q2 = vaddq_s32(vcvtnq_s32_f32(f2), vzp);
            int32x4_t q3 = vaddq_s32(vcvtnq_s32_f32(f3), vzp);
            
            int16x8_t s01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
            int16x8_t s23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
            
            int8x16_t out8 = vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23));
            vst1q_s8(&qt->data[i], out8);
        }
        for (; i < end; i++) {
            int32_t quantized = (int32_t)roundf(input->data[i] * inv_scale) + zero_point;
            if (quantized < -128) quantized = -128;
            if (quantized > 127) quantized = 127;
            qt->data[i] = (int8_t)quantized;
        }
    }
#else
    #pragma omp parallel for schedule(static) if(input->size >= 4096)
    for (uint32_t i = 0; i < input->size; i++) {
        int32_t quantized = (int32_t)roundf(input->data[i] * inv_scale) + zero_point;
        if (quantized < -128) quantized = -128;
        if (quantized > 127) quantized = 127;
        qt->data[i] = (int8_t)quantized;
    }
#endif
    
    return qt;
}

void tensor_batchnorm2d(Tensor* out, const Tensor* in, float* weight, float* bias,
                        float* running_mean, float* running_var,
                        float eps, bool training, float momentum) {
    if (!out || !in) return;
    
    // NCHW layout: shape[0] = N, shape[1] = C, shape[2] = H, shape[3] = W
    uint32_t N = in->shape[0];
    uint32_t C = (in->dims > 1) ? in->shape[1] : 1;
    uint32_t HW = 1;
    for (uint32_t d = 2; d < in->dims; d++) HW *= in->shape[d];
    uint32_t channel_stride = HW;
    uint32_t batch_stride = C * HW;
    uint32_t total_per_c = N * HW;
    
    #pragma omp parallel for schedule(static)
    for (uint32_t c = 0; c < C; c++) {
        float mean_c = 0.0f;
        float var_c = 1.0f;
        
        if (training) {
            float sum = 0.0f;
#if RPITORCH_HAS_NEON
            float32x4_t vsum = vdupq_n_f32(0.0f);
            for (uint32_t n = 0; n < N; n++) {
                const float* ptr = &in->data[n * batch_stride + c * channel_stride];
                uint32_t i = 0;
                for (; i + 4 <= HW; i += 4) {
                    vsum = vaddq_f32(vsum, vld1q_f32(&ptr[i]));
                }
                for (; i < HW; i++) sum += ptr[i];
            }
            sum += vaddvq_f32(vsum);
#else
            for (uint32_t n = 0; n < N; n++) {
                const float* ptr = &in->data[n * batch_stride + c * channel_stride];
                for (uint32_t i = 0; i < HW; i++) sum += ptr[i];
            }
#endif
            mean_c = sum / total_per_c;
            
            float sq_diff = 0.0f;
#if RPITORCH_HAS_NEON
            float32x4_t vmean = vdupq_n_f32(mean_c);
            float32x4_t vsq = vdupq_n_f32(0.0f);
            for (uint32_t n = 0; n < N; n++) {
                const float* ptr = &in->data[n * batch_stride + c * channel_stride];
                uint32_t i = 0;
                for (; i + 4 <= HW; i += 4) {
                    float32x4_t diff = vsubq_f32(vld1q_f32(&ptr[i]), vmean);
                    vsq = vfmaq_f32(vsq, diff, diff);
                }
                for (; i < HW; i++) {
                    float diff = ptr[i] - mean_c;
                    sq_diff += diff * diff;
                }
            }
            sq_diff += vaddvq_f32(vsq);
#else
            for (uint32_t n = 0; n < N; n++) {
                const float* ptr = &in->data[n * batch_stride + c * channel_stride];
                for (uint32_t i = 0; i < HW; i++) {
                    float diff = ptr[i] - mean_c;
                    sq_diff += diff * diff;
                }
            }
#endif
            var_c = sq_diff / total_per_c;
            
            if (running_mean) running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean_c;
            if (running_var) running_var[c] = (1.0f - momentum) * running_var[c] + momentum * var_c;
        } else {
            mean_c = running_mean ? running_mean[c] : 0.0f;
            var_c = running_var ? running_var[c] : 1.0f;
        }
        
        float inv_std = 1.0f / sqrtf(var_c + eps);
        float gamma = weight ? weight[c] : 1.0f;
        float beta = bias ? bias[c] : 0.0f;
        float alpha = gamma * inv_std;
        float bias_eff = beta - mean_c * alpha;
        
#if RPITORCH_HAS_NEON
        float32x4_t valpha = vdupq_n_f32(alpha);
        float32x4_t vbias = vdupq_n_f32(bias_eff);
        for (uint32_t n = 0; n < N; n++) {
            const float* in_ptr = &in->data[n * batch_stride + c * channel_stride];
            float* out_ptr = &out->data[n * batch_stride + c * channel_stride];
            uint32_t i = 0;
            for (; i + 16 <= HW; i += 16) {
                __builtin_prefetch(&in_ptr[i + 32], 0, 1);
                float32x4_t x0 = vld1q_f32(&in_ptr[i]);
                float32x4_t x1 = vld1q_f32(&in_ptr[i + 4]);
                float32x4_t x2 = vld1q_f32(&in_ptr[i + 8]);
                float32x4_t x3 = vld1q_f32(&in_ptr[i + 12]);
                vst1q_f32(&out_ptr[i],      vfmaq_f32(vbias, x0, valpha));
                vst1q_f32(&out_ptr[i + 4],  vfmaq_f32(vbias, x1, valpha));
                vst1q_f32(&out_ptr[i + 8],  vfmaq_f32(vbias, x2, valpha));
                vst1q_f32(&out_ptr[i + 12], vfmaq_f32(vbias, x3, valpha));
            }
            for (; i + 4 <= HW; i += 4) {
                vst1q_f32(&out_ptr[i], vfmaq_f32(vbias, vld1q_f32(&in_ptr[i]), valpha));
            }
            for (; i < HW; i++) {
                out_ptr[i] = in_ptr[i] * alpha + bias_eff;
            }
        }
#else
        for (uint32_t n = 0; n < N; n++) {
            const float* in_ptr = &in->data[n * batch_stride + c * channel_stride];
            float* out_ptr = &out->data[n * batch_stride + c * channel_stride];
            for (uint32_t i = 0; i < HW; i++) {
                out_ptr[i] = in_ptr[i] * alpha + bias_eff;
            }
        }
#endif
    }
}

void tensor_dropout(Tensor* out, const Tensor* in, float p, bool training) {
    if (!out || !in) return;
    if (!training || p <= 0.0f) {
        if (out != in) {
            memcpy(out->data, in->data, in->size * sizeof(float));
        }
        return;
    }
    if (p >= 1.0f) {
        memset(out->data, 0, out->size * sizeof(float));
        return;
    }
    
    float scale = 1.0f / (1.0f - p);
    
#if RPITORCH_HAS_NEON
    float32x4_t vscale = vdupq_n_f32(scale);
    float32x4_t vp = vdupq_n_f32(p);
    float32x4_t vzero = vdupq_n_f32(0.0f);
    
    #pragma omp parallel
    {
        uint32_t tid = omp_get_thread_num();
        uint32_t state = 123456789 + tid * 1013904223;
        
        #pragma omp for schedule(static)
        for (uint32_t i = 0; i < in->size; i += 4) {
            uint32_t rem = in->size - i;
            if (rem >= 4) {
                float r[4];
                for (int k = 0; k < 4; k++) {
                    state = state * 1664525u + 1013904223u;
                    r[k] = (float)(state >> 8) * (1.0f / 16777216.0f);
                }
                float32x4_t vr = vld1q_f32(r);
                uint32x4_t keep_mask = vcgtq_f32(vr, vp);
                float32x4_t val = vmulq_f32(vld1q_f32(&in->data[i]), vscale);
                float32x4_t res = vbslq_f32(keep_mask, val, vzero);
                vst1q_f32(&out->data[i], res);
            } else {
                for (uint32_t k = 0; k < rem; k++) {
                    state = state * 1664525u + 1013904223u;
                    float r = (float)(state >> 8) * (1.0f / 16777216.0f);
                    out->data[i + k] = (r >= p) ? (in->data[i + k] * scale) : 0.0f;
                }
            }
        }
    }
#else
    #pragma omp parallel
    {
        uint32_t tid = omp_get_thread_num();
        uint32_t state = 123456789 + tid * 1013904223;
        #pragma omp for schedule(static)
        for (uint32_t i = 0; i < in->size; i++) {
            state = state * 1664525u + 1013904223u;
            float r = (float)(state >> 8) * (1.0f / 16777216.0f);
            out->data[i] = (r >= p) ? (in->data[i] * scale) : 0.0f;
        }
    }
#endif
}

void gemm_init_buffers();
void gemm_free_buffers();

void conv2d_winograd_2x2_3x3(const Tensor* input, const Tensor* weight, Tensor* output) {
    if (!input || !weight || !output) return;
    int in_channels = (input->dims > 1) ? input->shape[1] : 1;
    int out_channels = weight->shape[0];
    int height = (input->dims > 2) ? input->shape[2] : 1;
    int width = (input->dims > 3) ? input->shape[3] : 1;
    conv2d_winograd_3x3(input->data, weight->data, output->data,
                        in_channels, out_channels, height, width, 1, 1);
}

void tensor_free_grad(Tensor* t) {
    if (t && t->grad) {
        pool_free(t->grad, t->_alloc_size);
        t->grad = NULL;
    }
}

void rpl_empty_cache(void) {
    if (!pool_initialized) return;
    for (int i = 0; i < POOL_NUM_BINS; i++) {
        while (pool_bins[i].count > 0) {
            free(pool_bins[i].slabs[--pool_bins[i].count]);
        }
    }
}

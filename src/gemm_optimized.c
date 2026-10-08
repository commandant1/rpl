/*
 * Highly Optimized GEMM for ARM Cortex-A72
 * 8x8 micro-kernel + true FMA + optimized prefetching
 * 
 * Optimizations:
 * - 8x8 micro-kernel for better register utilization (uses 24 of 32 NEON regs)
 * - True FMA (vfmaq_f32) instead of vmla for latency hiding
 * - L1/L2 prefetch with proper distances for Cortex-A72
 * - NEON-vectorized packing routines
 * - Loop unrolling by 8 in micro-kernel
 */

#include "rpl.h"
#include <string.h>
#include <omp.h>
#include <stdlib.h>

// Optimal blocking parameters for Cortex-A72 (32KB L1D, 1MB L2)
#define MC 128   // M-dimension blocking (fits in L2)
#define KC 256   // K-dimension blocking (A panel fits in L1)
#define NC 2048  // N-dimension blocking (B panel fits in L2)
#define MR 8     // Register blocking M (8x8 micro-kernel)
#define NR 8     // Register blocking N

// Prefetch distances (in floats)
#define PREFETCH_L1_DIST 64   // 256 bytes ahead for L1
#define PREFETCH_L2_DIST 256  // 1KB ahead for L2

// Thread-local packing buffers (allocated per-thread, freed on thread exit)
static __thread float* Ac_local     = NULL;
static __thread size_t Ac_local_size = 0;

// Pack A into MR x K panels (column-major within panel)
// NEON-optimized for 8-element wide packing
static inline void pack_A_8(const float* A, float* Ap, int M, int K, int lda) {
#if RPITORCH_HAS_NEON
    for (int i = 0; i < M; i += MR) {
        int rows = (i + MR <= M) ? MR : (M - i);
        float* dst = &Ap[(i/MR) * K * MR];
        
        for (int k = 0; k < K; k++) {
            // Prefetch next column
            if (k + 8 < K) {
                __builtin_prefetch(&A[(i)*lda + k + 8], 0, 1);
            }
            
            if (rows == MR) {
                // Full 8-row panel: gather from 8 rows
                float32x4_t lo = {A[(i+0)*lda+k], A[(i+1)*lda+k], A[(i+2)*lda+k], A[(i+3)*lda+k]};
                float32x4_t hi = {A[(i+4)*lda+k], A[(i+5)*lda+k], A[(i+6)*lda+k], A[(i+7)*lda+k]};
                vst1q_f32(&dst[k*MR + 0], lo);
                vst1q_f32(&dst[k*MR + 4], hi);
            } else {
                // Partial panel with zero padding
                for (int ii = 0; ii < MR; ii++) {
                    dst[k*MR + ii] = (ii < rows) ? A[(i+ii)*lda + k] : 0.0f;
                }
            }
        }
    }
#else
    // Scalar fallback
    for (int i = 0; i < M; i += MR) {
        for (int k = 0; k < K; k++) {
            for (int ii = 0; ii < MR; ii++) {
                int row = i + ii;
                Ap[(i/MR)*K*MR + k*MR + ii] = (row < M) ? A[row*lda + k] : 0.0f;
            }
        }
    }
#endif
}

// Pack B into K x NR panels (row-major within panel)
// NEON-optimized for 8-wide vectorized copy
static inline void pack_B_8(const float* B, float* Bp, int K, int N, int ldb) {
#if RPITORCH_HAS_NEON
    for (int j = 0; j < N; j += NR) {
        int cols = (j + NR <= N) ? NR : (N - j);
        float* dst = &Bp[j * K];
        
        for (int k = 0; k < K; k++) {
            // Prefetch next row
            if (k + 4 < K) {
                __builtin_prefetch(&B[(k+4)*ldb + j], 0, 1);
            }
            
            if (cols == NR) {
                // Full 8-column panel: direct vector copy
                float32x4_t lo = vld1q_f32(&B[k*ldb + j + 0]);
                float32x4_t hi = vld1q_f32(&B[k*ldb + j + 4]);
                vst1q_f32(&dst[k*NR + 0], lo);
                vst1q_f32(&dst[k*NR + 4], hi);
            } else {
                // Partial panel with zero padding
                for (int jj = 0; jj < NR; jj++) {
                    dst[k*NR + jj] = (jj < cols) ? B[k*ldb + j + jj] : 0.0f;
                }
            }
        }
    }
#else
    // Scalar fallback
    for (int j = 0; j < N; j += NR) {
        for (int k = 0; k < K; k++) {
            for (int jj = 0; jj < NR; jj++) {
                int col = j + jj;
                Bp[j*K + k*NR + jj] = (col < N) ? B[k*ldb + col] : 0.0f;
            }
        }
    }
#endif
}

// Pack A^T into MR x K panels (A is stored K x M with leading dimension lda)
// Element A_logical(i, k) is at A_stored[k * lda + i]
// Since i is along the M dimension, elements i..i+7 are contiguous in memory!
static inline void pack_A_8_trans(const float* A, float* Ap, int M, int K, int lda) {
#if RPITORCH_HAS_NEON
    for (int i = 0; i < M; i += MR) {
        int rows = (i + MR <= M) ? MR : (M - i);
        float* dst = &Ap[(i/MR) * K * MR];
        
        for (int k = 0; k < K; k++) {
            if (rows == MR) {
                float32x4_t lo = vld1q_f32(&A[k * lda + i + 0]);
                float32x4_t hi = vld1q_f32(&A[k * lda + i + 4]);
                vst1q_f32(&dst[k * MR + 0], lo);
                vst1q_f32(&dst[k * MR + 4], hi);
            } else {
                for (int ii = 0; ii < MR; ii++) {
                    dst[k * MR + ii] = (ii < rows) ? A[k * lda + i + ii] : 0.0f;
                }
            }
        }
    }
#else
    for (int i = 0; i < M; i += MR) {
        int rows = (i + MR <= M) ? MR : (M - i);
        float* dst = &Ap[(i/MR) * K * MR];
        for (int k = 0; k < K; k++) {
            for (int ii = 0; ii < MR; ii++) {
                dst[k * MR + ii] = (ii < rows) ? A[k * lda + i + ii] : 0.0f;
            }
        }
    }
#endif
}

// Pack B^T into K x NR panels (B is stored N x K with leading dimension ldb)
// Element B_logical(k, j) is at B_stored[j * ldb + k]
static inline void pack_B_8_trans(const float* B, float* Bp, int K, int N, int ldb) {
#if RPITORCH_HAS_NEON
    for (int j = 0; j < N; j += NR) {
        int cols = (j + NR <= N) ? NR : (N - j);
        float* dst = &Bp[j * K];
        
        for (int k = 0; k < K; k++) {
            if (cols == NR) {
                float32x4_t lo = {B[(j+0)*ldb + k], B[(j+1)*ldb + k], B[(j+2)*ldb + k], B[(j+3)*ldb + k]};
                float32x4_t hi = {B[(j+4)*ldb + k], B[(j+5)*ldb + k], B[(j+6)*ldb + k], B[(j+7)*ldb + k]};
                vst1q_f32(&dst[k * NR + 0], lo);
                vst1q_f32(&dst[k * NR + 4], hi);
            } else {
                for (int jj = 0; jj < NR; jj++) {
                    dst[k * NR + jj] = (jj < cols) ? B[(j+jj)*ldb + k] : 0.0f;
                }
            }
        }
    }
#else
    for (int j = 0; j < N; j += NR) {
        int cols = (j + NR <= N) ? NR : (N - j);
        float* dst = &Bp[j * K];
        for (int k = 0; k < K; k++) {
            for (int jj = 0; jj < NR; jj++) {
                dst[k * NR + jj] = (jj < cols) ? B[(j+jj)*ldb + k] : 0.0f;
            }
        }
    }
#endif
}

// 8x8 NEON micro-kernel with true FMA
// Uses 16 accumulators (c00-c77) + 8 A-loads + 8 B-loads = 32 registers
static inline void __attribute__((hot)) gemm_micro_kernel_8x8(
    const float* restrict Ap,
    const float* restrict Bp,
    float* restrict C,
    int ldc,
    int K
) {
#if RPITORCH_HAS_NEON
    // Accumulator registers for 8x8 output tile
    // We use 8 float32x4_t pairs (16 registers for C)
    float32x4_t c00 = vdupq_n_f32(0.0f), c01 = vdupq_n_f32(0.0f);
    float32x4_t c10 = vdupq_n_f32(0.0f), c11 = vdupq_n_f32(0.0f);
    float32x4_t c20 = vdupq_n_f32(0.0f), c21 = vdupq_n_f32(0.0f);
    float32x4_t c30 = vdupq_n_f32(0.0f), c31 = vdupq_n_f32(0.0f);
    float32x4_t c40 = vdupq_n_f32(0.0f), c41 = vdupq_n_f32(0.0f);
    float32x4_t c50 = vdupq_n_f32(0.0f), c51 = vdupq_n_f32(0.0f);
    float32x4_t c60 = vdupq_n_f32(0.0f), c61 = vdupq_n_f32(0.0f);
    float32x4_t c70 = vdupq_n_f32(0.0f), c71 = vdupq_n_f32(0.0f);
    
    // Main K-loop: unroll by 4 for latency hiding
    int k = 0;
    for (; k + 4 <= K; k += 4) {
        // Prefetch for L1 and L2
        __builtin_prefetch(&Ap[k*MR + PREFETCH_L1_DIST], 0, 3);
        __builtin_prefetch(&Bp[k*NR + PREFETCH_L1_DIST], 0, 3);
        __builtin_prefetch(&Ap[k*MR + PREFETCH_L2_DIST], 0, 2);
        __builtin_prefetch(&Bp[k*NR + PREFETCH_L2_DIST], 0, 2);
        
        // Unrolled iterations 0-3
        #define ITERATION(kk) do { \
            float32x4_t a_lo = vld1q_f32(&Ap[(k+(kk))*MR + 0]); \
            float32x4_t a_hi = vld1q_f32(&Ap[(k+(kk))*MR + 4]); \
            float32x4_t b_lo = vld1q_f32(&Bp[(k+(kk))*NR + 0]); \
            float32x4_t b_hi = vld1q_f32(&Bp[(k+(kk))*NR + 4]); \
            \
            c00 = vfmaq_laneq_f32(c00, b_lo, a_lo, 0); c01 = vfmaq_laneq_f32(c01, b_hi, a_lo, 0); \
            c10 = vfmaq_laneq_f32(c10, b_lo, a_lo, 1); c11 = vfmaq_laneq_f32(c11, b_hi, a_lo, 1); \
            c20 = vfmaq_laneq_f32(c20, b_lo, a_lo, 2); c21 = vfmaq_laneq_f32(c21, b_hi, a_lo, 2); \
            c30 = vfmaq_laneq_f32(c30, b_lo, a_lo, 3); c31 = vfmaq_laneq_f32(c31, b_hi, a_lo, 3); \
            c40 = vfmaq_laneq_f32(c40, b_lo, a_hi, 0); c41 = vfmaq_laneq_f32(c41, b_hi, a_hi, 0); \
            c50 = vfmaq_laneq_f32(c50, b_lo, a_hi, 1); c51 = vfmaq_laneq_f32(c51, b_hi, a_hi, 1); \
            c60 = vfmaq_laneq_f32(c60, b_lo, a_hi, 2); c61 = vfmaq_laneq_f32(c61, b_hi, a_hi, 2); \
            c70 = vfmaq_laneq_f32(c70, b_lo, a_hi, 3); c71 = vfmaq_laneq_f32(c71, b_hi, a_hi, 3); \
        } while(0)
        
        ITERATION(0);
        ITERATION(1);
        ITERATION(2);
        ITERATION(3);
        
        #undef ITERATION
    }
    
    // Handle remaining K iterations
    for (; k < K; k++) {
        float32x4_t a_lo = vld1q_f32(&Ap[k*MR + 0]);
        float32x4_t a_hi = vld1q_f32(&Ap[k*MR + 4]);
        float32x4_t b_lo = vld1q_f32(&Bp[k*NR + 0]);
        float32x4_t b_hi = vld1q_f32(&Bp[k*NR + 4]);
        
        c00 = vfmaq_laneq_f32(c00, b_lo, a_lo, 0); c01 = vfmaq_laneq_f32(c01, b_hi, a_lo, 0);
        c10 = vfmaq_laneq_f32(c10, b_lo, a_lo, 1); c11 = vfmaq_laneq_f32(c11, b_hi, a_lo, 1);
        c20 = vfmaq_laneq_f32(c20, b_lo, a_lo, 2); c21 = vfmaq_laneq_f32(c21, b_hi, a_lo, 2);
        c30 = vfmaq_laneq_f32(c30, b_lo, a_lo, 3); c31 = vfmaq_laneq_f32(c31, b_hi, a_lo, 3);
        c40 = vfmaq_laneq_f32(c40, b_lo, a_hi, 0); c41 = vfmaq_laneq_f32(c41, b_hi, a_hi, 0);
        c50 = vfmaq_laneq_f32(c50, b_lo, a_hi, 1); c51 = vfmaq_laneq_f32(c51, b_hi, a_hi, 1);
        c60 = vfmaq_laneq_f32(c60, b_lo, a_hi, 2); c61 = vfmaq_laneq_f32(c61, b_hi, a_hi, 2);
        c70 = vfmaq_laneq_f32(c70, b_lo, a_hi, 3); c71 = vfmaq_laneq_f32(c71, b_hi, a_hi, 3);
    }
    
    // Load existing C, accumulate, and store
    // Row 0
    c00 = vaddq_f32(c00, vld1q_f32(&C[0*ldc + 0]));
    c01 = vaddq_f32(c01, vld1q_f32(&C[0*ldc + 4]));
    vst1q_f32(&C[0*ldc + 0], c00); vst1q_f32(&C[0*ldc + 4], c01);
    // Row 1
    c10 = vaddq_f32(c10, vld1q_f32(&C[1*ldc + 0]));
    c11 = vaddq_f32(c11, vld1q_f32(&C[1*ldc + 4]));
    vst1q_f32(&C[1*ldc + 0], c10); vst1q_f32(&C[1*ldc + 4], c11);
    // Row 2
    c20 = vaddq_f32(c20, vld1q_f32(&C[2*ldc + 0]));
    c21 = vaddq_f32(c21, vld1q_f32(&C[2*ldc + 4]));
    vst1q_f32(&C[2*ldc + 0], c20); vst1q_f32(&C[2*ldc + 4], c21);
    // Row 3
    c30 = vaddq_f32(c30, vld1q_f32(&C[3*ldc + 0]));
    c31 = vaddq_f32(c31, vld1q_f32(&C[3*ldc + 4]));
    vst1q_f32(&C[3*ldc + 0], c30); vst1q_f32(&C[3*ldc + 4], c31);
    // Row 4
    c40 = vaddq_f32(c40, vld1q_f32(&C[4*ldc + 0]));
    c41 = vaddq_f32(c41, vld1q_f32(&C[4*ldc + 4]));
    vst1q_f32(&C[4*ldc + 0], c40); vst1q_f32(&C[4*ldc + 4], c41);
    // Row 5
    c50 = vaddq_f32(c50, vld1q_f32(&C[5*ldc + 0]));
    c51 = vaddq_f32(c51, vld1q_f32(&C[5*ldc + 4]));
    vst1q_f32(&C[5*ldc + 0], c50); vst1q_f32(&C[5*ldc + 4], c51);
    // Row 6
    c60 = vaddq_f32(c60, vld1q_f32(&C[6*ldc + 0]));
    c61 = vaddq_f32(c61, vld1q_f32(&C[6*ldc + 4]));
    vst1q_f32(&C[6*ldc + 0], c60); vst1q_f32(&C[6*ldc + 4], c61);
    // Row 7
    c70 = vaddq_f32(c70, vld1q_f32(&C[7*ldc + 0]));
    c71 = vaddq_f32(c71, vld1q_f32(&C[7*ldc + 4]));
    vst1q_f32(&C[7*ldc + 0], c70); vst1q_f32(&C[7*ldc + 4], c71);
    
#else
    // Scalar fallback
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < 8; i++) {
            float a_val = Ap[k*MR + i];
            for (int j = 0; j < 8; j++) {
                C[i*ldc + j] += a_val * Bp[k*NR + j];
            }
        }
    }
#endif
}

// 8x8 NEON micro-kernel edge handler
// Accumulates tile into a local 8x8 buffer, then writes only valid mr x nr elements to C
static inline void __attribute__((hot)) gemm_micro_kernel_8x8_edge(
    const float* restrict Ap,
    const float* restrict Bp,
    float* restrict C,
    int ldc,
    int K,
    int mr,
    int nr
) {
#if RPITORCH_HAS_NEON
    float32x4_t c00 = vdupq_n_f32(0.0f), c01 = vdupq_n_f32(0.0f);
    float32x4_t c10 = vdupq_n_f32(0.0f), c11 = vdupq_n_f32(0.0f);
    float32x4_t c20 = vdupq_n_f32(0.0f), c21 = vdupq_n_f32(0.0f);
    float32x4_t c30 = vdupq_n_f32(0.0f), c31 = vdupq_n_f32(0.0f);
    float32x4_t c40 = vdupq_n_f32(0.0f), c41 = vdupq_n_f32(0.0f);
    float32x4_t c50 = vdupq_n_f32(0.0f), c51 = vdupq_n_f32(0.0f);
    float32x4_t c60 = vdupq_n_f32(0.0f), c61 = vdupq_n_f32(0.0f);
    float32x4_t c70 = vdupq_n_f32(0.0f), c71 = vdupq_n_f32(0.0f);

    int k = 0;
    for (; k + 4 <= K; k += 4) {
        #define ITERATION_EDGE(kk) do { \
            float32x4_t a_lo = vld1q_f32(&Ap[(k+(kk))*MR + 0]); \
            float32x4_t a_hi = vld1q_f32(&Ap[(k+(kk))*MR + 4]); \
            float32x4_t b_lo = vld1q_f32(&Bp[(k+(kk))*NR + 0]); \
            float32x4_t b_hi = vld1q_f32(&Bp[(k+(kk))*NR + 4]); \
            \
            c00 = vfmaq_laneq_f32(c00, b_lo, a_lo, 0); c01 = vfmaq_laneq_f32(c01, b_hi, a_lo, 0); \
            c10 = vfmaq_laneq_f32(c10, b_lo, a_lo, 1); c11 = vfmaq_laneq_f32(c11, b_hi, a_lo, 1); \
            c20 = vfmaq_laneq_f32(c20, b_lo, a_lo, 2); c21 = vfmaq_laneq_f32(c21, b_hi, a_lo, 2); \
            c30 = vfmaq_laneq_f32(c30, b_lo, a_lo, 3); c31 = vfmaq_laneq_f32(c31, b_hi, a_lo, 3); \
            c40 = vfmaq_laneq_f32(c40, b_lo, a_hi, 0); c41 = vfmaq_laneq_f32(c41, b_hi, a_hi, 0); \
            c50 = vfmaq_laneq_f32(c50, b_lo, a_hi, 1); c51 = vfmaq_laneq_f32(c51, b_hi, a_hi, 1); \
            c60 = vfmaq_laneq_f32(c60, b_lo, a_hi, 2); c61 = vfmaq_laneq_f32(c61, b_hi, a_hi, 2); \
            c70 = vfmaq_laneq_f32(c70, b_lo, a_hi, 3); c71 = vfmaq_laneq_f32(c71, b_hi, a_hi, 3); \
        } while(0)

        ITERATION_EDGE(0);
        ITERATION_EDGE(1);
        ITERATION_EDGE(2);
        ITERATION_EDGE(3);

        #undef ITERATION_EDGE
    }

    for (; k < K; k++) {
        float32x4_t a_lo = vld1q_f32(&Ap[k*MR + 0]);
        float32x4_t a_hi = vld1q_f32(&Ap[k*MR + 4]);
        float32x4_t b_lo = vld1q_f32(&Bp[k*NR + 0]);
        float32x4_t b_hi = vld1q_f32(&Bp[k*NR + 4]);

        c00 = vfmaq_laneq_f32(c00, b_lo, a_lo, 0); c01 = vfmaq_laneq_f32(c01, b_hi, a_lo, 0);
        c10 = vfmaq_laneq_f32(c10, b_lo, a_lo, 1); c11 = vfmaq_laneq_f32(c11, b_hi, a_lo, 1);
        c20 = vfmaq_laneq_f32(c20, b_lo, a_lo, 2); c21 = vfmaq_laneq_f32(c21, b_hi, a_lo, 2);
        c30 = vfmaq_laneq_f32(c30, b_lo, a_lo, 3); c31 = vfmaq_laneq_f32(c31, b_hi, a_lo, 3);
        c40 = vfmaq_laneq_f32(c40, b_lo, a_hi, 0); c41 = vfmaq_laneq_f32(c41, b_hi, a_hi, 0);
        c50 = vfmaq_laneq_f32(c50, b_lo, a_hi, 1); c51 = vfmaq_laneq_f32(c51, b_hi, a_hi, 1);
        c60 = vfmaq_laneq_f32(c60, b_lo, a_hi, 2); c61 = vfmaq_laneq_f32(c61, b_hi, a_hi, 2);
        c70 = vfmaq_laneq_f32(c70, b_lo, a_hi, 3); c71 = vfmaq_laneq_f32(c71, b_hi, a_hi, 3);
    }

    // Store into local tile
    float tile[8][8];
    vst1q_f32(&tile[0][0], c00); vst1q_f32(&tile[0][4], c01);
    vst1q_f32(&tile[1][0], c10); vst1q_f32(&tile[1][4], c11);
    vst1q_f32(&tile[2][0], c20); vst1q_f32(&tile[2][4], c21);
    vst1q_f32(&tile[3][0], c30); vst1q_f32(&tile[3][4], c31);
    vst1q_f32(&tile[4][0], c40); vst1q_f32(&tile[4][4], c41);
    vst1q_f32(&tile[5][0], c50); vst1q_f32(&tile[5][4], c51);
    vst1q_f32(&tile[6][0], c60); vst1q_f32(&tile[6][4], c61);
    vst1q_f32(&tile[7][0], c70); vst1q_f32(&tile[7][4], c71);

    // Accumulate only valid mr x nr elements into C
    for (int i = 0; i < mr; i++) {
        for (int j = 0; j < nr; j++) {
            C[i * ldc + j] += tile[i][j];
        }
    }
#else
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < mr; i++) {
            float a_val = Ap[k*MR + i];
            for (int j = 0; j < nr; j++) {
                C[i*ldc + j] += a_val * Bp[k*NR + j];
            }
        }
    }
#endif
}

// Cleanup buffers — Ac_local is thread-local and freed on thread exit via the destructor
// registered in parallel_gemm_optimized; Bc is local per-call (see below).
void gemm_init_buffers() {}   // no-op: buffers are lazily allocated

void gemm_free_buffers() {
    if (Ac_local) {
        rpitorch_aligned_free(Ac_local);
        Ac_local = NULL;
        Ac_local_size = 0;
    }
}

// Optimized GEMM with 5-level blocking.
// NOT thread-safe to call recursively from inside an OMP parallel region.
// Callers that want outer parallelism must call parallel_gemm_optimized which
// manages its own omp parallel internally.
void gemm_optimized_cortex_a72_trans(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    bool trans_a, bool trans_b
) {
    if (M <= 0 || N <= 0 || K <= 0) return;

    // Allocate Bc locally per call — eliminates the global shared-pointer race.
    // Padded to multiples of NR and KC.
    const int max_nc = (N < NC) ? ((N + NR - 1) / NR * NR) : NC;
    const int max_kc = (K < KC) ? K : KC;
    const size_t bc_bytes = (size_t)max_nc * (size_t)max_kc * sizeof(float);
    const size_t ac_max_bytes = (size_t)MC * (size_t)KC * sizeof(float);

    if (omp_in_parallel()) {
        // Sequential/Single-threaded execution because we are already in an active parallel region
        float* Bc = (float*)rpitorch_aligned_alloc(64, bc_bytes > 0 ? bc_bytes : 64);

        // Ensure thread-local A buffer is large enough for any MC x KC panel
        if (Ac_local == NULL || Ac_local_size < ac_max_bytes) {
            if (Ac_local) rpitorch_aligned_free(Ac_local);
            Ac_local = (float*)rpitorch_aligned_alloc(64, ac_max_bytes);
            Ac_local_size = ac_max_bytes;
        }

        for (int jc = 0; jc < N; jc += NC) {
            int nc = (jc + NC > N) ? (N - jc) : NC;

            for (int pc = 0; pc < K; pc += KC) {
                int kc = (pc + KC > K) ? (K - pc) : KC;

                const float* b_blk = trans_b ? &B[jc * ldb + pc] : &B[pc * ldb + jc];
                if (trans_b) pack_B_8_trans(b_blk, Bc, kc, nc, ldb);
                else pack_B_8(b_blk, Bc, kc, nc, ldb);

                for (int ic = 0; ic < M; ic += MC) {
                    int mc = (ic + MC > M) ? (M - ic) : MC;

                    const float* a_blk = trans_a ? &A[pc * lda + ic] : &A[ic * lda + pc];
                    if (trans_a) pack_A_8_trans(a_blk, Ac_local, mc, kc, lda);
                    else pack_A_8(a_blk, Ac_local, mc, kc, lda);

                    for (int jr = 0; jr < nc; jr += NR) {
                        int nr_cur = (jr + NR <= nc) ? NR : (nc - jr);
                        for (int ir = 0; ir < mc; ir += MR) {
                            int mr_cur = (ir + MR <= mc) ? MR : (mc - ir);
                            if (mr_cur == MR && nr_cur == NR) {
                                gemm_micro_kernel_8x8(
                                    &Ac_local[(ir / MR) * kc * MR],
                                    &Bc[jr * kc],
                                    &C[(ic + ir) * ldc + jc + jr],
                                    ldc,
                                    kc
                                );
                            } else {
                                gemm_micro_kernel_8x8_edge(
                                    &Ac_local[(ir / MR) * kc * MR],
                                    &Bc[jr * kc],
                                    &C[(ic + ir) * ldc + jc + jr],
                                    ldc,
                                    kc,
                                    mr_cur,
                                    nr_cur
                                );
                            }
                        }
                    }
                }
            }
        }
        rpitorch_aligned_free(Bc);
    } else {
        // Multi-threaded execution with task-based double-buffered B packing
        float* Bc[2];
        Bc[0] = (float*)rpitorch_aligned_alloc(64, bc_bytes > 0 ? bc_bytes : 64);
        Bc[1] = (float*)rpitorch_aligned_alloc(64, bc_bytes > 0 ? bc_bytes : 64);

        for (int jc = 0; jc < N; jc += NC) {
            int nc = (jc + NC > N) ? (N - jc) : NC;

            // Pack first block of B into Bc[0]
            int kc_first = (0 + KC > K) ? (K - 0) : KC;
            const float* b_first = trans_b ? &B[jc * ldb + 0] : &B[0 * ldb + jc];
            if (trans_b) pack_B_8_trans(b_first, Bc[0], kc_first, nc, ldb);
            else pack_B_8(b_first, Bc[0], kc_first, nc, ldb);

            for (int pc = 0; pc < K; pc += KC) {
                int kc = (pc + KC > K) ? (K - pc) : KC;
                int buf_idx = (pc / KC) % 2;
                int next_buf_idx = 1 - buf_idx;
                int next_pc = pc + KC;
                int next_kc = (next_pc + KC > K) ? (K - next_pc) : KC;

                #pragma omp parallel
                {
                    // Ensure thread-local A buffer is large enough for any MC x KC panel
                    if (Ac_local == NULL || Ac_local_size < ac_max_bytes) {
                        if (Ac_local) rpitorch_aligned_free(Ac_local);
                        Ac_local = (float*)rpitorch_aligned_alloc(64, ac_max_bytes);
                        Ac_local_size = ac_max_bytes;
                    }

                    #pragma omp single nowait
                    {
                        if (next_pc < K) {
                            #pragma omp task
                            {
                                const float* next_b = trans_b ? &B[jc * ldb + next_pc] : &B[next_pc * ldb + jc];
                                if (trans_b) pack_B_8_trans(next_b, Bc[next_buf_idx], next_kc, nc, ldb);
                                else pack_B_8(next_b, Bc[next_buf_idx], next_kc, nc, ldb);
                            }
                        }
                    }

                    #pragma omp for schedule(dynamic, 1)
                    for (int ic = 0; ic < M; ic += MC) {
                        int mc = (ic + MC > M) ? (M - ic) : MC;

                        // Pack A panel (thread-local)
                        const float* a_blk = trans_a ? &A[pc * lda + ic] : &A[ic * lda + pc];
                        if (trans_a) pack_A_8_trans(a_blk, Ac_local, mc, kc, lda);
                        else pack_A_8(a_blk, Ac_local, mc, kc, lda);

                        // Micro-panel loop
                        for (int jr = 0; jr < nc; jr += NR) {
                            int nr_cur = (jr + NR <= nc) ? NR : (nc - jr);
                            for (int ir = 0; ir < mc; ir += MR) {
                                int mr_cur = (ir + MR <= mc) ? MR : (mc - ir);
                                if (mr_cur == MR && nr_cur == NR) {
                                    gemm_micro_kernel_8x8(
                                        &Ac_local[(ir / MR) * kc * MR],
                                        &Bc[buf_idx][jr * kc],
                                        &C[(ic + ir) * ldc + jc + jr],
                                        ldc,
                                        kc
                                    );
                                } else {
                                    gemm_micro_kernel_8x8_edge(
                                        &Ac_local[(ir / MR) * kc * MR],
                                        &Bc[buf_idx][jr * kc],
                                        &C[(ic + ir) * ldc + jc + jr],
                                        ldc,
                                        kc,
                                        mr_cur,
                                        nr_cur
                                    );
                                }
                            }
                        }
                    }

                    // Wait for next block packing task to finish before leaving parallel region
                    #pragma omp taskwait
                }
            }
        }

        rpitorch_aligned_free(Bc[0]);
        rpitorch_aligned_free(Bc[1]);
    }
}

void gemm_optimized_cortex_a72(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    int lda, int ldb, int ldc
) {
    gemm_optimized_cortex_a72_trans(A, B, C, M, N, K, lda, ldb, ldc, false, false);
}

void parallel_gemm_optimized_trans(const float* A, const float* B, float* C,
                                   uint32_t M, uint32_t N, uint32_t K,
                                   bool trans_a, bool trans_b) {
    if (M == 0 || N == 0 || K == 0) return;
    // Small matrix: NEON 4-wide for M<8 or N<8 (avoids 8x8 tile overrun)
    if (M < MR || N < NR) {
#if RPITORCH_HAS_NEON
        for (uint32_t i = 0; i < M; i++) {
            for (uint32_t j = 0; j < N; j++) {
                float32x4_t vsum = vdupq_n_f32(0.0f);
                uint32_t k = 0;
                for (; k + 4 <= K; k += 4) {
                    float a0 = trans_a ? A[(k+0)*M + i] : A[i*K + k+0];
                    float a1 = trans_a ? A[(k+1)*M + i] : A[i*K + k+1];
                    float a2 = trans_a ? A[(k+2)*M + i] : A[i*K + k+2];
                    float a3 = trans_a ? A[(k+3)*M + i] : A[i*K + k+3];
                    float b0 = trans_b ? B[j*K + k+0]   : B[(k+0)*N + j];
                    float b1 = trans_b ? B[j*K + k+1]   : B[(k+1)*N + j];
                    float b2 = trans_b ? B[j*K + k+2]   : B[(k+2)*N + j];
                    float b3 = trans_b ? B[j*K + k+3]   : B[(k+3)*N + j];
                    float32x4_t va = {a0, a1, a2, a3};
                    float32x4_t vb = {b0, b1, b2, b3};
                    vsum = vfmaq_f32(vsum, va, vb);
                }
                float sum = vaddvq_f32(vsum);
                for (; k < K; k++) {
                    float a_val = trans_a ? A[k*M + i] : A[i*K + k];
                    float b_val = trans_b ? B[j*K + k] : B[k*N + j];
                    sum += a_val * b_val;
                }
                C[i * N + j] += sum;
            }
        }
#else
        for (uint32_t i = 0; i < M; i++) {
            for (uint32_t j = 0; j < N; j++) {
                float sum = 0;
                for (uint32_t k = 0; k < K; k++) {
                    float a_val = trans_a ? A[k*M + i] : A[i*K + k];
                    float b_val = trans_b ? B[j*K + k] : B[k*N + j];
                    sum += a_val * b_val;
                }
                C[i * N + j] += sum;
            }
        }
#endif
        return;
    }
    int lda = trans_a ? M : K;
    int ldb = trans_b ? K : N;
    int ldc = N;
    gemm_optimized_cortex_a72_trans(A, B, C, M, N, K, lda, ldb, ldc, trans_a, trans_b);
}

// Wrapper for tensor interface
void parallel_gemm_optimized(const float* A, const float* B, float* C,
                             uint32_t M, uint32_t N, uint32_t K) {
    parallel_gemm_optimized_trans(A, B, C, M, N, K, false, false);
}

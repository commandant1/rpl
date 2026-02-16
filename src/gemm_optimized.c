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

// Thread-local packing buffers (allocated per-thread)
static __thread float* Ac_local = NULL;
static __thread size_t Ac_local_size = 0;

// Shared B packing buffer
static float* Bc __attribute__((aligned(64))) = NULL;
static size_t Bc_size = 0;

// Initialize/resize packing buffers
static inline void ensure_buffers(size_t ac_size, size_t bc_size) {
    if (Ac_local == NULL || Ac_local_size < ac_size) {
        if (Ac_local) rpitorch_aligned_free(Ac_local);
        Ac_local = (float*)rpitorch_aligned_alloc(64, ac_size);
        Ac_local_size = ac_size;
    }
    
    #pragma omp critical
    {
        if (Bc == NULL || Bc_size < bc_size) {
            if (Bc) rpitorch_aligned_free(Bc);
            Bc = (float*)rpitorch_aligned_alloc(64, bc_size);
            Bc_size = bc_size;
        }
    }
}

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

// 8x8 NEON micro-kernel with true FMA
// Computes C[8x8] += A[8xK] @ B[Kx8]
// Uses 16 accumulators (c00-c77) + 8 A-loads + 8 B-loads = 32 registers
static inline void gemm_micro_kernel_8x8(
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

// Cleanup buffers
void gemm_init_buffers() {
    // Now handled lazily in ensure_buffers
}

void gemm_free_buffers() {
    if (Bc) { rpitorch_aligned_free(Bc); Bc = NULL; Bc_size = 0; }
    // Thread-local Ac_local freed on thread exit
}

// Optimized GEMM with 5-level blocking
void gemm_optimized_cortex_a72(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    int lda, int ldb, int ldc
) {
    // Round up to tile boundaries
    int M_tiles = (M + MR - 1) / MR;
    int N_tiles = (N + NR - 1) / NR;
    
    // Level 5: Outer N-loop (L2 cache blocking for B)
    for (int jc = 0; jc < N; jc += NC) {
        int nc = (jc + NC > N) ? (N - jc) : NC;
        
        // Level 4: K-loop (L1 cache blocking)
        for (int pc = 0; pc < K; pc += KC) {
            int kc = (pc + KC > K) ? (K - pc) : KC;
            
            // Ensure B buffer is large enough
            size_t bc_needed = (size_t)nc * kc * sizeof(float);
            #pragma omp single
            {
                if (Bc == NULL || Bc_size < bc_needed) {
                    if (Bc) rpitorch_aligned_free(Bc);
                    Bc = (float*)rpitorch_aligned_alloc(64, bc_needed);
                    Bc_size = bc_needed;
                }
                // Pack B panel (single-threaded to avoid races)
                pack_B_8(&B[pc*ldb + jc], Bc, kc, nc, ldb);
            }
            
            // Level 3: M-loop (L2 cache blocking, parallelized)
            #pragma omp parallel
            {
                // Ensure thread-local A buffer
                size_t ac_needed = (size_t)MC * kc * sizeof(float);
                if (Ac_local == NULL || Ac_local_size < ac_needed) {
                    if (Ac_local) rpitorch_aligned_free(Ac_local);
                    Ac_local = (float*)rpitorch_aligned_alloc(64, ac_needed);
                    Ac_local_size = ac_needed;
                }
                
                #pragma omp for schedule(dynamic, 1)
                for (int ic = 0; ic < M; ic += MC) {
                    int mc = (ic + MC > M) ? (M - ic) : MC;
                    
                    // Pack A panel (thread-local)
                    pack_A_8(&A[ic*lda + pc], Ac_local, mc, kc, lda);
                    
                    // Level 2: Micro-panel N-loop
                    for (int jr = 0; jr < nc; jr += NR) {
                        // Level 1: Micro-panel M-loop
                        for (int ir = 0; ir < mc; ir += MR) {
                            // 8x8 micro-kernel
                            gemm_micro_kernel_8x8(
                                &Ac_local[(ir/MR)*kc*MR],
                                &Bc[jr*kc],
                                &C[(ic+ir)*ldc + jc + jr],
                                ldc,
                                kc
                            );
                        }
                    }
                }
            }
        }
    }
}

// Wrapper for tensor interface
void parallel_gemm_optimized(const float* A, const float* B, float* C,
                             uint32_t M, uint32_t N, uint32_t K) {
    // Small matrix: use simple scalar GEMM to avoid 8x8 micro-kernel overrun.
    // The tiled kernel writes a full MR×NR (8×8) tile which corrupts memory
    // when the output matrix is smaller than 8×8.
    if (M < MR || N < NR) {
        for (uint32_t i = 0; i < M; i++)
            for (uint32_t j = 0; j < N; j++) {
                float sum = 0;
                for (uint32_t k = 0; k < K; k++)
                    sum += A[i * K + k] * B[k * N + j];
                C[i * N + j] += sum;
            }
        return;
    }
    gemm_optimized_cortex_a72(A, B, C, M, N, K, K, N, N);
}

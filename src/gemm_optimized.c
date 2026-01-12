/*
 * Highly Optimized GEMM for ARM Cortex-A72
 * 5-level cache blocking + NEON FMA + prefetching
 */

#include "rpl.h"
#include <string.h>
#include <omp.h>
#include <stdlib.h>

// Optimal blocking parameters for Cortex-A72
#define MC 256   // M-dimension blocking (L2 cache)
#define KC 128   // K-dimension blocking (L1 cache)
#define NC 4096  // N-dimension blocking (L2 cache)
#define MR 4     // Register blocking M
#define NR 4     // Register blocking N

// Aligned buffer for packing
static float* Ac __attribute__((aligned(128))) = NULL;
static float* Bc __attribute__((aligned(128))) = NULL;

// Initialize packing buffers
void gemm_init_buffers() {
    Ac = (float*)rpitorch_aligned_alloc(128, MC * KC * sizeof(float));
    Bc = (float*)rpitorch_aligned_alloc(128, KC * NC * sizeof(float));
}

void gemm_free_buffers() {
    if (Ac) rpitorch_aligned_free(Ac);
    if (Bc) rpitorch_aligned_free(Bc);
    Ac = NULL;
    Bc = NULL;
}

// Pack A into column-major panels for streaming
static inline void pack_A(const float* A, float* Ap, int M, int K, int lda) {
    for (int i = 0; i < M; i += MR) {
        for (int k = 0; k < K; k++) {
            for (int ii = 0; ii < MR; ii++) {
                if ((i + ii) < M) {
                    Ap[(i/MR)*K*MR + k*MR + ii] = A[(i+ii)*lda + k];
                } else {
                    Ap[(i/MR)*K*MR + k*MR + ii] = 0.0f;
                }
            }
        }
    }
}

// Pack B into row-major panels for streaming
static inline void pack_B(const float* B, float* Bp, int K, int N, int ldb) {
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < N; j += NR) {
            for (int jj = 0; jj < NR; jj++) {
                if ((j + jj) < N) {
                    Bp[k*N + j + jj] = B[k*ldb + j + jj];
                } else {
                    Bp[k*N + j + jj] = 0.0f;
                }
            }
        }
    }
}

// 4x4 NEON micro-kernel with FMA
static inline void gemm_micro_kernel_4x4(
    const float* restrict Ap,
    const float* restrict Bp,
    float* restrict C,
    int ldc,
    int K
) {
#if RPITORCH_HAS_NEON
    // Load C into NEON registers
    float32x4_t c0 = vld1q_f32(&C[0*ldc]);
    float32x4_t c1 = vld1q_f32(&C[1*ldc]);
    float32x4_t c2 = vld1q_f32(&C[2*ldc]);
    float32x4_t c3 = vld1q_f32(&C[3*ldc]);
    
    // Main loop: unroll by 4 for dual-issue
    for (int k = 0; k < K; k += 4) {
        // Prefetch next iteration (L1 cache)
        #ifdef __GNUC__
        __builtin_prefetch(&Ap[k+64], 0, 3);
        __builtin_prefetch(&Bp[k*4+64], 0, 3);
        #endif
        
        // Iteration 0
        float32x4_t a0 = vld1q_dup_f32(&Ap[k+0]);
        float32x4_t b0 = vld1q_f32(&Bp[(k+0)*4]);
        c0 = vmlaq_f32(c0, a0, b0);
        
        // Iteration 1
        float32x4_t a1 = vld1q_dup_f32(&Ap[k+1]);
        float32x4_t b1 = vld1q_f32(&Bp[(k+1)*4]);
        c1 = vmlaq_f32(c1, a1, b1);
        
        // Iteration 2
        float32x4_t a2 = vld1q_dup_f32(&Ap[k+2]);
        float32x4_t b2 = vld1q_f32(&Bp[(k+2)*4]);
        c2 = vmlaq_f32(c2, a2, b2);
        
        // Iteration 3
        float32x4_t a3 = vld1q_dup_f32(&Ap[k+3]);
        float32x4_t b3 = vld1q_f32(&Bp[(k+3)*4]);
        c3 = vmlaq_f32(c3, a3, b3);
    }
    
    // Store results
    vst1q_f32(&C[0*ldc], c0);
    vst1q_f32(&C[1*ldc], c1);
    vst1q_f32(&C[2*ldc], c2);
    vst1q_f32(&C[3*ldc], c3);
#else
    // Scalar fallback
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                C[i*ldc + j] += Ap[k*4 + i] * Bp[k*4 + j];
            }
        }
    }
#endif
}

// Optimized GEMM with 5-level blocking
void gemm_optimized_cortex_a72(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    int lda, int ldb, int ldc
) {
    // Initialize buffers if needed
    static int initialized = 0;
    if (!initialized) {
        gemm_init_buffers();
        initialized = 1;
    }
    
    // Level 5: Outer N-loop (L2 cache blocking)
    for (int jc = 0; jc < N; jc += NC) {
        int nc = (jc + NC > N) ? (N - jc) : NC;
        
        // Level 4: K-loop (L1 cache blocking)
        for (int pc = 0; pc < K; pc += KC) {
            int kc = (pc + KC > K) ? (K - pc) : KC;
            
            // Pack B panel
            pack_B(&B[pc*ldb + jc], Bc, kc, nc, ldb);
            
            // Level 3: M-loop (L2 cache blocking)
            #pragma omp parallel for schedule(dynamic, 1) num_threads(4)
            for (int ic = 0; ic < M; ic += MC) {
                int mc = (ic + MC > M) ? (M - ic) : MC;
                
                // Pack A panel (thread-local)
                float* Ac_local = (float*)rpitorch_aligned_alloc(64, mc * kc * sizeof(float));
                pack_A(&A[ic*lda + pc], Ac_local, mc, kc, lda);
                
                // Level 2: Micro-panel N-loop
                for (int jr = 0; jr < nc; jr += NR) {
                    // Level 1: Micro-panel M-loop
                    for (int ir = 0; ir < mc; ir += MR) {
                        // Micro-kernel: 4x4 block
                        gemm_micro_kernel_4x4(
                            &Ac_local[(ir/MR)*kc*MR],
                            &Bc[jr],
                            &C[(ic+ir)*ldc + jc + jr],
                            ldc,
                            kc
                        );
                    }
                }
                
                rpitorch_aligned_free(Ac_local);
            }
        }
    }
}

// Wrapper for tensor interface
void parallel_gemm_optimized(const float* A, const float* B, float* C,
                             uint32_t M, uint32_t N, uint32_t K) {
    gemm_optimized_cortex_a72(A, B, C, M, N, K, K, N, N);
}

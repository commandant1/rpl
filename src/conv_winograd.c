/*
 * Winograd Convolution F(2x2, 3x3) for ARM Cortex-A72
 * 2.25x fewer multiplications than direct convolution
 * 
 * Optimizations:
 * - Corrected BᵀdB and AᵀmA transforms
 * - Proper 4x4 NEON transpose using vtrn/vzip
 * - True FMA in accumulation
 * - Optimized tile extraction with prefetching
 */

#include "rpl.h"
#include <omp.h>
#include <stdlib.h>
#include <string.h>

// Winograd transform matrices for F(2x2, 3x3)
// Bᵀ transform (input):  d' = Bᵀ d B
// Aᵀ transform (output): m' = Aᵀ m A

// 4x4 NEON transpose using vtrn and vuzp
#if RPITORCH_HAS_NEON
static inline void transpose_4x4_neon(float32x4_t* r0, float32x4_t* r1, 
                                       float32x4_t* r2, float32x4_t* r3) {
    // Transpose using vtrn (interleave pairs)
    float32x4x2_t t01 = vtrnq_f32(*r0, *r1);
    float32x4x2_t t23 = vtrnq_f32(*r2, *r3);
    
    // Combine to get final transpose
    *r0 = vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]));
    *r1 = vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1]));
    *r2 = vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0]));
    *r3 = vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1]));
}
#endif

// Winograd kernel transform G g Gᵀ for 3x3 kernel
static inline void winograd_transform_kernel(const float* kernel, float* output) {
    // G matrix for F(2x2, 3x3)
    // G = [1    0    0  ]
    //     [1/2  1/2  1/2]
    //     [1/2 -1/2  1/2]
    //     [0    0    1  ]
    
    float g[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            g[i][j] = kernel[i * 3 + j];
    
    // Gg
    float Gg[4][3];
    for (int j = 0; j < 3; j++) {
        Gg[0][j] = g[0][j];
        Gg[1][j] = 0.5f * (g[0][j] + g[1][j] + g[2][j]);
        Gg[2][j] = 0.5f * (g[0][j] - g[1][j] + g[2][j]);
        Gg[3][j] = g[2][j];
    }
    
    // (Gg)Gᵀ
    for (int i = 0; i < 4; i++) {
        output[i * 4 + 0] = Gg[i][0];
        output[i * 4 + 1] = 0.5f * (Gg[i][0] + Gg[i][1] + Gg[i][2]);
        output[i * 4 + 2] = 0.5f * (Gg[i][0] - Gg[i][1] + Gg[i][2]);
        output[i * 4 + 3] = Gg[i][2];
    }
}

// Transform 4x4 input tile: Bᵀ d B
// B = [1  0 -1  0]
//     [0  1  1  0]
//     [0 -1  1  0]
//     [0  1  0 -1]
static inline void winograd_transform_input(const float* input, float* output, int stride) {
#if RPITORCH_HAS_NEON
    // Load 4x4 tile
    float32x4_t d0 = vld1q_f32(&input[0 * stride]);
    float32x4_t d1 = vld1q_f32(&input[1 * stride]);
    float32x4_t d2 = vld1q_f32(&input[2 * stride]);
    float32x4_t d3 = vld1q_f32(&input[3 * stride]);
    
    // Bᵀ d (columnwise transform)
    float32x4_t bt_d0 = vsubq_f32(d0, d2);                    // d0 - d2
    float32x4_t bt_d1 = vaddq_f32(d1, d2);                    // d1 + d2
    float32x4_t bt_d2 = vsubq_f32(d2, d1);                    // d2 - d1
    float32x4_t bt_d3 = vsubq_f32(d1, d3);                    // d1 - d3
    
    // Transpose for row-wise transform
    transpose_4x4_neon(&bt_d0, &bt_d1, &bt_d2, &bt_d3);
    
    // (Bᵀ d) B (now rows are in registers)
    float32x4_t r0 = vsubq_f32(bt_d0, bt_d2);
    float32x4_t r1 = vaddq_f32(bt_d1, bt_d2);
    float32x4_t r2 = vsubq_f32(bt_d2, bt_d1);
    float32x4_t r3 = vsubq_f32(bt_d1, bt_d3);
    
    // Store result
    vst1q_f32(&output[0], r0);
    vst1q_f32(&output[4], r1);
    vst1q_f32(&output[8], r2);
    vst1q_f32(&output[12], r3);
#else
    // Scalar: Bᵀ d
    float temp[4][4];
    for (int j = 0; j < 4; j++) {
        temp[0][j] = input[0 * stride + j] - input[2 * stride + j];
        temp[1][j] = input[1 * stride + j] + input[2 * stride + j];
        temp[2][j] = input[2 * stride + j] - input[1 * stride + j];
        temp[3][j] = input[1 * stride + j] - input[3 * stride + j];
    }
    // (Bᵀ d) B
    for (int i = 0; i < 4; i++) {
        output[i * 4 + 0] = temp[i][0] - temp[i][2];
        output[i * 4 + 1] = temp[i][1] + temp[i][2];
        output[i * 4 + 2] = temp[i][2] - temp[i][1];
        output[i * 4 + 3] = temp[i][1] - temp[i][3];
    }
#endif
}

// Transform 4x4 accumulated tile to 2x2 output: Aᵀ m A
// A = [1  1  1  0]
//     [0  1 -1 -1]
static inline void winograd_transform_output(const float* input, float* output, int stride) {
#if RPITORCH_HAS_NEON
    // Load 4x4 tile
    float32x4_t m0 = vld1q_f32(&input[0]);
    float32x4_t m1 = vld1q_f32(&input[4]);
    float32x4_t m2 = vld1q_f32(&input[8]);
    float32x4_t m3 = vld1q_f32(&input[12]);
    
    // Aᵀ m (columnwise)
    float32x4_t t0 = vaddq_f32(vaddq_f32(m0, m1), m2);       // m0 + m1 + m2
    float32x4_t t1 = vsubq_f32(vsubq_f32(m1, m2), m3);       // m1 - m2 - m3
    
    // (Aᵀ m) A (extract 2x2 from first 2 rows)
    // Row 0: [t0[0]+t0[1]+t0[2], t0[1]-t0[2]-t0[3]]
    // Row 1: [t1[0]+t1[1]+t1[2], t1[1]-t1[2]-t1[3]]
    output[0 * stride + 0] = vgetq_lane_f32(t0, 0) + vgetq_lane_f32(t0, 1) + vgetq_lane_f32(t0, 2);
    output[0 * stride + 1] = vgetq_lane_f32(t0, 1) - vgetq_lane_f32(t0, 2) - vgetq_lane_f32(t0, 3);
    output[1 * stride + 0] = vgetq_lane_f32(t1, 0) + vgetq_lane_f32(t1, 1) + vgetq_lane_f32(t1, 2);
    output[1 * stride + 1] = vgetq_lane_f32(t1, 1) - vgetq_lane_f32(t1, 2) - vgetq_lane_f32(t1, 3);
#else
    float temp[2][4];
    for (int j = 0; j < 4; j++) {
        temp[0][j] = input[0 * 4 + j] + input[1 * 4 + j] + input[2 * 4 + j];
        temp[1][j] = input[1 * 4 + j] - input[2 * 4 + j] - input[3 * 4 + j];
    }
    output[0 * stride + 0] = temp[0][0] + temp[0][1] + temp[0][2];
    output[0 * stride + 1] = temp[0][1] - temp[0][2] - temp[0][3];
    output[1 * stride + 0] = temp[1][0] + temp[1][1] + temp[1][2];
    output[1 * stride + 1] = temp[1][1] - temp[1][2] - temp[1][3];
#endif
}

// Extract 4x4 input tile with boundary handling
static inline void extract_input_tile(const float* input, float* tile,
                                       int h_start, int w_start,
                                       int height, int width, int stride) {
    for (int i = 0; i < 4; i++) {
        int h = h_start + i;
        if (h >= 0 && h < height) {
            for (int j = 0; j < 4; j++) {
                int w = w_start + j;
                tile[i * 4 + j] = (w >= 0 && w < width) ? input[h * stride + w] : 0.0f;
            }
        } else {
            for (int j = 0; j < 4; j++) tile[i * 4 + j] = 0.0f;
        }
    }
}

// Winograd convolution for 3x3 kernels, stride=1, padding=1
void conv2d_winograd_3x3(
    const float* input,
    const float* kernel,
    float* output,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int stride,
    int padding
) {
    if (stride != 1 || padding != 1) return;  // Fallback to caller
    
    int out_h = height;
    int out_w = width;
    int num_tiles_h = (out_h + 1) / 2;
    int num_tiles_w = (out_w + 1) / 2;
    
    // Pre-transform all kernels using G g Gᵀ
    float* kernel_transformed = (float*)rpitorch_aligned_alloc(64,
        out_channels * in_channels * 16 * sizeof(float));
    
    #pragma omp parallel for collapse(2)
    for (int oc = 0; oc < out_channels; oc++) {
        for (int ic = 0; ic < in_channels; ic++) {
            winograd_transform_kernel(
                &kernel[oc * in_channels * 9 + ic * 9],
                &kernel_transformed[(oc * in_channels + ic) * 16]
            );
        }
    }
    
    // Initialize output to zero
    memset(output, 0, out_channels * out_h * out_w * sizeof(float));
    
    // Process tiles
    #pragma omp parallel for collapse(2)
    for (int oc = 0; oc < out_channels; oc++) {
        for (int th = 0; th < num_tiles_h; th++) {
            for (int tw = 0; tw < num_tiles_w; tw++) {
                float tile_accum[16] __attribute__((aligned(64))) = {0};
                
                int h_start = th * 2 - padding;
                int w_start = tw * 2 - padding;
                
                for (int ic = 0; ic < in_channels; ic++) {
                    float input_tile[16] __attribute__((aligned(64)));
                    float input_transformed[16] __attribute__((aligned(64)));
                    
                    // Prefetch next channel
                    if (ic + 1 < in_channels) {
                        __builtin_prefetch(&input[(ic + 1) * height * width], 0, 1);
                    }
                    
                    // Extract and transform input tile
                    extract_input_tile(&input[ic * height * width], input_tile,
                                       h_start, w_start, height, width, width);
                    winograd_transform_input(input_tile, input_transformed, 4);
                    
                    // Element-wise multiply-accumulate
                    const float* kt = &kernel_transformed[(oc * in_channels + ic) * 16];
#if RPITORCH_HAS_NEON
                    float32x4_t* it = (float32x4_t*)input_transformed;
                    float32x4_t* ktv = (float32x4_t*)kt;
                    float32x4_t* acc = (float32x4_t*)tile_accum;
                    acc[0] = vfmaq_f32(acc[0], it[0], ktv[0]);
                    acc[1] = vfmaq_f32(acc[1], it[1], ktv[1]);
                    acc[2] = vfmaq_f32(acc[2], it[2], ktv[2]);
                    acc[3] = vfmaq_f32(acc[3], it[3], ktv[3]);
#else
                    for (int i = 0; i < 16; i++) {
                        tile_accum[i] += input_transformed[i] * kt[i];
                    }
#endif
                }
                
                // Inverse transform and accumulate to output
                int out_y = th * 2;
                int out_x = tw * 2;
                
                float out_tile[4];  // 2x2 output
                winograd_transform_output(tile_accum, out_tile, 2);
                
                // Write 2x2 tile to output (with boundary check)
                if (out_y < out_h && out_x < out_w)
                    output[oc * out_h * out_w + out_y * out_w + out_x] += out_tile[0];
                if (out_y < out_h && out_x + 1 < out_w)
                    output[oc * out_h * out_w + out_y * out_w + out_x + 1] += out_tile[1];
                if (out_y + 1 < out_h && out_x < out_w)
                    output[oc * out_h * out_w + (out_y + 1) * out_w + out_x] += out_tile[2];
                if (out_y + 1 < out_h && out_x + 1 < out_w)
                    output[oc * out_h * out_w + (out_y + 1) * out_w + out_x + 1] += out_tile[3];
            }
        }
    }
    
    rpitorch_aligned_free(kernel_transformed);
}


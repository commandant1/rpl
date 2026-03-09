/*
 * rpl_gpu_ops_test.c
 * Comprehensive test for all GPU ops added to the RPL library.
 *
 * Build:  cmake .. -DUSE_GPU=ON && make rpl_gpu_ops_test
 * Run:    ./tests/rpl_gpu_ops_test
 *
 * On a machine without a real GLES 3.1 GPU, rpl_gpu_init() returns false
 * and the test exits with code 0 (skipped), not a failure.
 */

#include "rpl.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

/* ------------------------------------------------------------------ */
#define CHECK(cond, msg) \
    do { if (!(cond)) { printf("FAIL [%s] at line %d: %s\n", __func__, __LINE__, msg); return false; } } while(0)

#define ABS_CLOSE(a, b, eps) (fabsf((a)-(b)) <= (eps))

static float ref_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/* ======================================================
 * 1. Tiled 8x8 GEMM
 * ====================================================== */
static bool test_gemm(void) {
    uint32_t M = 64, K = 128, N = 64;
    uint32_t sA[] = {M, K}, sB[] = {K, N}, sC[] = {M, N};
    Tensor* A = tensor_create(2, sA, false);
    Tensor* B = tensor_create(2, sB, false);
    Tensor* C = tensor_create(2, sC, false);

    /* A = all 1s, B = identity => C should be all K (128.0) */
    for (uint32_t i = 0; i < M*K; i++) A->data[i] = 1.0f;
    for (uint32_t r = 0; r < K; r++)
        for (uint32_t c = 0; c < N; c++)
            B->data[r*N+c] = (r == c) ? 1.0f : 0.0f;

    tensor_matmul_gpu(C, A, B);
    tensor_from_gpu(C);

    for (uint32_t i = 0; i < M*N; i++)
        CHECK(ABS_CLOSE(C->data[i], (float)K, 0.5f), "GEMM value mismatch");

    tensor_free(A); tensor_free(B); tensor_free(C);
    printf("PASS  GEMM (8x8 tiled, %ux%u @ %ux%u)\n", M, K, K, N);
    return true;
}

/* ======================================================
 * 2. In-place ReLU (single SSBO)
 * ====================================================== */
static bool test_relu_inplace(void) {
    const int N = 1024;
    uint32_t shape[] = {(uint32_t)N};
    Tensor* t = tensor_create(1, shape, false);
    for (int i = 0; i < N; i++) t->data[i] = (float)(i - N/2);  /* -512..511 */

    tensor_to_gpu(t);
    tensor_relu_inplace_gpu(t);
    tensor_from_gpu(t);

    for (int i = 0; i < N; i++) {
        float expected = (i - N/2) > 0 ? (float)(i - N/2) : 0.0f;
        CHECK(ABS_CLOSE(t->data[i], expected, 1e-4f), "in-place ReLU mismatch");
    }
    tensor_free(t);
    printf("PASS  In-place ReLU\n");
    return true;
}

/* ======================================================
 * 3. ReLU (out-of-place) — auto-dispatch via tensor_relu
 * ====================================================== */
static bool test_relu_dispatch(void) {
    const int N = 512;
    uint32_t shape[] = {(uint32_t)N};
    Tensor* in = tensor_create(1, shape, false);
    Tensor* out = tensor_create(1, shape, false);
    for (int i = 0; i < N; i++) in->data[i] = (float)(i - N/2);

    tensor_to_gpu(in);
    tensor_relu_gpu(out, in);
    tensor_from_gpu(out);

    for (int i = 0; i < N; i++) {
        float expected = (i - N/2) > 0 ? (float)(i - N/2) : 0.0f;
        CHECK(ABS_CLOSE(out->data[i], expected, 1e-4f), "ReLU dispatch mismatch");
    }
    tensor_free(in); tensor_free(out);
    printf("PASS  ReLU (out-of-place GPU)\n");
    return true;
}

/* ======================================================
 * 4. LeakyReLU
 * ====================================================== */
static bool test_leaky_relu(void) {
    const int N = 512;
    const float slope = 0.1f;
    uint32_t shape[] = {(uint32_t)N};
    Tensor* in  = tensor_create(1, shape, false);
    Tensor* out = tensor_create(1, shape, false);
    for (int i = 0; i < N; i++) in->data[i] = (float)(i - N/2);

    tensor_to_gpu(in);
    tensor_leaky_relu_gpu(out, in, slope);
    tensor_from_gpu(out);

    for (int i = 0; i < N; i++) {
        float x = (float)(i - N/2);
        float expected = x >= 0.0f ? x : slope * x;
        CHECK(ABS_CLOSE(out->data[i], expected, 1e-4f), "LeakyReLU mismatch");
    }
    tensor_free(in); tensor_free(out);
    printf("PASS  LeakyReLU (slope=%.2f)\n", slope);
    return true;
}

/* ======================================================
 * 5. ELU
 * ====================================================== */
static bool test_elu(void) {
    const int N = 256;
    const float alpha = 1.0f;
    uint32_t shape[] = {(uint32_t)N};
    Tensor* in  = tensor_create(1, shape, false);
    Tensor* out = tensor_create(1, shape, false);
    for (int i = 0; i < N; i++) in->data[i] = (float)(i - N/2) / 32.0f;

    tensor_to_gpu(in);
    tensor_elu_gpu(out, in, alpha);
    tensor_from_gpu(out);

    for (int i = 0; i < N; i++) {
        float x = (float)(i - N/2) / 32.0f;
        float expected = x >= 0.0f ? x : alpha * (expf(x) - 1.0f);
        CHECK(ABS_CLOSE(out->data[i], expected, 1e-3f), "ELU mismatch");
    }
    tensor_free(in); tensor_free(out);
    printf("PASS  ELU (alpha=%.1f)\n", alpha);
    return true;
}

/* ======================================================
 * 6. Swish
 * ====================================================== */
static bool test_swish(void) {
    float vals[] = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f};
    const int N = 5;
    uint32_t shape[] = {(uint32_t)N};
    Tensor* in  = tensor_create(1, shape, false);
    Tensor* out = tensor_create(1, shape, false);
    for (int i = 0; i < N; i++) in->data[i] = vals[i];

    tensor_to_gpu(in);
    tensor_swish_gpu(out, in);
    tensor_from_gpu(out);

    for (int i = 0; i < N; i++) {
        float expected = vals[i] * ref_sigmoid(vals[i]);
        CHECK(ABS_CLOSE(out->data[i], expected, 1e-3f), "Swish mismatch");
    }
    tensor_free(in); tensor_free(out);
    printf("PASS  Swish / SiLU\n");
    return true;
}

/* ======================================================
 * 7. Sigmoid
 * ====================================================== */
static bool test_sigmoid(void) {
    const int N = 256;
    uint32_t shape[] = {(uint32_t)N};
    Tensor* in  = tensor_create(1, shape, false);
    Tensor* out = tensor_create(1, shape, false);
    for (int i = 0; i < N; i++) in->data[i] = (float)(i - N/2) / 32.0f;

    tensor_to_gpu(in);
    tensor_sigmoid_gpu(out, in);
    tensor_from_gpu(out);

    for (int i = 0; i < N; i++) {
        float x = (float)(i - N/2) / 32.0f;
        float expected = ref_sigmoid(x);
        CHECK(ABS_CLOSE(out->data[i], expected, 2e-3f), "Sigmoid mismatch");
    }
    tensor_free(in); tensor_free(out);
    printf("PASS  Sigmoid\n");
    return true;
}

/* ======================================================
 * 8. Tanh
 * ====================================================== */
static bool test_tanh(void) {
    const int N = 128;
    uint32_t shape[] = {(uint32_t)N};
    Tensor* in  = tensor_create(1, shape, false);
    Tensor* out = tensor_create(1, shape, false);
    for (int i = 0; i < N; i++) in->data[i] = (float)(i - N/2) / 32.0f;

    tensor_to_gpu(in);
    tensor_tanh_gpu(out, in);
    tensor_from_gpu(out);

    for (int i = 0; i < N; i++) {
        float expected = tanhf(in->data[i]);
        CHECK(ABS_CLOSE(out->data[i], expected, 1e-3f), "Tanh mismatch");
    }
    tensor_free(in); tensor_free(out);
    printf("PASS  Tanh\n");
    return true;
}

/* ======================================================
 * 9. GELU
 * ====================================================== */
static bool test_gelu(void) {
    const int N = 128;
    uint32_t shape[] = {(uint32_t)N};
    Tensor* in  = tensor_create(1, shape, false);
    Tensor* out = tensor_create(1, shape, false);
    for (int i = 0; i < N; i++) in->data[i] = (float)(i - N/2) / 32.0f;

    tensor_to_gpu(in);
    tensor_gelu_gpu(out, in);
    tensor_from_gpu(out);

    for (int i = 0; i < N; i++) {
        float x = in->data[i];
        float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
        float expected = 0.5f * x * (1.0f + tanhf(inner));
        CHECK(ABS_CLOSE(out->data[i], expected, 1e-3f), "GELU mismatch");
    }
    tensor_free(in); tensor_free(out);
    printf("PASS  GELU\n");
    return true;
}

/* ======================================================
 * 10. SELU
 * ====================================================== */
static bool test_selu(void) {
    const float lam = 1.0507009873554804934f;
    const float alp = 1.6732632423543772848f;
    float vals[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    const int N = 5;
    uint32_t shape[] = {(uint32_t)N};
    Tensor* in  = tensor_create(1, shape, false);
    Tensor* out = tensor_create(1, shape, false);
    for (int i = 0; i < N; i++) in->data[i] = vals[i];

    tensor_to_gpu(in);
    tensor_selu_gpu(out, in);
    tensor_from_gpu(out);

    for (int i = 0; i < N; i++) {
        float x = vals[i];
        float expected = x >= 0.0f ? lam*x : lam*alp*(expf(x)-1.0f);
        CHECK(ABS_CLOSE(out->data[i], expected, 1e-3f), "SELU mismatch");
    }
    tensor_free(in); tensor_free(out);
    printf("PASS  SELU\n");
    return true;
}

/* ======================================================
 * 11. Mish
 * ====================================================== */
static bool test_mish(void) {
    float vals[] = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f};
    const int N = 5;
    uint32_t shape[] = {(uint32_t)N};
    Tensor* in  = tensor_create(1, shape, false);
    Tensor* out = tensor_create(1, shape, false);
    for (int i = 0; i < N; i++) in->data[i] = vals[i];

    tensor_to_gpu(in);
    tensor_mish_gpu(out, in);
    tensor_from_gpu(out);

    for (int i = 0; i < N; i++) {
        float x = vals[i];
        float expected = x * tanhf(logf(1.0f + expf(x)));
        CHECK(ABS_CLOSE(out->data[i], expected, 2e-3f), "Mish mismatch");
    }
    tensor_free(in); tensor_free(out);
    printf("PASS  Mish\n");
    return true;
}

/* ======================================================
 * 12. Hardswish
 * ====================================================== */
static bool test_hardswish(void) {
    float vals[] = {-4.0f, -1.5f, 0.0f, 1.5f, 4.0f};
    const int N = 5;
    uint32_t shape[] = {(uint32_t)N};
    Tensor* in  = tensor_create(1, shape, false);
    Tensor* out = tensor_create(1, shape, false);
    for (int i = 0; i < N; i++) in->data[i] = vals[i];

    tensor_to_gpu(in);
    tensor_hardswish_gpu(out, in);
    tensor_from_gpu(out);

    for (int i = 0; i < N; i++) {
        float x = vals[i];
        float clip = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
        float expected = x * clip / 6.0f;
        CHECK(ABS_CLOSE(out->data[i], expected, 1e-4f), "Hardswish mismatch");
    }
    tensor_free(in); tensor_free(out);
    printf("PASS  Hardswish\n");
    return true;
}

/* ======================================================
 * 13. Hardsigmoid
 * ====================================================== */
static bool test_hardsigmoid(void) {
    float vals[] = {-4.0f, -1.5f, 0.0f, 1.5f, 4.0f};
    const int N = 5;
    uint32_t shape[] = {(uint32_t)N};
    Tensor* in  = tensor_create(1, shape, false);
    Tensor* out = tensor_create(1, shape, false);
    for (int i = 0; i < N; i++) in->data[i] = vals[i];

    tensor_to_gpu(in);
    tensor_hardsigmoid_gpu(out, in);
    tensor_from_gpu(out);

    for (int i = 0; i < N; i++) {
        float x = vals[i];
        float expected = fminf(fmaxf(x / 6.0f + 0.5f, 0.0f), 1.0f);
        CHECK(ABS_CLOSE(out->data[i], expected, 1e-4f), "Hardsigmoid mismatch");
    }
    tensor_free(in); tensor_free(out);
    printf("PASS  Hardsigmoid\n");
    return true;
}

/* ======================================================
 * 14. Softplus
 * ====================================================== */
static bool test_softplus(void) {
    float vals[] = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f};
    const int N = 5;
    const float beta = 1.0f, threshold = 20.0f;
    uint32_t shape[] = {(uint32_t)N};
    Tensor* in  = tensor_create(1, shape, false);
    Tensor* out = tensor_create(1, shape, false);
    for (int i = 0; i < N; i++) in->data[i] = vals[i];

    tensor_to_gpu(in);
    tensor_softplus_gpu(out, in, beta, threshold);
    tensor_from_gpu(out);

    for (int i = 0; i < N; i++) {
        float x = vals[i];
        float bx = beta * x;
        float expected = (bx > threshold) ? x : logf(1.0f + expf(bx)) / beta;
        CHECK(ABS_CLOSE(out->data[i], expected, 1e-3f), "Softplus mismatch");
    }
    tensor_free(in); tensor_free(out);
    printf("PASS  Softplus\n");
    return true;
}

/* ======================================================
 * 15. Softmax — row sums == 1, ordering preserved
 * ====================================================== */
static bool test_softmax(void) {
    /* 4 rows, 8 classes */
    const int rows = 4, cols = 8;
    uint32_t shape[] = {(uint32_t)rows, (uint32_t)cols};
    Tensor* in  = tensor_create(2, shape, false);
    Tensor* out = tensor_create(2, shape, false);

    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            in->data[r*cols + c] = (float)(r * cols + c) * 0.5f - 4.0f;

    tensor_to_gpu(in);
    tensor_softmax_gpu(out, in, 1);
    tensor_from_gpu(out);

    for (int r = 0; r < rows; r++) {
        float rowsum = 0.0f;
        for (int c = 0; c < cols; c++) {
            rowsum += out->data[r*cols + c];
            CHECK(out->data[r*cols+c] > 0.0f, "Softmax output must be positive");
        }
        CHECK(ABS_CLOSE(rowsum, 1.0f, 1e-4f), "Softmax row sum != 1");
    }
    tensor_free(in); tensor_free(out);
    printf("PASS  Softmax (%dx%d)\n", rows, cols);
    return true;
}

/* ======================================================
 * 16. Log-Softmax
 * ====================================================== */
static bool test_log_softmax(void) {
    const int rows = 2, cols = 4;
    uint32_t shape[] = {(uint32_t)rows, (uint32_t)cols};
    Tensor* in  = tensor_create(2, shape, false);
    Tensor* out = tensor_create(2, shape, false);
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f,
                    0.1f, 0.2f, 0.3f, 0.4f};
    memcpy(in->data, data, sizeof(data));

    tensor_to_gpu(in);
    tensor_log_softmax_gpu(out, in);
    tensor_from_gpu(out);

    /* exp(log_softmax) should sum to 1 per row */
    for (int r = 0; r < rows; r++) {
        float rowsum = 0.0f;
        for (int c = 0; c < cols; c++) rowsum += expf(out->data[r*cols + c]);
        CHECK(ABS_CLOSE(rowsum, 1.0f, 1e-4f), "Log-Softmax exp-sum != 1");
    }
    tensor_free(in); tensor_free(out);
    printf("PASS  Log-Softmax\n");
    return true;
}

/* ======================================================
 * 17. Scale (inplace)
 * ====================================================== */
static bool test_scale(void) {
    const int N = 256;
    const float scalar = 3.14f;
    uint32_t shape[] = {(uint32_t)N};
    Tensor* t = tensor_create(1, shape, false);
    for (int i = 0; i < N; i++) t->data[i] = (float)i;

    tensor_to_gpu(t);
    tensor_scale_gpu(t, scalar);
    tensor_from_gpu(t);

    for (int i = 0; i < N; i++)
        CHECK(ABS_CLOSE(t->data[i], (float)i * scalar, 1e-3f), "Scale mismatch");

    tensor_free(t);
    printf("PASS  Scale inplace (%.4f)\n", scalar);
    return true;
}

/* ======================================================
 * 18. Binary ops (add / sub / mul / div)
 * ====================================================== */
static bool test_binary_ops(void) {
    const int N = 256;
    uint32_t shape[] = {(uint32_t)N};
    Tensor* a   = tensor_create(1, shape, false);
    Tensor* b   = tensor_create(1, shape, false);
    Tensor* out = tensor_create(1, shape, false);
    for (int i = 0; i < N; i++) { a->data[i] = (float)(i+1); b->data[i] = 2.0f; }

    tensor_to_gpu(a); tensor_to_gpu(b);

    tensor_add_gpu(out, a, b); tensor_from_gpu(out);
    for (int i = 0; i < N; i++)
        CHECK(ABS_CLOSE(out->data[i], (float)(i+1)+2.0f, 1e-4f), "Add mismatch");

    tensor_sub_gpu(out, a, b); tensor_from_gpu(out);
    for (int i = 0; i < N; i++)
        CHECK(ABS_CLOSE(out->data[i], (float)(i+1)-2.0f, 1e-4f), "Sub mismatch");

    tensor_mul_gpu(out, a, b); tensor_from_gpu(out);
    for (int i = 0; i < N; i++)
        CHECK(ABS_CLOSE(out->data[i], (float)(i+1)*2.0f, 1e-4f), "Mul mismatch");

    tensor_div_gpu(out, a, b); tensor_from_gpu(out);
    for (int i = 0; i < N; i++)
        CHECK(ABS_CLOSE(out->data[i], (float)(i+1)/2.0f, 1e-4f), "Div mismatch");

    tensor_free(a); tensor_free(b); tensor_free(out);
    printf("PASS  Binary ops (add/sub/mul/div)\n");
    return true;
}

/* ======================================================
 * 19. Conv2D via GL_TEXTURE_2D
 *     Identity 1x1 kernel => output == input
 * ====================================================== */
static bool test_conv2d(void) {
    /* 2 input channels, 4x4 spatial, 2 output channels, 1x1 kernel */
    int C_in = 2, H = 4, W = 4, C_out = 2, kH = 1, kW = 1;
    uint32_t shapeIn[]   = {(uint32_t)C_in, (uint32_t)H, (uint32_t)W};
    uint32_t shapeKern[] = {(uint32_t)C_out, (uint32_t)C_in, (uint32_t)kH, (uint32_t)kW};
    uint32_t shapeOut[]  = {(uint32_t)C_out, (uint32_t)H, (uint32_t)W};

    Tensor* in   = tensor_create(3, shapeIn,   false);
    Tensor* kern = tensor_create(4, shapeKern, false);
    Tensor* out  = tensor_create(3, shapeOut,  false);

    /* Fill input with known values */
    for (int i = 0; i < C_in*H*W; i++) in->data[i] = (float)(i + 1);

    /* Identity kernel: for each output channel oc, only weight [oc, oc, 0, 0] = 1 */
    memset(kern->data, 0, C_out * C_in * kH * kW * sizeof(float));
    for (int oc = 0; oc < C_out && oc < C_in; oc++)
        kern->data[oc * C_in * kH * kW + oc * kH * kW] = 1.0f;

    tensor_conv2d_gpu(out, in, kern, kH, kW, 1, 0);
    tensor_from_gpu(out);

    /* With identity 1x1 kernel: out[oc, h, w] == in[oc, h, w] */
    bool ok = true;
    for (int oc = 0; oc < C_out && oc < C_in; oc++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                float expected = in->data[oc*H*W + h*W + w];
                float got      = out->data[oc*H*W + h*W + w];
                if (!ABS_CLOSE(got, expected, 1e-3f)) {
                    printf("  Conv2D mismatch at oc=%d h=%d w=%d: expected %.2f got %.2f\n",
                           oc, h, w, expected, got);
                    ok = false;
                }
            }
        }
    }
    CHECK(ok, "Conv2D identity kernel failed");

    tensor_free(in); tensor_free(kern); tensor_free(out);
    printf("PASS  Conv2D via GL_TEXTURE_2D (identity 1x1 kernel, %dch→%dch, %dx%d)\n",
           C_in, C_out, H, W);
    return true;
}

/* ======================================================
 * 20. Math Unary ops (sin, cos, exp, log, sqrt, abs, neg)
 * ====================================================== */
static bool test_math_unary(void) {
    /* Use small positive values safe for all ops */
    float vals[] = {0.1f, 0.5f, 1.0f, 2.0f, 3.0f};
    const int N = 5;
    uint32_t shape[] = {(uint32_t)N};
    Tensor* in  = tensor_create(1, shape, false);
    Tensor* out = tensor_create(1, shape, false);
    for (int i = 0; i < N; i++) in->data[i] = vals[i];
    tensor_to_gpu(in);

    /* sin */
    tensor_sin_gpu(out, in); tensor_from_gpu(out);
    for (int i = 0; i < N; i++) CHECK(ABS_CLOSE(out->data[i], sinf(vals[i]), 1e-4f), "sin mismatch");

    /* cos */
    tensor_cos_gpu(out, in); tensor_from_gpu(out);
    for (int i = 0; i < N; i++) CHECK(ABS_CLOSE(out->data[i], cosf(vals[i]), 1e-4f), "cos mismatch");

    /* exp */
    tensor_exp_gpu(out, in); tensor_from_gpu(out);
    for (int i = 0; i < N; i++) CHECK(ABS_CLOSE(out->data[i], expf(vals[i]), 1e-4f), "exp mismatch");

    /* log */
    tensor_log_gpu(out, in); tensor_from_gpu(out);
    for (int i = 0; i < N; i++) CHECK(ABS_CLOSE(out->data[i], logf(vals[i]), 1e-4f), "log mismatch");

    /* sqrt */
    tensor_sqrt_gpu(out, in); tensor_from_gpu(out);
    for (int i = 0; i < N; i++) CHECK(ABS_CLOSE(out->data[i], sqrtf(vals[i]), 1e-4f), "sqrt mismatch");

    /* abs (use negative values, fresh tensor) */
    Tensor* neg_in  = tensor_create(1, shape, false);
    Tensor* neg_out = tensor_create(1, shape, false);
    for (int i = 0; i < N; i++) neg_in->data[i] = -vals[i];
    tensor_to_gpu(neg_in);
    tensor_abs_gpu(neg_out, neg_in); tensor_from_gpu(neg_out);
    for (int i = 0; i < N; i++) CHECK(ABS_CLOSE(neg_out->data[i], vals[i], 1e-4f), "abs mismatch");

    /* neg: negate the negative values to get positive -> compare to vals */
    tensor_neg_gpu(neg_out, neg_in); tensor_from_gpu(neg_out);
    for (int i = 0; i < N; i++) CHECK(ABS_CLOSE(neg_out->data[i], vals[i], 1e-4f), "neg mismatch");

    tensor_free(neg_in); tensor_free(neg_out);
    tensor_free(in); tensor_free(out);
    printf("PASS  Math unary GPU ops (sin/cos/exp/log/sqrt/abs/neg)\n");
    return true;
}

/* ======================================================
 * 21. Math Binary ops (pow, maximum, minimum)
 * ====================================================== */
static bool test_math_binary(void) {
    float a_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_vals[] = {2.0f, 2.0f, 2.0f, 2.0f};
    const int N = 4;
    uint32_t shape[] = {(uint32_t)N};
    Tensor* a   = tensor_create(1, shape, false);
    Tensor* b   = tensor_create(1, shape, false);
    Tensor* out = tensor_create(1, shape, false);
    for (int i = 0; i < N; i++) { a->data[i] = a_vals[i]; b->data[i] = b_vals[i]; }
    tensor_to_gpu(a); tensor_to_gpu(b);

    /* pow: a^2 */
    tensor_pow_gpu(out, a, b); tensor_from_gpu(out);
    for (int i = 0; i < N; i++)
        CHECK(ABS_CLOSE(out->data[i], powf(a_vals[i], b_vals[i]), 1e-4f), "pow mismatch");

    /* maximum */
    float c_vals[] = {0.5f, 3.0f, 2.5f, 5.0f};
    Tensor* c = tensor_create(1, shape, false);
    for (int i = 0; i < N; i++) c->data[i] = c_vals[i];
    tensor_to_gpu(c);
    tensor_maximum_gpu(out, a, c); tensor_from_gpu(out);
    for (int i = 0; i < N; i++)
        CHECK(ABS_CLOSE(out->data[i], fmaxf(a_vals[i], c_vals[i]), 1e-4f), "maximum mismatch");

    /* minimum */
    tensor_minimum_gpu(out, a, c); tensor_from_gpu(out);
    for (int i = 0; i < N; i++)
        CHECK(ABS_CLOSE(out->data[i], fminf(a_vals[i], c_vals[i]), 1e-4f), "minimum mismatch");

    tensor_free(a); tensor_free(b); tensor_free(c); tensor_free(out);
    printf("PASS  Math binary GPU ops (pow/maximum/minimum)\n");
    return true;
}

/* ======================================================
 * 22. Clamp / Hardtanh
 * ====================================================== */
static bool test_clamp_gpu(void) {
    float vals[] = {-5.0f, -1.0f, 0.0f, 1.0f, 5.0f};
    const int N = 5;
    const float lo = -1.0f, hi = 1.0f;
    uint32_t shape[] = {(uint32_t)N};
    Tensor* in  = tensor_create(1, shape, false);
    Tensor* out = tensor_create(1, shape, false);
    for (int i = 0; i < N; i++) in->data[i] = vals[i];
    tensor_to_gpu(in);
    tensor_clamp_gpu(out, in, lo, hi);
    tensor_from_gpu(out);
    for (int i = 0; i < N; i++) {
        float expected = fmaxf(lo, fminf(hi, vals[i]));
        CHECK(ABS_CLOSE(out->data[i], expected, 1e-4f), "clamp mismatch");
    }
    tensor_free(in); tensor_free(out);
    printf("PASS  Clamp GPU [-1, 1]\n");
    return true;
}

/* ======================================================
 * 23. CELU
 * ====================================================== */
static bool test_celu_gpu(void) {
    float vals[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    const int N = 5;
    const float alpha = 1.0f;
    uint32_t shape[] = {(uint32_t)N};
    Tensor* in  = tensor_create(1, shape, false);
    Tensor* out = tensor_create(1, shape, false);
    for (int i = 0; i < N; i++) in->data[i] = vals[i];
    tensor_to_gpu(in);
    tensor_celu_gpu(out, in, alpha);
    tensor_from_gpu(out);
    for (int i = 0; i < N; i++) {
        float x = vals[i];
        float expected = fmaxf(0.0f, x) + fminf(0.0f, alpha * (expf(x / alpha) - 1.0f));
        CHECK(ABS_CLOSE(out->data[i], expected, 2e-3f), "CELU mismatch");
    }
    tensor_free(in); tensor_free(out);
    printf("PASS  CELU GPU (alpha=1.0)\n");
    return true;
}

/* ======================================================
 * 24. Softsign
 * ====================================================== */
static bool test_softsign_gpu(void) {
    float vals[] = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f};
    const int N = 5;
    uint32_t shape[] = {(uint32_t)N};
    Tensor* in  = tensor_create(1, shape, false);
    Tensor* out = tensor_create(1, shape, false);
    for (int i = 0; i < N; i++) in->data[i] = vals[i];
    tensor_to_gpu(in);
    tensor_softsign_gpu(out, in);
    tensor_from_gpu(out);
    for (int i = 0; i < N; i++) {
        float x = vals[i];
        float expected = x / (1.0f + fabsf(x));
        CHECK(ABS_CLOSE(out->data[i], expected, 1e-4f), "Softsign mismatch");
    }
    tensor_free(in); tensor_free(out);
    printf("PASS  Softsign GPU\n");
    return true;
}

/* ======================================================
 * 25. RReLU (eval: slope = mean of lower/upper)
 * ====================================================== */
static bool test_rrelu_gpu(void) {
    float vals[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    const int N = 5;
    const float lower = 0.1f, upper = 0.3f;
    float slope = (lower + upper) * 0.5f;
    uint32_t shape[] = {(uint32_t)N};
    Tensor* in  = tensor_create(1, shape, false);
    Tensor* out = tensor_create(1, shape, false);
    for (int i = 0; i < N; i++) in->data[i] = vals[i];
    tensor_to_gpu(in);
    tensor_rrelu_gpu(out, in, lower, upper);
    tensor_from_gpu(out);
    for (int i = 0; i < N; i++) {
        float x = vals[i];
        float expected = (x >= 0.0f) ? x : slope * x;
        CHECK(ABS_CLOSE(out->data[i], expected, 1e-4f), "RReLU mismatch");
    }
    tensor_free(in); tensor_free(out);
    printf("PASS  RReLU GPU (lower=%.1f upper=%.1f)\n", lower, upper);
    return true;
}

/* ======================================================
 * 26. Threshold
 * ====================================================== */
static bool test_threshold_gpu(void) {
    float vals[] = {-1.0f, 0.5f, 1.0f, 2.0f, 3.0f};
    const int N = 5;
    const float thresh = 1.0f, rep = -99.0f;
    uint32_t shape[] = {(uint32_t)N};
    Tensor* in  = tensor_create(1, shape, false);
    Tensor* out = tensor_create(1, shape, false);
    for (int i = 0; i < N; i++) in->data[i] = vals[i];
    tensor_to_gpu(in);
    tensor_threshold_gpu(out, in, thresh, rep);
    tensor_from_gpu(out);
    for (int i = 0; i < N; i++) {
        float expected = (vals[i] > thresh) ? vals[i] : rep;
        CHECK(ABS_CLOSE(out->data[i], expected, 1e-4f), "Threshold mismatch");
    }
    tensor_free(in); tensor_free(out);
    printf("PASS  Threshold GPU (thresh=%.1f val=%.1f)\n", thresh, rep);
    return true;
}

/* ======================================================
 * Main
 * ====================================================== */
int main(void) {
    printf("RPL GPU Ops Test Suite\n");
    printf("======================\n");

    if (!rpl_gpu_init()) {
        printf("SKIP — GPU not available (no GLES 3.1 device found).\n");
        return 0;
    }

    int passed = 0, total = 0;

#define RUN(fn) do { total++; if (fn()) passed++; } while(0)

    RUN(test_gemm);
    RUN(test_relu_inplace);
    RUN(test_relu_dispatch);
    RUN(test_leaky_relu);
    RUN(test_elu);
    RUN(test_swish);
    RUN(test_sigmoid);
    RUN(test_tanh);
    RUN(test_gelu);
    RUN(test_selu);
    RUN(test_mish);
    RUN(test_hardswish);
    RUN(test_hardsigmoid);
    RUN(test_softplus);
    RUN(test_softmax);
    RUN(test_log_softmax);
    RUN(test_scale);
    RUN(test_binary_ops);
    RUN(test_conv2d);
    /* New ops */
    RUN(test_math_unary);
    RUN(test_math_binary);
    RUN(test_clamp_gpu);
    RUN(test_celu_gpu);
    RUN(test_softsign_gpu);
    RUN(test_rrelu_gpu);
    RUN(test_threshold_gpu);

    printf("\n======================\n");
    printf("Results: %d / %d passed\n", passed, total);

    rpl_gpu_shutdown();
    return (passed == total) ? 0 : 1;
}

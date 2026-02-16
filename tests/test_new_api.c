/*
 * test_new_api.c — Comprehensive tests for all new RPL functions
 */
#include "rpl.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

static int tests_passed = 0, tests_failed = 0;

#define ASSERT_CLOSE(name, got, expected, tol) do { \
    float _g = (got), _e = (expected); \
    if (fabsf(_g - _e) <= (tol) || (isnan(_g) && isnan(_e))) { tests_passed++; } \
    else { printf("FAIL %s: got %.6f, expected %.6f\n", name, _g, _e); tests_failed++; } \
} while(0)

#define ASSERT_TRUE(name, cond) do { \
    if (cond) { tests_passed++; } \
    else { printf("FAIL %s\n", name); tests_failed++; } \
} while(0)

static Tensor* make_1d(const float* vals, uint32_t n) {
    uint32_t s[1] = {n};
    Tensor* t = tensor_create(1, s, false);
    memcpy(t->data, vals, n * sizeof(float));
    return t;
}

static Tensor* make_2d(const float* vals, uint32_t r, uint32_t c) {
    uint32_t s[2] = {r, c};
    Tensor* t = tensor_create(2, s, false);
    memcpy(t->data, vals, r * c * sizeof(float));
    return t;
}

// ===== Math Tests =====
static void test_math(void) {
    printf("--- Math ---\n");
    float v[] = {0.0f, 0.5f, 1.0f, -1.0f};
    Tensor* t = make_1d(v, 4);

    // sin
    Tensor* r = tensor_sin(t);
    ASSERT_CLOSE("sin(0)", r->data[0], 0.0f, 1e-5f);
    ASSERT_CLOSE("sin(0.5)", r->data[1], sinf(0.5f), 1e-5f);
    tensor_free(r);

    // cos
    r = tensor_cos(t);
    ASSERT_CLOSE("cos(0)", r->data[0], 1.0f, 1e-5f);
    tensor_free(r);

    // exp
    r = tensor_exp(t);
    ASSERT_CLOSE("exp(0)", r->data[0], 1.0f, 1e-5f);
    ASSERT_CLOSE("exp(1)", r->data[2], expf(1.0f), 1e-5f);
    tensor_free(r);

    // log
    float lv[] = {1.0f, 2.718281828f, 10.0f, 100.0f};
    Tensor* lt = make_1d(lv, 4);
    r = tensor_log(lt);
    ASSERT_CLOSE("log(1)", r->data[0], 0.0f, 1e-5f);
    ASSERT_CLOSE("log(e)", r->data[1], 1.0f, 1e-3f);
    tensor_free(r);
    tensor_free(lt);

    // sqrt
    float sv[] = {0.0f, 1.0f, 4.0f, 9.0f};
    Tensor* st = make_1d(sv, 4);
    r = tensor_sqrt_op(st);
    ASSERT_CLOSE("sqrt(4)", r->data[2], 2.0f, 1e-5f);
    ASSERT_CLOSE("sqrt(9)", r->data[3], 3.0f, 1e-5f);
    tensor_free(r); tensor_free(st);

    // abs
    r = tensor_abs_op(t);
    ASSERT_CLOSE("abs(-1)", r->data[3], 1.0f, 1e-5f);
    tensor_free(r);

    // clamp
    r = tensor_clamp(t, -0.5f, 0.5f);
    ASSERT_CLOSE("clamp(1, -0.5, 0.5)", r->data[2], 0.5f, 1e-5f);
    ASSERT_CLOSE("clamp(-1, -0.5, 0.5)", r->data[3], -0.5f, 1e-5f);
    tensor_free(r);

    // neg
    r = tensor_neg(t);
    ASSERT_CLOSE("neg(1)", r->data[2], -1.0f, 1e-5f);
    tensor_free(r);

    // sign
    r = tensor_sign(t);
    ASSERT_CLOSE("sign(0.5)", r->data[1], 1.0f, 1e-5f);
    ASSERT_CLOSE("sign(-1)", r->data[3], -1.0f, 1e-5f);
    ASSERT_CLOSE("sign(0)", r->data[0], 0.0f, 1e-5f);
    tensor_free(r);

    // sub
    float av[] = {3.0f, 5.0f, 7.0f, 9.0f};
    float bv[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor* a = make_1d(av, 4);
    Tensor* b = make_1d(bv, 4);
    r = tensor_sub(a, b);
    ASSERT_CLOSE("sub", r->data[0], 2.0f, 1e-5f);
    ASSERT_CLOSE("sub", r->data[3], 5.0f, 1e-5f);
    tensor_free(r);

    // div
    r = tensor_div(a, b);
    ASSERT_CLOSE("div(3/1)", r->data[0], 3.0f, 1e-5f);
    ASSERT_CLOSE("div(9/4)", r->data[3], 2.25f, 1e-5f);
    tensor_free(r);

    // lerp
    r = tensor_lerp(a, b, 0.5f);
    ASSERT_CLOSE("lerp(3,1,0.5)", r->data[0], 2.0f, 1e-5f);
    tensor_free(r);

    // erf
    r = tensor_erf(t);
    ASSERT_CLOSE("erf(0)", r->data[0], 0.0f, 1e-5f);
    tensor_free(r);

    // nan_to_num
    float nv[] = {1.0f, NAN, INFINITY, -INFINITY};
    Tensor* nt = make_1d(nv, 4);
    r = tensor_nan_to_num(nt, 0.0f, 999.0f, -999.0f);
    ASSERT_CLOSE("nan_to_num(1)", r->data[0], 1.0f, 1e-5f);
    ASSERT_CLOSE("nan_to_num(nan)", r->data[1], 0.0f, 1e-5f);
    ASSERT_CLOSE("nan_to_num(inf)", r->data[2], 999.0f, 1e-5f);
    ASSERT_CLOSE("nan_to_num(-inf)", r->data[3], -999.0f, 1e-5f);
    tensor_free(r); tensor_free(nt);

    // addcmul / addcdiv
    Tensor* one = make_1d((float[]){1,1,1,1}, 4);
    r = tensor_addcmul(one, a, b, 2.0f);
    ASSERT_CLOSE("addcmul", r->data[0], 1.0f + 2.0f * 3.0f * 1.0f, 1e-5f);
    tensor_free(r);
    r = tensor_addcdiv(one, a, b, 1.0f);
    ASSERT_CLOSE("addcdiv", r->data[0], 1.0f + 3.0f / 1.0f, 1e-5f);
    tensor_free(r); tensor_free(one);

    // inplace
    Tensor* ip = tensor_clone(t);
    tensor_sin_inplace(ip);
    ASSERT_CLOSE("sin_inplace(0)", ip->data[0], 0.0f, 1e-5f);
    tensor_free(ip);

    tensor_free(t); tensor_free(a); tensor_free(b);
}

// ===== Manipulation Tests =====
static void test_manipulation(void) {
    printf("--- Manipulation ---\n");

    // reshape
    float v[] = {1,2,3,4,5,6};
    Tensor* t = make_1d(v, 6);
    uint32_t ns[] = {2, 3};
    Tensor* r = tensor_reshape(t, 2, ns);
    ASSERT_TRUE("reshape dims", r->dims == 2);
    ASSERT_TRUE("reshape shape", r->shape[0] == 2 && r->shape[1] == 3);
    ASSERT_CLOSE("reshape data", r->data[5], 6.0f, 1e-5f);
    tensor_free(r);

    // squeeze / unsqueeze
    uint32_t ss[] = {1, 3, 1};
    Tensor* sq = tensor_create(3, ss, false);
    sq->data[0] = 1; sq->data[1] = 2; sq->data[2] = 3;
    r = tensor_squeeze(sq);
    ASSERT_TRUE("squeeze", r->dims == 1 && r->shape[0] == 3);
    tensor_free(r);
    r = tensor_unsqueeze(sq, 0);
    ASSERT_TRUE("unsqueeze", r->dims == 4 && r->shape[0] == 1);
    tensor_free(r); tensor_free(sq);

    // flatten
    float f2d[] = {1,2,3,4,5,6};
    Tensor* f = make_2d(f2d, 2, 3);
    r = tensor_flatten(f, 0, 1);
    ASSERT_TRUE("flatten", r->dims == 1 && r->size == 6);
    tensor_free(r); tensor_free(f);

    // t_op (2D transpose)
    float m[] = {1,2,3,4,5,6};
    Tensor* mt = make_2d(m, 2, 3);
    r = tensor_t_op(mt);
    ASSERT_TRUE("t_op shape", r->shape[0] == 3 && r->shape[1] == 2);
    ASSERT_CLOSE("t_op[0,0]", r->data[0], 1.0f, 1e-5f);
    ASSERT_CLOSE("t_op[0,1]", r->data[1], 4.0f, 1e-5f);
    ASSERT_CLOSE("t_op[1,0]", r->data[2], 2.0f, 1e-5f);
    tensor_free(r); tensor_free(mt);

    // cat
    Tensor* a = make_1d((float[]){1,2,3}, 3);
    Tensor* b = make_1d((float[]){4,5,6}, 3);
    const Tensor* ts[] = {a, b};
    r = tensor_cat(ts, 2, 0);
    ASSERT_TRUE("cat size", r->size == 6);
    ASSERT_CLOSE("cat[3]", r->data[3], 4.0f, 1e-5f);
    tensor_free(r);

    // stack
    r = tensor_stack(ts, 2, 0);
    ASSERT_TRUE("stack shape", r->dims == 2 && r->shape[0] == 2 && r->shape[1] == 3);
    tensor_free(r);

    // chunk
    Tensor* ch = make_1d((float[]){1,2,3,4,5,6}, 6);
    uint32_t nc;
    Tensor** chunks = tensor_chunk(ch, 3, 0, &nc);
    ASSERT_TRUE("chunk count", nc == 3);
    ASSERT_TRUE("chunk[0] size", chunks[0]->size == 2);
    ASSERT_CLOSE("chunk[0][0]", chunks[0]->data[0], 1.0f, 1e-5f);
    ASSERT_CLOSE("chunk[2][1]", chunks[2]->data[1], 6.0f, 1e-5f);
    for (uint32_t i = 0; i < nc; i++) tensor_free(chunks[i]);
    free(chunks); tensor_free(ch);

    // clone
    r = tensor_clone(a);
    ASSERT_CLOSE("clone", r->data[0], 1.0f, 1e-5f);
    tensor_free(r);

    // flip
    int32_t fd = 0;
    r = tensor_flip(a, &fd, 1);
    ASSERT_CLOSE("flip[0]", r->data[0], 3.0f, 1e-5f);
    ASSERT_CLOSE("flip[2]", r->data[2], 1.0f, 1e-5f);
    tensor_free(r);

    // roll
    r = tensor_roll(a, 1, 0);
    ASSERT_CLOSE("roll[0]", r->data[0], 3.0f, 1e-5f);
    ASSERT_CLOSE("roll[1]", r->data[1], 1.0f, 1e-5f);
    tensor_free(r);

    // narrow
    Tensor* nr2 = make_1d((float[]){10,20,30,40,50}, 5);
    r = tensor_narrow(nr2, 0, 1, 3);
    ASSERT_TRUE("narrow size", r->size == 3);
    ASSERT_CLOSE("narrow[0]", r->data[0], 20.0f, 1e-5f);
    ASSERT_CLOSE("narrow[2]", r->data[2], 40.0f, 1e-5f);
    tensor_free(r); tensor_free(nr2);

    // index_select
    uint32_t idx[] = {0, 2};
    r = tensor_index_select(a, 0, idx, 2);
    ASSERT_TRUE("index_select size", r->size == 2);
    ASSERT_CLOSE("index_select[0]", r->data[0], 1.0f, 1e-5f);
    ASSERT_CLOSE("index_select[1]", r->data[1], 3.0f, 1e-5f);
    tensor_free(r);

    // where
    Tensor* cond = make_1d((float[]){1,0,1}, 3);
    r = tensor_where_cond(cond, a, b);
    ASSERT_CLOSE("where[0]", r->data[0], 1.0f, 1e-5f);
    ASSERT_CLOSE("where[1]", r->data[1], 5.0f, 1e-5f);
    tensor_free(r); tensor_free(cond);

    // tile
    uint32_t reps[] = {3};
    r = tensor_tile(a, reps, 1);
    ASSERT_TRUE("tile size", r->size == 9);
    ASSERT_CLOSE("tile[3]", r->data[3], 1.0f, 1e-5f);
    tensor_free(r);

    tensor_free(a); tensor_free(b);
}

// ===== Reduction Tests =====
static void test_reduce(void) {
    printf("--- Reduce ---\n");
    float v[] = {1,2,3,4,5,6};
    Tensor* t = make_1d(v, 6);

    ASSERT_CLOSE("sum_all", tensor_sum_all(t), 21.0f, 1e-5f);
    ASSERT_CLOSE("prod_all", tensor_prod_all(t), 720.0f, 1e-5f);
    ASSERT_CLOSE("mean_all", tensor_mean_all(t), 3.5f, 1e-5f);
    ASSERT_CLOSE("max_all", tensor_max_all(t), 6.0f, 1e-5f);
    ASSERT_CLOSE("min_all", tensor_min_all(t), 1.0f, 1e-5f);
    ASSERT_TRUE("argmax_all", tensor_argmax_all(t) == 5);
    ASSERT_TRUE("argmin_all", tensor_argmin_all(t) == 0);
    ASSERT_TRUE("count_nonzero", tensor_count_nonzero_all(t) == 6);
    ASSERT_CLOSE("median", tensor_median_all(t), 3.5f, 1e-5f);

    // Norm
    float nv[] = {3.0f, 4.0f};
    Tensor* nt = make_1d(nv, 2);
    ASSERT_CLOSE("norm L2", tensor_norm_all(nt, 2.0f), 5.0f, 1e-5f);
    tensor_free(nt);

    // all/any
    ASSERT_TRUE("all(nonzero)", tensor_all(t));
    float zv[] = {0, 1, 0};
    Tensor* zt = make_1d(zv, 3);
    ASSERT_TRUE("any", tensor_any(zt));
    ASSERT_TRUE("!all(has_zero)", !tensor_all(zt));
    tensor_free(zt);

    // Axis reductions (2D)
    float m[] = {1,2,3, 4,5,6};
    Tensor* mt = make_2d(m, 2, 3);
    Tensor* r = tensor_sum(mt, 0);
    ASSERT_TRUE("sum dim0 shape", r->size == 3);
    ASSERT_CLOSE("sum dim0 [0]", r->data[0], 5.0f, 1e-5f);
    ASSERT_CLOSE("sum dim0 [2]", r->data[2], 9.0f, 1e-5f);
    tensor_free(r);

    r = tensor_sum(mt, 1);
    ASSERT_TRUE("sum dim1 shape", r->size == 2);
    ASSERT_CLOSE("sum dim1 [0]", r->data[0], 6.0f, 1e-5f);
    ASSERT_CLOSE("sum dim1 [1]", r->data[1], 15.0f, 1e-5f);
    tensor_free(r);

    r = tensor_mean(mt, 1);
    ASSERT_CLOSE("mean dim1 [0]", r->data[0], 2.0f, 1e-5f);
    tensor_free(r);

    r = tensor_max_dim(mt, 1);
    ASSERT_CLOSE("max dim1 [0]", r->data[0], 3.0f, 1e-5f);
    tensor_free(r);

    r = tensor_argmax_dim(mt, 1);
    ASSERT_CLOSE("argmax dim1 [0]", r->data[0], 2.0f, 1e-5f);
    tensor_free(r);

    // cumsum
    r = tensor_cumsum(t, 0);
    ASSERT_CLOSE("cumsum[0]", r->data[0], 1.0f, 1e-5f);
    ASSERT_CLOSE("cumsum[2]", r->data[2], 6.0f, 1e-5f);
    ASSERT_CLOSE("cumsum[5]", r->data[5], 21.0f, 1e-5f);
    tensor_free(r);

    // diff
    r = tensor_diff(t, 0);
    ASSERT_TRUE("diff size", r->size == 5);
    ASSERT_CLOSE("diff[0]", r->data[0], 1.0f, 1e-5f);
    tensor_free(r);

    // nan-safe
    float nanv[] = {1, NAN, 3, NAN, 5};
    Tensor* nant = make_1d(nanv, 5);
    ASSERT_CLOSE("nansum", tensor_nansum_all(nant), 9.0f, 1e-5f);
    ASSERT_CLOSE("nanmean", tensor_nanmean_all(nant), 3.0f, 1e-5f);
    ASSERT_CLOSE("nanmax", tensor_nanmax_all(nant), 5.0f, 1e-5f);
    ASSERT_CLOSE("nanmin", tensor_nanmin_all(nant), 1.0f, 1e-5f);
    tensor_free(nant);

    // dist
    Tensor* da = make_1d((float[]){0,0}, 2);
    Tensor* db = make_1d((float[]){3,4}, 2);
    ASSERT_CLOSE("dist L2", tensor_dist(da, db, 2.0f), 5.0f, 1e-4f);
    tensor_free(da); tensor_free(db);

    tensor_free(t); tensor_free(mt);
}

// ===== Compare Tests =====
static void test_compare(void) {
    printf("--- Compare ---\n");
    Tensor* a = make_1d((float[]){1,2,3,4}, 4);
    Tensor* b = make_1d((float[]){2,2,2,2}, 4);

    Tensor* r = tensor_eq(a, b);
    ASSERT_CLOSE("eq[0]", r->data[0], 0.0f, 1e-5f);
    ASSERT_CLOSE("eq[1]", r->data[1], 1.0f, 1e-5f);
    tensor_free(r);

    r = tensor_lt(a, b);
    ASSERT_CLOSE("lt[0]", r->data[0], 1.0f, 1e-5f);
    ASSERT_CLOSE("lt[2]", r->data[2], 0.0f, 1e-5f);
    tensor_free(r);

    r = tensor_gt(a, b);
    ASSERT_CLOSE("gt[2]", r->data[2], 1.0f, 1e-5f);
    tensor_free(r);

    ASSERT_TRUE("equal", !tensor_equal(a, b));
    ASSERT_TRUE("equal self", tensor_equal(a, a));

    r = tensor_maximum(a, b);
    ASSERT_CLOSE("maximum[0]", r->data[0], 2.0f, 1e-5f);
    ASSERT_CLOSE("maximum[2]", r->data[2], 3.0f, 1e-5f);
    tensor_free(r);

    r = tensor_minimum(a, b);
    ASSERT_CLOSE("minimum[0]", r->data[0], 1.0f, 1e-5f);
    ASSERT_CLOSE("minimum[2]", r->data[2], 2.0f, 1e-5f);
    tensor_free(r);

    // logical
    Tensor* la = make_1d((float[]){1,0,1,0}, 4);
    Tensor* lb = make_1d((float[]){1,1,0,0}, 4);
    r = tensor_logical_and(la, lb);
    ASSERT_CLOSE("and[0]", r->data[0], 1.0f, 1e-5f);
    ASSERT_CLOSE("and[1]", r->data[1], 0.0f, 1e-5f);
    tensor_free(r);
    r = tensor_logical_or(la, lb);
    ASSERT_CLOSE("or[0]", r->data[0], 1.0f, 1e-5f);
    ASSERT_CLOSE("or[3]", r->data[3], 0.0f, 1e-5f);
    tensor_free(r);
    r = tensor_logical_not(la);
    ASSERT_CLOSE("not[0]", r->data[0], 0.0f, 1e-5f);
    ASSERT_CLOSE("not[1]", r->data[1], 1.0f, 1e-5f);
    tensor_free(r);
    tensor_free(la); tensor_free(lb);

    // isnan/isinf
    Tensor* ni = make_1d((float[]){1, NAN, INFINITY, -INFINITY}, 4);
    r = tensor_isnan_op(ni);
    ASSERT_CLOSE("isnan(1)", r->data[0], 0.0f, 1e-5f);
    ASSERT_CLOSE("isnan(nan)", r->data[1], 1.0f, 1e-5f);
    tensor_free(r);
    r = tensor_isinf_op(ni);
    ASSERT_CLOSE("isinf(inf)", r->data[2], 1.0f, 1e-5f);
    tensor_free(r);
    r = tensor_isfinite_op(ni);
    ASSERT_CLOSE("isfinite(1)", r->data[0], 1.0f, 1e-5f);
    ASSERT_CLOSE("isfinite(nan)", r->data[1], 0.0f, 1e-5f);
    tensor_free(r); tensor_free(ni);

    // sort
    Tensor* us = make_1d((float[]){3,1,4,1,5,9}, 6);
    Tensor* idx;
    r = tensor_sort_op(us, 0, false, &idx);
    ASSERT_CLOSE("sort[0]", r->data[0], 1.0f, 1e-5f);
    ASSERT_CLOSE("sort[5]", r->data[5], 9.0f, 1e-5f);
    tensor_free(r); tensor_free(idx); tensor_free(us);

    // allclose
    Tensor* c = make_1d((float[]){1.0001f, 2.0001f, 3.0001f, 4.0001f}, 4);
    ASSERT_TRUE("allclose", tensor_allclose(a, c, 1e-3f, 1e-3f));
    tensor_free(c);

    // unique
    Tensor* uq = make_1d((float[]){3,1,2,1,3,2}, 6);
    uint32_t uc;
    r = tensor_unique(uq, &uc);
    ASSERT_TRUE("unique count", uc == 3);
    tensor_free(r); tensor_free(uq);

    tensor_free(a); tensor_free(b);
}

// ===== Linear Algebra Tests =====
static void test_linalg(void) {
    printf("--- Linalg ---\n"); fflush(stdout);

    // dot
    Tensor* a = make_1d((float[]){1,2,3}, 3);
    Tensor* b = make_1d((float[]){4,5,6}, 3);
    ASSERT_CLOSE("dot", tensor_dot(a, b), 32.0f, 1e-5f);
    printf("  dot OK\n"); fflush(stdout);

    // outer
    Tensor* r = tensor_outer(a, b);
    ASSERT_TRUE("outer shape", r->shape[0] == 3 && r->shape[1] == 3);
    ASSERT_CLOSE("outer[0,0]", r->data[0], 4.0f, 1e-5f);
    ASSERT_CLOSE("outer[0,2]", r->data[2], 6.0f, 1e-5f);
    tensor_free(r);
    printf("  outer OK\n"); fflush(stdout);

    // cross
    Tensor* cx = make_1d((float[]){1,0,0}, 3);
    Tensor* cy = make_1d((float[]){0,1,0}, 3);
    r = tensor_cross(cx, cy);
    ASSERT_CLOSE("cross z", r->data[2], 1.0f, 1e-5f);
    tensor_free(r); tensor_free(cx); tensor_free(cy);
    printf("  cross OK\n"); fflush(stdout);

    // mv
    float mv[] = {1,2,3,4,5,6};
    Tensor* mat = make_2d(mv, 2, 3);
    Tensor* vec = make_1d((float[]){1,1,1}, 3);
    r = tensor_mv(mat, vec);
    ASSERT_CLOSE("mv[0]", r->data[0], 6.0f, 1e-5f);
    ASSERT_CLOSE("mv[1]", r->data[1], 15.0f, 1e-5f);
    tensor_free(r); tensor_free(vec);
    printf("  mv OK\n"); fflush(stdout);

    // eye
    r = tensor_eye(3);
    ASSERT_CLOSE("eye[0,0]", r->data[0], 1.0f, 1e-5f);
    ASSERT_CLOSE("eye[0,1]", r->data[1], 0.0f, 1e-5f);
    ASSERT_CLOSE("eye[1,1]", r->data[4], 1.0f, 1e-5f);
    tensor_free(r);
    printf("  eye OK\n"); fflush(stdout);

    // trace
    float trv[] = {1,2,3,4};
    Tensor* tr = make_2d(trv, 2, 2);
    ASSERT_CLOSE("trace", tensor_trace(tr), 5.0f, 1e-5f);
    printf("  trace OK\n"); fflush(stdout);

    // det (2x2)
    ASSERT_CLOSE("det 2x2", tensor_det(tr), -2.0f, 1e-5f);
    printf("  det OK\n"); fflush(stdout);

    // inverse (2x2)
    r = tensor_inverse(tr);
    // For [[1,2],[3,4]], inv = [[-2,1],[1.5,-0.5]]
    ASSERT_CLOSE("inv[0,0]", r->data[0], -2.0f, 1e-4f);
    ASSERT_CLOSE("inv[0,1]", r->data[1], 1.0f, 1e-4f);
    ASSERT_CLOSE("inv[1,0]", r->data[2], 1.5f, 1e-4f);
    ASSERT_CLOSE("inv[1,1]", r->data[3], -0.5f, 1e-4f);
    tensor_free(r);
    printf("  inverse OK\n"); fflush(stdout);

    // tril/triu
    r = tensor_tril(tr, 0);
    ASSERT_CLOSE("tril[0,1]", r->data[1], 0.0f, 1e-5f);
    ASSERT_CLOSE("tril[1,0]", r->data[2], 3.0f, 1e-5f);
    tensor_free(r);
    r = tensor_triu(tr, 0);
    ASSERT_CLOSE("triu[1,0]", r->data[2], 0.0f, 1e-5f);
    ASSERT_CLOSE("triu[0,1]", r->data[1], 2.0f, 1e-5f);
    tensor_free(r);
    printf("  tril/triu OK\n"); fflush(stdout);

    // diag (1D -> 2D)
    Tensor* dv = make_1d((float[]){1,2,3}, 3);
    r = tensor_diag(dv, 0);
    ASSERT_TRUE("diag shape", r->shape[0] == 3 && r->shape[1] == 3);
    ASSERT_CLOSE("diag[0,0]", r->data[0], 1.0f, 1e-5f);
    ASSERT_CLOSE("diag[1,1]", r->data[4], 2.0f, 1e-5f);
    ASSERT_CLOSE("diag[0,1]", r->data[1], 0.0f, 1e-5f);
    tensor_free(r); tensor_free(dv);
    printf("  diag OK\n"); fflush(stdout);

    // cholesky (SPD matrix)
    float spd[] = {4,2, 2,3};
    Tensor* spdm = make_2d(spd, 2, 2);
    r = tensor_cholesky(spdm);
    // L = [[2,0],[1,sqrt(2)]]
    ASSERT_CLOSE("chol[0,0]", r->data[0], 2.0f, 1e-4f);
    ASSERT_CLOSE("chol[1,0]", r->data[2], 1.0f, 1e-4f);
    ASSERT_CLOSE("chol[1,1]", r->data[3], sqrtf(2.0f), 1e-4f);
    tensor_free(r); tensor_free(spdm);
    printf("  cholesky OK\n"); fflush(stdout);

    // matrix_power
    r = tensor_matrix_power(tr, 2);
    // [[1,2],[3,4]]^2 = [[7,10],[15,22]]
    ASSERT_CLOSE("matpow[0,0]", r->data[0], 7.0f, 1e-4f);
    ASSERT_CLOSE("matpow[1,1]", r->data[3], 22.0f, 1e-4f);
    tensor_free(r);
    printf("  matrix_power OK\n"); fflush(stdout);

    // bmm
    float bma[] = {1,2,3,4, 5,6,7,8};  // (2,2,2)
    float bmb[] = {1,0,0,1, 2,0,0,2};
    uint32_t bs[3] = {2, 2, 2};
    Tensor* ba = tensor_create(3, bs, false); memcpy(ba->data, bma, 8*sizeof(float));
    Tensor* bb = tensor_create(3, bs, false); memcpy(bb->data, bmb, 8*sizeof(float));
    r = tensor_bmm(ba, bb);
    ASSERT_CLOSE("bmm[0,0,0]", r->data[0], 1.0f, 1e-5f);
    ASSERT_CLOSE("bmm[1,0,0]", r->data[4], 10.0f, 1e-5f);
    tensor_free(r); tensor_free(ba); tensor_free(bb);
    printf("  bmm OK\n"); fflush(stdout);

    tensor_free(tr); tensor_free(mat); tensor_free(a); tensor_free(b);
}


// ===== FFT Tests =====
static void test_fft(void) {
    printf("--- FFT ---\n");
    // FFT of [1,0,0,0] should give [1,1,1,1]
    Tensor* t = make_1d((float[]){1,0,0,0}, 4);
    Tensor* r = tensor_fft(t);
    ASSERT_TRUE("fft shape", r->shape[0] == 4 && r->shape[1] == 2);
    ASSERT_CLOSE("fft[0] re", r->data[0], 1.0f, 1e-5f);
    ASSERT_CLOSE("fft[0] im", r->data[1], 0.0f, 1e-5f);
    ASSERT_CLOSE("fft[1] re", r->data[2], 1.0f, 1e-5f);
    ASSERT_CLOSE("fft[2] re", r->data[4], 1.0f, 1e-5f);

    // IFFT should recover original
    Tensor* ir = tensor_ifft(r);
    ASSERT_CLOSE("ifft[0] re", ir->data[0], 1.0f, 1e-5f);
    ASSERT_CLOSE("ifft[1] re", ir->data[2], 0.0f, 1e-5f);
    tensor_free(ir); tensor_free(r); tensor_free(t);
}

// ===== Random Tests =====
static void test_random(void) {
    printf("--- Random ---\n");
    rpl_manual_seed(42);

    uint32_t s[] = {100};
    Tensor* r = tensor_rand(1, s);
    ASSERT_TRUE("rand size", r->size == 100);
    // All values in [0,1)
    bool in_range = true;
    for (uint32_t i = 0; i < r->size; i++)
        if (r->data[i] < 0 || r->data[i] >= 1.0f) { in_range = false; break; }
    ASSERT_TRUE("rand range", in_range);
    tensor_free(r);

    r = tensor_randn(1, s);
    ASSERT_TRUE("randn size", r->size == 100);
    tensor_free(r);

    // zeros/ones
    Tensor* z = tensor_zeros(1, s);
    ASSERT_CLOSE("zeros", z->data[0], 0.0f, 1e-5f);
    tensor_free(z);

    Tensor* o = tensor_ones(1, s);
    ASSERT_CLOSE("ones", o->data[0], 1.0f, 1e-5f);
    tensor_free(o);

    // arange
    r = tensor_arange(0, 5, 1);
    ASSERT_TRUE("arange size", r->size == 5);
    ASSERT_CLOSE("arange[2]", r->data[2], 2.0f, 1e-5f);
    tensor_free(r);

    // linspace
    r = tensor_linspace(0, 1, 5);
    ASSERT_CLOSE("linspace[0]", r->data[0], 0.0f, 1e-5f);
    ASSERT_CLOSE("linspace[4]", r->data[4], 1.0f, 1e-5f);
    ASSERT_CLOSE("linspace[2]", r->data[2], 0.5f, 1e-5f);
    tensor_free(r);

    // randperm
    r = tensor_randperm(10);
    ASSERT_TRUE("randperm size", r->size == 10);
    // Check all values present
    float sum = 0;
    for (uint32_t i = 0; i < 10; i++) sum += r->data[i];
    ASSERT_CLOSE("randperm sum", sum, 45.0f, 1e-5f);
    tensor_free(r);

    // zeros_like / ones_like
    Tensor* ref = make_1d((float[]){1,2,3}, 3);
    z = tensor_zeros_like(ref);
    ASSERT_TRUE("zeros_like shape", z->size == 3 && z->data[0] == 0.0f);
    tensor_free(z);
    o = tensor_ones_like(ref);
    ASSERT_TRUE("ones_like", o->data[0] == 1.0f);
    tensor_free(o); tensor_free(ref);
}

// ===== Utility Tests =====
static void test_util(void) {
    printf("--- Util ---\n");
    Tensor* t = make_1d((float[]){1,2,3,4}, 4);
    ASSERT_TRUE("numel", tensor_numel(t) == 4);
    ASSERT_TRUE("is_floating_point", tensor_is_floating_point(t));

    // Window functions
    Tensor* w = tensor_hann_window(5);
    ASSERT_TRUE("hann size", w->size == 5);
    ASSERT_CLOSE("hann[0]", w->data[0], 0.0f, 1e-5f);
    ASSERT_CLOSE("hann[2]", w->data[2], 1.0f, 1e-5f);
    tensor_free(w);

    w = tensor_hamming_window(5);
    ASSERT_TRUE("hamming size", w->size == 5);
    tensor_free(w);

    // bincount
    Tensor* bc = make_1d((float[]){0,1,1,2,2,2}, 6);
    Tensor* r = tensor_bincount(bc, 0);
    ASSERT_CLOSE("bincount[0]", r->data[0], 1.0f, 1e-5f);
    ASSERT_CLOSE("bincount[1]", r->data[1], 2.0f, 1e-5f);
    ASSERT_CLOSE("bincount[2]", r->data[2], 3.0f, 1e-5f);
    tensor_free(r); tensor_free(bc);

    // histogram
    Tensor* hd = make_1d((float[]){0.1f, 0.5f, 0.9f, 1.5f, 2.5f}, 5);
    r = tensor_histc(hd, 3, 0.0f, 3.0f);
    ASSERT_TRUE("histc size", r->size == 3);
    ASSERT_CLOSE("histc[0]", r->data[0], 3.0f, 1e-5f);  // [0,1)
    ASSERT_CLOSE("histc[1]", r->data[1], 1.0f, 1e-5f);  // [1,2)
    ASSERT_CLOSE("histc[2]", r->data[2], 1.0f, 1e-5f);  // [2,3)
    tensor_free(r); tensor_free(hd);

    // broadcast
    uint32_t bs[] = {4, 4};
    r = tensor_broadcast_to(t, 2, bs);
    ASSERT_TRUE("broadcast shape", r->shape[0] == 4 && r->shape[1] == 4);
    ASSERT_CLOSE("broadcast[1,2]", r->data[1*4+2], 3.0f, 1e-5f);
    tensor_free(r);

    // convolve
    Tensor* cv_a = make_1d((float[]){1,2,3}, 3);
    Tensor* cv_b = make_1d((float[]){0,1,0.5f}, 3);
    r = tensor_convolve(cv_a, cv_b);
    ASSERT_TRUE("convolve size", r->size == 5);
    ASSERT_CLOSE("convolve[2]", r->data[2], 2.5f, 1e-5f);
    tensor_free(r); tensor_free(cv_a); tensor_free(cv_b);

    // interp
    Tensor* xp = make_1d((float[]){0,1,2,3}, 4);
    Tensor* fp = make_1d((float[]){0,2,4,6}, 4);
    Tensor* x = make_1d((float[]){0.5f, 1.5f, 2.5f}, 3);
    r = tensor_interp(x, xp, fp);
    ASSERT_CLOSE("interp[0]", r->data[0], 1.0f, 1e-5f);
    ASSERT_CLOSE("interp[1]", r->data[1], 3.0f, 1e-5f);
    tensor_free(r); tensor_free(xp); tensor_free(fp); tensor_free(x);

    // trapezoid
    Tensor* ty = make_1d((float[]){0, 1, 2, 3}, 4);
    ASSERT_CLOSE("trapezoid", tensor_trapezoid(ty, 1.0f), 4.5f, 1e-5f);
    tensor_free(ty);

    tensor_free(t);
}

// ============================================================
// Activation Tests
// ============================================================

static Tensor* act_like(const Tensor* t) { return tensor_create(t->dims, t->shape, false); }

static void test_activations(void) {
    printf("--- Activations ---\n");

    // Test input: [-2, -1, 0, 1, 2]
    Tensor* t = make_1d((float[]){-2, -1, 0, 1, 2}, 5);
    Tensor* out = act_like(t);

    // GELU: GELU(-2)≈-0.0454, GELU(0)=0, GELU(1)≈0.8413, GELU(2)≈1.9545
    tensor_gelu(out, t);
    ASSERT_CLOSE("gelu[2]", out->data[2], 0.0f, 1e-3f);       // GELU(0) = 0
    ASSERT_CLOSE("gelu[3]", out->data[3], 0.8412f, 0.02f);     // GELU(1)
    ASSERT_CLOSE("gelu[4]", out->data[4], 1.9545f, 0.02f);     // GELU(2)

    // SELU: SELU(1) = 1.0507*1 = 1.0507, SELU(-1) = 1.0507*1.6733*(e^-1 - 1) ≈ -1.1113
    tensor_selu(out, t);
    ASSERT_CLOSE("selu[3]", out->data[3], 1.0507f, 0.02f);
    ASSERT_CLOSE("selu[1]", out->data[1], -1.1113f, 0.05f);

    // Mish: Mish(0) = 0, Mish(1) = 1*tanh(ln(1+e^1)) ≈ 0.8651
    tensor_mish(out, t);
    ASSERT_CLOSE("mish[2]", out->data[2], 0.0f, 1e-3f);
    ASSERT_CLOSE("mish[3]", out->data[3], 0.8651f, 0.02f);

    // Hardswish: x*clip(x+3,0,6)/6
    // HS(-2) = -2*max(min(-2+3,6),0)/6 = -2*1/6 = -0.3333
    // HS(0) = 0*3/6 = 0
    // HS(2) = 2*5/6 = 1.6667
    tensor_hardswish(out, t);
    ASSERT_CLOSE("hardswish[0]", out->data[0], -0.3333f, 1e-3f);
    ASSERT_CLOSE("hardswish[2]", out->data[2], 0.0f, 1e-5f);
    ASSERT_CLOSE("hardswish[4]", out->data[4], 1.6667f, 1e-3f);

    // Hardsigmoid: clip(x/6+0.5, 0, 1)
    // HS(-2) = clip(-2/6+0.5, 0, 1) = clip(0.1667, 0, 1) = 0.1667
    // HS(0) = 0.5
    // HS(2) = clip(0.8333, 0, 1) = 0.8333
    tensor_hardsigmoid(out, t);
    ASSERT_CLOSE("hardsigmoid[0]", out->data[0], 0.1667f, 1e-3f);
    ASSERT_CLOSE("hardsigmoid[2]", out->data[2], 0.5f, 1e-5f);
    ASSERT_CLOSE("hardsigmoid[4]", out->data[4], 0.8333f, 1e-3f);

    // Hardtanh: clamp to [-1, 1]
    tensor_hardtanh(out, t, -1.0f, 1.0f);
    ASSERT_CLOSE("hardtanh[0]", out->data[0], -1.0f, 1e-5f);  // -2 clamped to -1
    ASSERT_CLOSE("hardtanh[2]", out->data[2], 0.0f, 1e-5f);   // 0 unchanged
    ASSERT_CLOSE("hardtanh[4]", out->data[4], 1.0f, 1e-5f);   // 2 clamped to 1

    // CELU: max(0,x) + min(0, alpha*(exp(x/alpha)-1)), alpha=1
    // CELU(-1, alpha=1) = 0 + min(0, 1*(e^-1 - 1)) = e^-1 - 1 ≈ -0.6321
    // CELU(1, alpha=1) = 1
    tensor_celu(out, t, 1.0f);
    ASSERT_CLOSE("celu[1]", out->data[1], -0.6321f, 0.02f);
    ASSERT_CLOSE("celu[3]", out->data[3], 1.0f, 1e-3f);

    // Softsign: x/(1+|x|)
    // SS(-1) = -1/2 = -0.5, SS(0) = 0, SS(1) = 0.5
    tensor_softsign(out, t);
    ASSERT_CLOSE("softsign[1]", out->data[1], -0.5f, 0.01f);
    ASSERT_CLOSE("softsign[2]", out->data[2], 0.0f, 1e-3f);
    ASSERT_CLOSE("softsign[3]", out->data[3], 0.5f, 0.01f);

    // LogSoftmax: x - log(sum(exp(x)))
    Tensor* lsm_in = make_1d((float[]){1, 2, 3}, 3);
    Tensor* lsm_out = act_like(lsm_in);
    tensor_log_softmax(lsm_out, lsm_in);
    // log(e^1 + e^2 + e^3) = 3 + log(e^-2 + e^-1 + 1) ≈ 3.4076
    // lsm[2] = 3 - 3.4076 = -0.4076
    ASSERT_CLOSE("logsoftmax[2]", lsm_out->data[2], -0.4076f, 0.01f);
    // All should sum to approximately log(1).. actually sum(exp(lsm)) = 1
    float lsm_sum = expf(lsm_out->data[0]) + expf(lsm_out->data[1]) + expf(lsm_out->data[2]);
    ASSERT_CLOSE("logsoftmax sum(exp)", lsm_sum, 1.0f, 0.01f);
    tensor_free(lsm_in); tensor_free(lsm_out);

    // RReLU: at eval, slope = (lower+upper)/2
    // RReLU(-1, 0.1, 0.3) = -1 * 0.2 = -0.2
    // RReLU(1, ...) = 1
    tensor_rrelu(out, t, 0.1f, 0.3f);
    ASSERT_CLOSE("rrelu[1]", out->data[1], -0.2f, 1e-3f);
    ASSERT_CLOSE("rrelu[3]", out->data[3], 1.0f, 1e-5f);

    // Threshold: x if x > thresh else value
    tensor_threshold(out, t, 0.0f, -99.0f);
    ASSERT_CLOSE("threshold[0]", out->data[0], -99.0f, 1e-5f);  // -2 <= 0
    ASSERT_CLOSE("threshold[2]", out->data[2], -99.0f, 1e-5f);  // 0 <= 0
    ASSERT_CLOSE("threshold[3]", out->data[3], 1.0f, 1e-5f);    // 1 > 0

    // PReLU with scalar weight
    Tensor* w = make_1d((float[]){0.25f}, 1);
    tensor_prelu(out, t, w);
    ASSERT_CLOSE("prelu[0]", out->data[0], -0.5f, 1e-3f);     // -2 * 0.25
    ASSERT_CLOSE("prelu[3]", out->data[3], 1.0f, 1e-5f);
    tensor_free(w);

    tensor_free(t);
    tensor_free(out);
}

int main(void) {
    printf("=== RPL New API Tests ===\n\n");
    test_math();
    test_manipulation();
    test_reduce();
    test_compare();
    test_linalg();
    test_fft();
    test_random();
    test_util();
    test_activations();

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}

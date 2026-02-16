#!/usr/bin/env python3
"""Comprehensive Python API test for RPL library.
Matches C test_new_api.c coverage: 221+ tests across 9 categories.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))

import rpl
import numpy as np
import math

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name} {detail}")

def approx(a, b, tol=0.05):
    return abs(a - b) < tol

# ============================================================
# 1. Tensor Creation & Properties
# ============================================================
print("--- Tensor Creation ---")
t = rpl.Tensor([[1, 2, 3], [4, 5, 6]])
check("shape", t.shape == (2, 3))
check("ndim", t.ndim == 2)
check("size", t.size == 6)
check("data", np.allclose(t.data, [[1, 2, 3], [4, 5, 6]]))

t1d = rpl.Tensor([1, 2, 3, 4, 5])
check("1d shape", t1d.shape == (5,))
check("len", len(t1d) == 5)

# ============================================================
# 2. Arithmetic
# ============================================================
print("--- Arithmetic ---")
a = rpl.Tensor([1, 2, 3])
b = rpl.Tensor([4, 5, 6])

c = a + b
check("add", np.allclose(c.data, [5, 7, 9]))

c = a - b
check("sub", np.allclose(c.data, [-3, -3, -3]))

c = a * b
check("mul", np.allclose(c.data, [4, 10, 18]))

c = -a
check("neg op", np.allclose(c.data, [-1, -2, -3]))

c = abs(rpl.Tensor([-1, 2, -3]))
check("abs op", np.allclose(c.data, [1, 2, 3]))

# Matmul
A = rpl.Tensor([[1, 2], [3, 4]])
B = rpl.Tensor([[5, 6], [7, 8]])
C = A @ B
check("matmul", np.allclose(C.data, [[19, 22], [43, 50]]))

# Scalar mul
c = a * 2
check("scalar mul", np.allclose(c.data, [2, 4, 6]))

# Div
c = rpl.Tensor([3, 5, 7, 9]) / rpl.Tensor([1, 2, 3, 4])
check("div(3/1)", approx(c.data[0], 3.0, 0.01))
check("div(9/4)", approx(c.data[3], 2.25, 0.01))

# Lerp
la = rpl.Tensor([3, 5, 7, 9])
lb = rpl.Tensor([1, 2, 3, 4])
c = la.lerp(lb, 0.5)
check("lerp(3,1,0.5)", approx(c.data[0], 2.0, 0.01))

# ============================================================
# 3. Activations (Tensor methods)
# ============================================================
print("--- Activations (Tensor methods) ---")
x = rpl.Tensor([-2, -1, 0, 1, 2])

c = x.relu()
check("relu", np.allclose(c.data, [0, 0, 0, 1, 2]))

c = x.leaky_relu(0.1)
check("leaky_relu", approx(c.data[0], -0.2) and approx(c.data[3], 1.0))

c = x.elu(1.0)
check("elu", approx(c.data[3], 1.0) and c.data[1] < 0)

c = x.selu()
check("selu", approx(c.data[3], 1.0507, 0.05))

c = x.gelu()
check("gelu", approx(c.data[2], 0.0) and approx(c.data[4], 1.95, 0.1))

c = x.mish()
check("mish", approx(c.data[2], 0.0) and approx(c.data[3], 0.865, 0.05))

c = x.swish()
check("swish", approx(c.data[2], 0.0))

c = x.hardswish()
check("hardswish", approx(c.data[2], 0.0) and approx(c.data[4], 1.6667, 0.01))

c = x.hardsigmoid()
check("hardsigmoid", approx(c.data[2], 0.5, 0.01))

c = x.hardtanh(-1, 1)
check("hardtanh", np.allclose(c.data, [-1, -1, 0, 1, 1]))

c = x.celu(1.0)
check("celu", approx(c.data[3], 1.0) and c.data[1] < 0)

c = x.softsign()
check("softsign", approx(c.data[1], -0.5, 0.02) and approx(c.data[3], 0.5, 0.02))

c = x.softplus()
check("softplus", c.data[3] > 0 and c.data[0] > 0)

c = x.rrelu(0.1, 0.3)
check("rrelu", approx(c.data[1], -0.2, 0.01) and approx(c.data[3], 1.0))

c = x.threshold(0.0, -99.0)
check("threshold", approx(c.data[0], -99.0) and approx(c.data[3], 1.0))

# Log softmax
lsm = rpl.Tensor([1, 2, 3])
c = lsm.log_softmax()
check("log_softmax sum(exp)=1", approx(sum(np.exp(c.data)), 1.0, 0.02))

# Softmax
sm = rpl.Tensor([1, 2, 3])
c = sm.softmax()
check("softmax sum=1", approx(sum(c.data), 1.0, 0.01))

# Sigmoid
sig = rpl.Tensor([0])
c = sig.sigmoid()
check("sigmoid(0)=0.5", approx(c.data[0], 0.5, 0.01))

# Tanh
th = rpl.Tensor([0])
c = th.tanh()
check("tanh(0)=0", approx(c.data[0], 0.0, 0.01))

# ============================================================
# 4. Activation Modules (nn.*)
# ============================================================
print("--- Activation Modules (nn.*) ---")
x = rpl.Tensor([-2, -1, 0, 1, 2])

for mod_cls, name in [
    (rpl.nn.ReLU, "ReLU"),
    (rpl.nn.Sigmoid, "Sigmoid"),
    (rpl.nn.Tanh, "Tanh"),
    (rpl.nn.GELU, "GELU"),
    (rpl.nn.SELU, "SELU"),
    (rpl.nn.Swish, "Swish"),
    (rpl.nn.Mish, "Mish"),
    (rpl.nn.Hardswish, "Hardswish"),
    (rpl.nn.Hardsigmoid, "Hardsigmoid"),
    (rpl.nn.Softsign, "Softsign"),
    (rpl.nn.Softmax, "Softmax"),
    (rpl.nn.LogSoftmax, "LogSoftmax"),
]:
    mod = mod_cls()
    out = mod(x)
    check(f"nn.{name} runs", out is not None and out.size == 5)

# Parametric modules
for mod_cls, args, name in [
    (rpl.nn.LeakyReLU, (0.01,), "LeakyReLU"),
    (rpl.nn.ELU, (1.0,), "ELU"),
    (rpl.nn.CELU, (1.0,), "CELU"),
    (rpl.nn.Softplus, (1.0, 20.0), "Softplus"),
    (rpl.nn.Hardtanh, (-1.0, 1.0), "Hardtanh"),
    (rpl.nn.RReLU, (0.125, 0.333), "RReLU"),
    (rpl.nn.Threshold, (0.0, -1.0), "Threshold"),
]:
    mod = mod_cls(*args)
    out = mod(x)
    check(f"nn.{name} runs", out is not None and out.size == 5)

# PReLU
prelu = rpl.nn.PReLU(1, 0.25)
out = prelu(x)
check("nn.PReLU runs", out is not None and out.size == 5)
check("nn.PReLU neg", approx(out.data[0], -0.5, 0.02))

# ============================================================
# 5. Math Operations
# ============================================================
print("--- Math Ops ---")
t = rpl.Tensor([1, 4, 9])

c = t.sqrt()
check("sqrt", np.allclose(c.data, [1, 2, 3]))

c = t.square()
check("square", np.allclose(c.data, [1, 16, 81]))

c = rpl.Tensor([1, 2, 3]).exp()
check("exp", approx(c.data[0], np.e, 0.01))

c = rpl.Tensor([1, np.e, np.e**2]).log()
check("log", approx(c.data[0], 0.0, 0.01) and approx(c.data[1], 1.0, 0.01))

c = rpl.Tensor([4, 9]).reciprocal()
check("reciprocal", approx(c.data[0], 0.25, 0.01))

c = rpl.Tensor([4, 9]).rsqrt()
check("rsqrt", approx(c.data[0], 0.5, 0.01))

c = rpl.Tensor([-1, 0, 1]).sign()
check("sign", np.allclose(c.data, [-1, 0, 1]))

c = rpl.Tensor([1.7, -2.3]).floor()
check("floor", np.allclose(c.data, [1, -3]))

c = rpl.Tensor([1.2, -2.7]).ceil()
check("ceil", np.allclose(c.data, [2, -2]))

c = rpl.Tensor([-5, 0, 5]).clamp(-2, 3)
check("clamp", np.allclose(c.data, [-2, 0, 3]))

# sin
c = rpl.Tensor([0, 0.5, 1.0]).sin()
check("sin(0)", approx(c.data[0], 0.0, 1e-5))
check("sin(0.5)", approx(c.data[1], math.sin(0.5), 1e-4))

# cos
c = rpl.Tensor([0, 0.5]).cos()
check("cos(0)", approx(c.data[0], 1.0, 1e-5))

# tan
c = rpl.Tensor([0, 0.5]).tan()
check("tan(0)", approx(c.data[0], 0.0, 1e-5))
check("tan(0.5)", approx(c.data[1], math.tan(0.5), 1e-4))

# asin/acos/atan
c = rpl.Tensor([0, 0.5]).asin()
check("asin(0)", approx(c.data[0], 0.0, 1e-5))

c = rpl.Tensor([1, 0.5]).acos()
check("acos(1)", approx(c.data[0], 0.0, 1e-5))

c = rpl.Tensor([0, 1]).atan()
check("atan(0)", approx(c.data[0], 0.0, 1e-5))

# sinh/cosh
c = rpl.Tensor([0, 1]).sinh()
check("sinh(0)", approx(c.data[0], 0.0, 1e-5))

c = rpl.Tensor([0, 1]).cosh()
check("cosh(0)", approx(c.data[0], 1.0, 1e-5))

# exp2/expm1
c = rpl.Tensor([0, 1, 2]).exp2()
check("exp2(2)", approx(c.data[2], 4.0, 1e-4))

c = rpl.Tensor([0, 1]).expm1()
check("expm1(0)", approx(c.data[0], 0.0, 1e-5))

# log2/log10/log1p
c = rpl.Tensor([1, 2, 4]).log2()
check("log2(4)", approx(c.data[2], 2.0, 1e-4))

c = rpl.Tensor([1, 10, 100]).log10()
check("log10(100)", approx(c.data[2], 2.0, 1e-4))

c = rpl.Tensor([0, 1]).log1p()
check("log1p(0)", approx(c.data[0], 0.0, 1e-5))

# frac/cbrt
c = rpl.Tensor([1.7, -2.3]).frac()
check("frac(1.7)", approx(c.data[0], 0.7, 1e-4))

c = rpl.Tensor([8, 27]).cbrt()
check("cbrt(8)", approx(c.data[0], 2.0, 1e-4))

# erf/erfc
c = rpl.Tensor([0]).erf()
check("erf(0)", approx(c.data[0], 0.0, 1e-5))

c = rpl.Tensor([0]).erfc()
check("erfc(0)", approx(c.data[0], 1.0, 1e-5))

# nan_to_num
c = rpl.Tensor([1.0, float('nan'), float('inf'), float('-inf')]).nan_to_num(0.0, 999.0, -999.0)
check("nan_to_num(1)", approx(c.data[0], 1.0, 1e-5))
check("nan_to_num(nan)", approx(c.data[1], 0.0, 1e-5))
check("nan_to_num(inf)", approx(c.data[2], 999.0, 1e-5))
check("nan_to_num(-inf)", approx(c.data[3], -999.0, 1e-5))

# trunc/round
c = rpl.Tensor([1.7, -2.3]).trunc()
check("trunc(1.7)", approx(c.data[0], 1.0, 1e-5))

c = rpl.Tensor([1.5, 2.5]).round()
check("round(1.5)", approx(c.data[0], 2.0, 1e-5))

# addcmul / addcdiv
one = rpl.Tensor([1, 1, 1])
va = rpl.Tensor([3, 5, 7])
vb = rpl.Tensor([1, 2, 3])
from rpl.core import _lib
c = rpl.Tensor(_ptr=_lib.tensor_addcmul(one._ptr, va._ptr, vb._ptr, 2.0))
check("addcmul", approx(c.data[0], 1.0 + 2.0*3.0*1.0, 1e-4))
c = rpl.Tensor(_ptr=_lib.tensor_addcdiv(one._ptr, va._ptr, vb._ptr, 1.0))
check("addcdiv", approx(c.data[0], 1.0 + 3.0/1.0, 1e-4))

# ============================================================
# 6. Manipulation
# ============================================================
print("--- Manipulation ---")

# reshape
t = rpl.Tensor([1, 2, 3, 4, 5, 6])
r = t.reshape(2, 3)
check("reshape shape", r.shape == (2, 3))
check("reshape data", approx(r.data.flat[5], 6.0, 1e-5))

# squeeze / unsqueeze
t3 = rpl.Tensor(np.array([1, 2, 3], dtype=np.float32).reshape(1, 3, 1))
r = t3.squeeze()
check("squeeze", r.ndim == 1 and r.shape == (3,))
r = t3.unsqueeze(0)
check("unsqueeze", r.ndim == 4 and r.shape[0] == 1)

# flatten
t2d = rpl.Tensor([[1, 2, 3], [4, 5, 6]])
r = t2d.flatten()
check("flatten", r.ndim == 1 and r.size == 6)

# T (transpose)
r = t2d.T
check("T shape", r.shape == (3, 2))
check("T[0,1]", approx(r.data[0, 1], 4.0, 1e-5))

# cat
a = rpl.Tensor([1, 2, 3])
b = rpl.Tensor([4, 5, 6])
r = rpl.cat([a, b])
check("cat size", r.size == 6)
check("cat[3]", approx(r.data[3], 4.0, 1e-5))

# stack
r = rpl.stack([a, b])
check("stack shape", r.shape == (2, 3))

# clone via C
r = rpl.Tensor([1, 2, 3])
c = r.clone()
check("clone", np.allclose(r.data, c.data))

# flip
r = a.flip(0)
check("flip[0]", approx(r.data[0], 3.0, 1e-5))
check("flip[2]", approx(r.data[2], 1.0, 1e-5))

# roll
r = a.roll(1, dim=0)
check("roll[0]", approx(r.data[0], 3.0, 1e-5))
check("roll[1]", approx(r.data[1], 1.0, 1e-5))

# narrow
t5 = rpl.Tensor([10, 20, 30, 40, 50])
r = t5.narrow(0, 1, 3)
check("narrow size", r.size == 3)
check("narrow[0]", approx(r.data[0], 20.0, 1e-5))
check("narrow[2]", approx(r.data[2], 40.0, 1e-5))

# index_select
r = a.index_select(0, [0, 2])
check("index_select size", r.size == 2)
check("index_select[0]", approx(r.data[0], 1.0, 1e-5))
check("index_select[1]", approx(r.data[1], 3.0, 1e-5))

# where
cond = rpl.Tensor([1, 0, 1])
r = rpl.where(cond, a, b)
check("where[0]", approx(r.data[0], 1.0, 1e-5))
check("where[1]", approx(r.data[1], 5.0, 1e-5))

# tile
r = a.tile([3])
check("tile size", r.size == 9)
check("tile[3]", approx(r.data[3], 1.0, 1e-5))

# ============================================================
# 7. Reductions
# ============================================================
print("--- Reductions ---")
t = rpl.Tensor([1, 2, 3, 4, 5])
check("sum", approx(t.sum(), 15.0))
check("mean", approx(t.mean(), 3.0))
check("max", approx(t.max(), 5.0))
check("min", approx(t.min(), 1.0))
check("argmax", t.argmax() == 4)
check("argmin", t.argmin() == 0)

t2 = rpl.Tensor([1, 2, 3])
check("norm L2", approx(t2.norm(2.0), np.sqrt(14), 0.01))
check("var", approx(rpl.Tensor([2, 4, 4, 4, 5, 5, 7, 9]).var(), 4.571, 0.1))

# prod
check("prod", approx(rpl.Tensor([1, 2, 3, 4, 5, 6]).prod(), 720.0, 0.01))

# median
check("median", approx(rpl.Tensor([1, 2, 3, 4, 5, 6]).median(), 3.5, 0.01))

# count_nonzero
check("count_nonzero", rpl.Tensor([1, 2, 3, 4, 5, 6]).count_nonzero() == 6)

# all/any
check("all(nonzero)", rpl.Tensor([1, 2, 3]).all())
zt = rpl.Tensor([0, 1, 0])
check("any", zt.any())
check("!all(has_zero)", not zt.all())

# Dim reduction
t2d = rpl.Tensor([[1, 2, 3], [4, 5, 6]])
c = t2d.sum(dim=0)
check("sum dim=0 [0]", approx(c.data[0], 5.0, 1e-5))
check("sum dim=0 [2]", approx(c.data[2], 9.0, 1e-5))

c = t2d.sum(dim=1)
check("sum dim=1", np.allclose(c.data, [6, 15]))
c = t2d.mean(dim=1)
check("mean dim=1", np.allclose(c.data, [2, 5]))

c = t2d.max(dim=1)
check("max dim=1 [0]", approx(c.data[0], 3.0, 1e-5))

c = t2d.argmax_dim(1)
check("argmax dim=1 [0]", approx(c.data[0], 2.0, 1e-5))

# cumsum
t6 = rpl.Tensor([1, 2, 3, 4, 5, 6])
c = t6.cumsum()
check("cumsum[0]", approx(c.data[0], 1.0, 1e-5))
check("cumsum[2]", approx(c.data[2], 6.0, 1e-5))
check("cumsum[5]", approx(c.data[5], 21.0, 1e-5))

# diff
c = t6.diff()
check("diff size", c.size == 5)
check("diff[0]", approx(c.data[0], 1.0, 1e-5))

# nan-safe
nanv = rpl.Tensor([1, float('nan'), 3, float('nan'), 5])
check("nansum", approx(nanv.nansum(), 9.0, 1e-4))
check("nanmean", approx(nanv.nanmean(), 3.0, 1e-4))

# dist
da = rpl.Tensor([0, 0])
db = rpl.Tensor([3, 4])
check("dist L2", approx(da.dist(db, 2.0), 5.0, 1e-3))

# ============================================================
# 8. Comparisons
# ============================================================
print("--- Comparisons ---")
a = rpl.Tensor([1, 2, 3])
b = rpl.Tensor([3, 2, 1])
check("eq", np.allclose(a.eq(b).data, [0, 1, 0]))
check("ne", np.allclose(a.ne(b).data, [1, 0, 1]))
check("lt", np.allclose(a.lt(b).data, [1, 0, 0]))
check("gt", np.allclose(a.gt(b).data, [0, 0, 1]))
check("le", np.allclose(a.le(b).data, [1, 1, 0]))
check("ge", np.allclose(a.ge(b).data, [0, 1, 1]))

# equal
check("equal", not a.equal(b))
check("equal self", a.equal(a))

# maximum/minimum
c1 = rpl.Tensor([1, 2, 3, 4])
c2 = rpl.Tensor([2, 2, 2, 2])
r = c1.maximum(c2)
check("maximum[0]", approx(r.data[0], 2.0, 1e-5))
check("maximum[2]", approx(r.data[2], 3.0, 1e-5))
r = c1.minimum(c2)
check("minimum[0]", approx(r.data[0], 1.0, 1e-5))
check("minimum[2]", approx(r.data[2], 2.0, 1e-5))

# logical
la = rpl.Tensor([1, 0, 1, 0])
lb = rpl.Tensor([1, 1, 0, 0])
r = la.logical_and(lb)
check("and[0]", approx(r.data[0], 1.0, 1e-5))
check("and[1]", approx(r.data[1], 0.0, 1e-5))
r = la.logical_or(lb)
check("or[0]", approx(r.data[0], 1.0, 1e-5))
check("or[3]", approx(r.data[3], 0.0, 1e-5))
r = la.logical_not()
check("not[0]", approx(r.data[0], 0.0, 1e-5))
check("not[1]", approx(r.data[1], 1.0, 1e-5))

# isnan/isinf/isfinite
ni = rpl.Tensor([1, float('nan'), float('inf'), float('-inf')])
r = ni.isnan()
check("isnan(1)=0", approx(r.data[0], 0.0, 1e-5))
check("isnan(nan)=1", approx(r.data[1], 1.0, 1e-5))
r = ni.isinf()
check("isinf(inf)=1", approx(r.data[2], 1.0, 1e-5))
r = ni.isfinite()
check("isfinite(1)=1", approx(r.data[0], 1.0, 1e-5))
check("isfinite(nan)=0", approx(r.data[1], 0.0, 1e-5))

# sort
us = rpl.Tensor([3, 1, 4, 1, 5, 9])
sorted_t, _ = us.sort()
check("sort[0]", approx(sorted_t.data[0], 1.0, 1e-5))
check("sort[5]", approx(sorted_t.data[5], 9.0, 1e-5))

# allclose
ac = rpl.Tensor([1.0001, 2.0001, 3.0001])
check("allclose", a.allclose(ac, rtol=1e-3, atol=1e-3))

# unique
uq = rpl.Tensor([3, 1, 2, 1, 3, 2])
_, uc = uq.unique()
check("unique count", uc == 3)

# ============================================================
# 9. Linear Algebra
# ============================================================
print("--- Linalg ---")
a = rpl.Tensor([1, 2, 3])
b = rpl.Tensor([4, 5, 6])

check("dot", approx(a.dot(b), 32.0, 1e-4))

# outer
r = a.outer(b)
check("outer shape", r.shape == (3, 3))
check("outer[0,0]", approx(r.data[0, 0], 4.0, 1e-5))
check("outer[0,2]", approx(r.data[0, 2], 6.0, 1e-5))

# cross
cx = rpl.Tensor([1, 0, 0])
cy = rpl.Tensor([0, 1, 0])
r = cx.cross(cy)
check("cross z", approx(r.data[2], 1.0, 1e-5))

# mv
mat = rpl.Tensor([[1, 2, 3], [4, 5, 6]])
vec = rpl.Tensor([1, 1, 1])
r = mat.mv(vec)
check("mv[0]", approx(r.data[0], 6.0, 1e-5))
check("mv[1]", approx(r.data[1], 15.0, 1e-5))

# eye
r = rpl.eye(3)
check("eye[0,0]", approx(r.data[0, 0], 1.0, 1e-5))
check("eye[0,1]", approx(r.data[0, 1], 0.0, 1e-5))
check("eye[1,1]", approx(r.data[1, 1], 1.0, 1e-5))

# trace
tr = rpl.Tensor([[1, 2], [3, 4]])
check("trace", approx(tr.trace(), 5.0, 1e-5))

# det (2x2)
check("det 2x2", approx(tr.det(), -2.0, 1e-5))

# inverse (2x2)
r = tr.inverse()
check("inv[0,0]", approx(r.data[0, 0], -2.0, 1e-3))
check("inv[0,1]", approx(r.data[0, 1], 1.0, 1e-3))
check("inv[1,0]", approx(r.data[1, 0], 1.5, 1e-3))
check("inv[1,1]", approx(r.data[1, 1], -0.5, 1e-3))

# tril/triu
r = tr.tril()
check("tril[0,1]", approx(r.data[0, 1], 0.0, 1e-5))
check("tril[1,0]", approx(r.data[1, 0], 3.0, 1e-5))
r = tr.triu()
check("triu[1,0]", approx(r.data[1, 0], 0.0, 1e-5))
check("triu[0,1]", approx(r.data[0, 1], 2.0, 1e-5))

# diag (1D -> 2D)
dv = rpl.Tensor([1, 2, 3])
r = dv.diag()
check("diag shape", r.shape == (3, 3))
check("diag[0,0]", approx(r.data[0, 0], 1.0, 1e-5))
check("diag[1,1]", approx(r.data[1, 1], 2.0, 1e-5))
check("diag[0,1]", approx(r.data[0, 1], 0.0, 1e-5))

# cholesky
spd = rpl.Tensor([[4, 2], [2, 3]])
r = spd.cholesky()
check("chol[0,0]", approx(r.data[0, 0], 2.0, 1e-3))
check("chol[1,0]", approx(r.data[1, 0], 1.0, 1e-3))
check("chol[1,1]", approx(r.data[1, 1], math.sqrt(2.0), 1e-3))

# matrix_power
r = tr.matrix_power(2)
check("matpow[0,0]", approx(r.data[0, 0], 7.0, 1e-3))
check("matpow[1,1]", approx(r.data[1, 1], 22.0, 1e-3))

# bmm
ba = rpl.Tensor(np.array([[[1,2],[3,4]], [[5,6],[7,8]]], dtype=np.float32))
bb = rpl.Tensor(np.array([[[1,0],[0,1]], [[2,0],[0,2]]], dtype=np.float32))
r = ba.bmm(bb)
check("bmm[0,0,0]", approx(r.data[0, 0, 0], 1.0, 1e-5))
check("bmm[1,0,0]", approx(r.data[1, 0, 0], 10.0, 1e-5))

# ============================================================
# 10. FFT
# ============================================================
print("--- FFT ---")
t = rpl.Tensor([1, 0, 0, 0])
r = t.fft()
check("fft shape", r.shape == (4, 2))
check("fft[0] re", approx(r.data[0, 0], 1.0, 1e-5))
check("fft[0] im", approx(r.data[0, 1], 0.0, 1e-5))
check("fft[1] re", approx(r.data[1, 0], 1.0, 1e-5))
check("fft[2] re", approx(r.data[2, 0], 1.0, 1e-5))

# IFFT recovery
ir = r.ifft()
check("ifft[0] re", approx(ir.data[0, 0], 1.0, 1e-5))
check("ifft[1] re", approx(ir.data[1, 0], 0.0, 1e-5))

# ============================================================
# 11. Random / Creation
# ============================================================
print("--- Random ---")
rpl.manual_seed(42)

r = rpl.rand(100)
check("rand size", r.size == 100)
in_range = all(0 <= r.data[i] < 1.0 for i in range(r.size))
check("rand range", in_range)

r = rpl.randn(100)
check("randn size", r.size == 100)

z = rpl.zeros(100)
check("zeros", approx(z.data[0], 0.0, 1e-5))

o = rpl.ones(100)
check("ones", approx(o.data[0], 1.0, 1e-5))

r = rpl.arange(0, 5, 1)
check("arange size", r.size == 5)
check("arange[2]", approx(r.data[2], 2.0, 1e-5))

r = rpl.linspace(0, 1, 5)
check("linspace[0]", approx(r.data[0], 0.0, 1e-5))
check("linspace[4]", approx(r.data[4], 1.0, 1e-5))
check("linspace[2]", approx(r.data[2], 0.5, 1e-5))

r = rpl.randperm(10)
check("randperm size", r.size == 10)
check("randperm sum", approx(sum(r.data), 45.0, 1e-4))

# zeros_like / ones_like (via C)
import ctypes
ref = rpl.Tensor([1, 2, 3])
from rpl.core import _lib, RTensor
z_ptr = _lib.tensor_zeros_like(ref._ptr)
z = rpl.Tensor(_ptr=z_ptr)
check("zeros_like shape", z.size == 3 and approx(z.data[0], 0.0, 1e-5))
o_ptr = _lib.tensor_ones_like(ref._ptr)
o = rpl.Tensor(_ptr=o_ptr)
check("ones_like", approx(o.data[0], 1.0, 1e-5))

# ============================================================
# 12. Utility
# ============================================================
print("--- Utility ---")
t = rpl.Tensor([1, 2, 3, 4])
check("numel", t.numel() == 4)
check("is_floating_point", t.is_floating_point())

# Window functions
w = rpl.hann_window(5)
check("hann size", w.size == 5)
check("hann[0]", approx(w.data[0], 0.0, 1e-5))
check("hann[2]", approx(w.data[2], 1.0, 1e-5))

w = rpl.hamming_window(5)
check("hamming size", w.size == 5)

# Fill and randomize
t = rpl.Tensor(shape=(3, 3))
t.fill_(7.0)
check("fill_", np.allclose(t.data, np.full((3, 3), 7.0)))

t.randomize_()
check("randomize_ changes", not np.allclose(t.data, np.full((3, 3), 7.0)))

t_copy = t.clone()
check("clone", np.allclose(t.data, t_copy.data))

n = t.numpy()
check("numpy", isinstance(n, np.ndarray) and n.shape == (3, 3))

# ============================================================
# 13. Linear Layer
# ============================================================
print("--- Linear Layer ---")
fc = rpl.nn.Linear(3, 2)
x = rpl.Tensor([[1.0, 1.0, 1.0]])
y = fc(x)
check("linear shape", y.shape == (1, 2))
expected = np.matmul(x.data, fc.weight.data.T) + fc.bias.data
check("linear values", np.allclose(y.data, expected, atol=1e-4))

# ============================================================
# Results
# ============================================================
print(f"\n=== Python API: {passed} passed, {failed} failed ===")
sys.exit(1 if failed > 0 else 0)

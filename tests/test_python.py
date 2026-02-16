#!/usr/bin/env python3
"""Comprehensive Python API test for RPL library.
Tests: Tensor creation, arithmetic, activations, math, reductions, comparisons.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))

import rpl
import numpy as np

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

# ============================================================
# 6. Reductions
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

# Dim reduction
t2d = rpl.Tensor([[1, 2, 3], [4, 5, 6]])
c = t2d.sum(dim=1)
check("sum dim=1", np.allclose(c.data, [6, 15]))
c = t2d.mean(dim=1)
check("mean dim=1", np.allclose(c.data, [2, 5]))

# ============================================================
# 7. Comparisons
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

# ============================================================
# 8. Linear Layer
# ============================================================
print("--- Linear Layer ---")
fc = rpl.nn.Linear(3, 2)
x = rpl.Tensor([[1.0, 1.0, 1.0]])
y = fc(x)
check("linear shape", y.shape == (1, 2))
expected = np.matmul(x.data, fc.weight.data.T) + fc.bias.data
check("linear values", np.allclose(y.data, expected, atol=1e-4))

# ============================================================
# 9. Utility
# ============================================================
print("--- Utility ---")
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
# Results
# ============================================================
print(f"\n=== Python API: {passed} passed, {failed} failed ===")
sys.exit(1 if failed > 0 else 0)

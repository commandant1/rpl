import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))

import rpl
import numpy as np

def check(name, condition, detail=""):
    if condition:
        print(f"PASS: {name}")
    else:
        print(f"FAIL: {name} {detail}")
        sys.exit(1)

# 1. Dynamic Intermediate Gradient Freeing
print("--- Dynamic Intermediate Gradient Freeing Check ---")
x = rpl.Tensor([2.0, -1.0, 3.0], requires_grad=True)
y = x.relu()
z = y.sum(0)
z.backward()

check("Leaf node gradient populated", x.grad is not None and np.allclose(x.grad, [1.0, 0.0, 1.0]))
check("Intermediate node gradient is freed", y.grad is None)

# 2. zero_grad(set_to_none=True/False)
print("--- zero_grad set_to_none Check ---")
# Check on individual Tensor
x.zero_grad(set_to_none=False)
check("zero_grad(set_to_none=False) leaves grad as zeros", x.grad is not None and np.allclose(x.grad, 0.0))

x.zero_grad(set_to_none=True)
check("zero_grad(set_to_none=True) sets grad to None", x.grad is None)

# Check on Optimizer
model_param = rpl.Tensor([1.0, 2.0], requires_grad=True)
opt = rpl.optim.SGD([model_param], lr=0.1)

# Populate gradient
y2 = (model_param * 2.0).sum(0)
y2.backward()
check("Parameter gradient populated before zero_grad", model_param.grad is not None)

opt.zero_grad(set_to_none=False)
check("Optimizer zero_grad(set_to_none=False) leaves parameter grad as zeros", model_param.grad is not None and np.allclose(model_param.grad, 0.0))

# Populate gradient again
y3 = (model_param * 3.0).sum(0)
y3.backward()
check("Parameter gradient populated again before zero_grad(set_to_none=True)", model_param.grad is not None)

opt.zero_grad(set_to_none=True)
check("Optimizer zero_grad(set_to_none=True) sets parameter grad to None", model_param.grad is None)

# 3. empty_cache
print("--- empty_cache Check ---")
rpl.empty_cache()
check("empty_cache can be called without error", True)

# 4. Inplace Activation Methods
print("--- Inplace Activation Methods Check ---")

# relu_
a = rpl.Tensor([-1.0, 0.0, 2.0])
res = a.relu_()
check("relu_ modifies tensor in-place", a is res and np.allclose(a.data, [0.0, 0.0, 2.0]))

# sigmoid_
a = rpl.Tensor([0.0, 1.0, -1.0])
expected_sig = 1.0 / (1.0 + np.exp(-np.array([0.0, 1.0, -1.0])))
res = a.sigmoid_()
check("sigmoid_ modifies tensor in-place", a is res and np.allclose(a.data, expected_sig, atol=1e-5))

# tanh_
a = rpl.Tensor([0.0, 0.5, -0.5])
expected_tanh = np.tanh(np.array([0.0, 0.5, -0.5]))
res = a.tanh_()
check("tanh_ modifies tensor in-place", a is res and np.allclose(a.data, expected_tanh, atol=1e-5))

# gelu_
a = rpl.Tensor([-1.0, 0.0, 1.0])
# We can check by running non-inplace gelu on a fresh clone and compare
a_clone = rpl.Tensor([-1.0, 0.0, 1.0])
expected_gelu = a_clone.gelu()
res = a.gelu_()
check("gelu_ modifies tensor in-place", a is res and np.allclose(a.data, expected_gelu.data, atol=1e-5))

# softmax_
a = rpl.Tensor([1.0, 2.0, 3.0])
# We can check by running non-inplace softmax on a fresh clone
a_clone = rpl.Tensor([1.0, 2.0, 3.0])
expected_softmax = a_clone.softmax()
res = a.softmax_()
check("softmax_ modifies tensor in-place", a is res and np.allclose(a.data, expected_softmax.data, atol=1e-5))

print("=== All memory optimization verification checks passed successfully! ===")

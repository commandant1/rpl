import ctypes
from .core import _lib, Tensor, RTensor

# ============================================================
# Linear Layer
# ============================================================

class RLinear(ctypes.Structure):
    _fields_ = [
        ("weight", ctypes.POINTER(RTensor)),
        ("bias", ctypes.POINTER(RTensor)),
        ("in_features", ctypes.c_uint32),
        ("out_features", ctypes.c_uint32),
    ]

_lib.linear_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
_lib.linear_create.restype = ctypes.POINTER(RLinear)
_lib.linear_forward.argtypes = [ctypes.POINTER(RLinear), ctypes.POINTER(RTensor)]
_lib.linear_forward.restype = ctypes.POINTER(RTensor)
_lib.linear_free.argtypes = [ctypes.POINTER(RLinear)]
_lib.linear_free.restype = None

class Linear:
    def __init__(self, in_features, out_features):
        self._ptr = _lib.linear_create(in_features, out_features)
        self.weight = Tensor(_ptr=self._ptr.contents.weight)
        self.bias = Tensor(_ptr=self._ptr.contents.bias)

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            _lib.linear_free(self._ptr)
            self._ptr = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        out_ptr = _lib.linear_forward(self._ptr, x._ptr)
        return Tensor(_ptr=out_ptr)

# ============================================================
# Activation Modules — PyTorch-compatible API
# ============================================================

class ReLU:
    """ReLU activation: max(0, x)"""
    def __call__(self, x):
        return x.relu()

class Sigmoid:
    """Sigmoid activation: 1/(1+exp(-x))"""
    def __call__(self, x):
        return x.sigmoid()

class Tanh:
    """Tanh activation"""
    def __call__(self, x):
        return x.tanh()

class GELU:
    """Gaussian Error Linear Unit"""
    def __call__(self, x):
        return x.gelu()

class LeakyReLU:
    """Leaky ReLU: max(0,x) + negative_slope*min(0,x)"""
    def __init__(self, negative_slope=0.01):
        self.negative_slope = negative_slope
    def __call__(self, x):
        return x.leaky_relu(self.negative_slope)

class ELU:
    """Exponential Linear Unit"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    def __call__(self, x):
        return x.elu(self.alpha)

class SELU:
    """Scaled Exponential Linear Unit"""
    def __call__(self, x):
        return x.selu()

class Swish:
    """Swish/SiLU: x * sigmoid(x)"""
    def __call__(self, x):
        return x.swish()

SiLU = Swish  # PyTorch alias

class Mish:
    """Mish: x * tanh(softplus(x))"""
    def __call__(self, x):
        return x.mish()

class Hardswish:
    """Hardswish: x * clip(x+3, 0, 6) / 6"""
    def __call__(self, x):
        return x.hardswish()

class Hardsigmoid:
    """Hardsigmoid: clip(x/6 + 0.5, 0, 1)"""
    def __call__(self, x):
        return x.hardsigmoid()

class Hardtanh:
    """Hardtanh: clamp to [min_val, max_val]"""
    def __init__(self, min_val=-1.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val
    def __call__(self, x):
        return x.hardtanh(self.min_val, self.max_val)

class CELU:
    """Continuously-differentiable ELU"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    def __call__(self, x):
        return x.celu(self.alpha)

class Softplus:
    """Softplus: log(1 + exp(beta*x)) / beta"""
    def __init__(self, beta=1.0, threshold=20.0):
        self.beta = beta
        self.threshold = threshold
    def __call__(self, x):
        return x.softplus(self.beta, self.threshold)

class Softsign:
    """Softsign: x / (1 + |x|)"""
    def __call__(self, x):
        return x.softsign()

class LogSoftmax:
    """LogSoftmax: x - log(sum(exp(x)))"""
    def __call__(self, x):
        return x.log_softmax()

class Softmax:
    """Softmax"""
    def __call__(self, x):
        return x.softmax()

class PReLU:
    """Parametric ReLU"""
    def __init__(self, num_parameters=1, init=0.25):
        self.weight = Tensor([init] * num_parameters)
    def __call__(self, x):
        out = x._make_like()
        _lib.tensor_prelu(out, x._ptr, self.weight._ptr)
        return Tensor(_ptr=out)

class RReLU:
    """Randomized ReLU (uses mean slope at eval)"""
    def __init__(self, lower=1.0/8, upper=1.0/3):
        self.lower = lower
        self.upper = upper
    def __call__(self, x):
        return x.rrelu(self.lower, self.upper)

class Threshold:
    """Threshold: x if x > threshold, else value"""
    def __init__(self, threshold, value):
        self.threshold_val = threshold
        self.value = value
    def __call__(self, x):
        return x.threshold(self.threshold_val, self.value)

# ============================================================
# Loss Functions
# ============================================================

_lib.tensor_mse_loss.argtypes = [ctypes.POINTER(RTensor), ctypes.POINTER(RTensor)]
_lib.tensor_mse_loss.restype = ctypes.POINTER(RTensor)
_lib.mse_loss.argtypes = [ctypes.POINTER(RTensor), ctypes.POINTER(RTensor)]
_lib.mse_loss.restype = ctypes.c_float

def mse_loss(input, target):
    out_ptr = _lib.tensor_mse_loss(input._ptr, target._ptr)
    return Tensor(_ptr=out_ptr)

def cross_entropy_loss(input, target):
    return _lib.cross_entropy_loss(input._ptr, target._ptr)

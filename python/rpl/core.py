import ctypes
import os
import numpy as np

# Load the shared library
_lib_dir = os.path.join(os.path.dirname(__file__), "../../build")
lib_path = os.path.join(_lib_dir, "librpl.so")
try:
    _lib = ctypes.CDLL(lib_path)
except OSError:
    _lib = ctypes.CDLL("librpl.so")

MAX_DIMS = 8

class RTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("grad", ctypes.POINTER(ctypes.c_float)),
        ("dims", ctypes.c_uint32),
        ("shape", ctypes.c_uint32 * MAX_DIMS),
        ("strides", ctypes.c_uint32 * MAX_DIMS),
        ("size", ctypes.c_uint32),
        ("requires_grad", ctypes.c_bool),
        ("_allocation", ctypes.c_void_p),
        ("_alloc_size", ctypes.c_size_t),
        ("is_leaf", ctypes.c_bool),
        ("parent1", ctypes.c_void_p),
        ("parent2", ctypes.c_void_p),
        ("backward_fn", ctypes.c_void_p),
        ("device", ctypes.c_int),
        ("gpu_buffer", ctypes.c_uint32),
    ]

class Device:
    CPU = 0
    GPU = 1

# ============================================================
# C function prototypes
# ============================================================
_PTR = ctypes.POINTER(RTensor)
_F = ctypes.c_float
_U32 = ctypes.c_uint32
_I32 = ctypes.c_int32
_BOOL = ctypes.c_bool

def _sig(name, argtypes, restype):
    fn = getattr(_lib, name)
    fn.argtypes = argtypes
    fn.restype = restype
    return fn

# Core
_sig("tensor_create", [_U32, ctypes.POINTER(_U32), _BOOL], _PTR)
_sig("tensor_free", [_PTR], None)
_sig("tensor_fill", [_PTR, _F], None)
_sig("tensor_add", [_PTR, _PTR], _PTR)
_sig("tensor_add_out", [_PTR, _PTR, _PTR], None)
_sig("tensor_mul", [_PTR, _PTR], _PTR)
_sig("tensor_mul_out", [_PTR, _PTR, _PTR], None)
_sig("tensor_matmul", [_PTR, _PTR], _PTR)
_sig("tensor_add_inplace", [_PTR, _PTR], None)
_sig("tensor_mul_inplace", [_PTR, _F], None)
_sig("tensor_randomize", [_PTR], None)
_sig("tensor_backward", [_PTR], None)
_sig("tensor_zero_grad", [_PTR], None)

# Activations returning new tensor
_sig("tensor_relu", [_PTR], _PTR)
_sig("tensor_sigmoid", [_PTR], _PTR)

# Inplace activations
_sig("tensor_relu_inplace", [_PTR], None)
_sig("tensor_sigmoid_inplace", [_PTR], None)
_sig("tensor_tanh_inplace", [_PTR], None)
_sig("tensor_gelu_inplace", [_PTR], None)
_sig("tensor_softmax_inplace", [_PTR], None)

# Activations (out, in) style
_sig("tensor_leaky_relu", [_PTR, _PTR, _F], None)
_sig("tensor_leaky_relu_inplace", [_PTR, _F], None)
_sig("tensor_elu", [_PTR, _PTR, _F], None)
_sig("tensor_elu_inplace", [_PTR, _F], None)
_sig("tensor_swish", [_PTR, _PTR], None)
_sig("tensor_swish_inplace", [_PTR], None)
_sig("tensor_softplus", [_PTR, _PTR, _F, _F], None)
_sig("tensor_softplus_inplace", [_PTR, _F, _F], None)
_sig("tensor_gelu", [_PTR, _PTR], None)
_sig("tensor_selu", [_PTR, _PTR], None)
_sig("tensor_selu_inplace", [_PTR], None)
_sig("tensor_mish", [_PTR, _PTR], None)
_sig("tensor_mish_inplace", [_PTR], None)
_sig("tensor_hardswish", [_PTR, _PTR], None)
_sig("tensor_hardswish_inplace", [_PTR], None)
_sig("tensor_hardsigmoid", [_PTR, _PTR], None)
_sig("tensor_hardsigmoid_inplace", [_PTR], None)
_sig("tensor_hardtanh", [_PTR, _PTR, _F, _F], None)
_sig("tensor_hardtanh_inplace", [_PTR, _F, _F], None)
_sig("tensor_celu", [_PTR, _PTR, _F], None)
_sig("tensor_celu_inplace", [_PTR, _F], None)
_sig("tensor_softsign", [_PTR, _PTR], None)
_sig("tensor_softsign_inplace", [_PTR], None)
_sig("tensor_log_softmax", [_PTR, _PTR], None)
_sig("tensor_log_softmax_inplace", [_PTR], None)
_sig("tensor_prelu", [_PTR, _PTR, _PTR], None)
_sig("tensor_rrelu", [_PTR, _PTR, _F, _F], None)
_sig("tensor_rrelu_inplace", [_PTR, _F, _F], None)
_sig("tensor_threshold", [_PTR, _PTR, _F, _F], None)
_sig("tensor_threshold_inplace", [_PTR, _F, _F], None)

# Math — unary ops returning new tensor
for _name in [
    "neg", "abs_op", "sign", "reciprocal", "square", "rsqrt",
    "sqrt_op", "cbrt",
    "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh",
    "exp", "exp2", "expm1", "log", "log2", "log10", "log1p",
    "floor_op", "ceil_op", "round_op", "trunc_op", "frac",
    "erf", "erfc",
]:
    _sig(f"tensor_{_name}", [_PTR], _PTR)

# Math — binary ops returning new tensor
for _name in ["pow_op", "atan2", "hypot", "fmod_op", "copysign_op"]:
    _sig(f"tensor_{_name}", [_PTR, _PTR], _PTR)

# Clamp, lerp, sub, div
_sig("tensor_clamp", [_PTR, _F, _F], _PTR)
_sig("tensor_lerp", [_PTR, _PTR, _F], _PTR)
_sig("tensor_sub", [_PTR, _PTR], _PTR)
_sig("tensor_div", [_PTR, _PTR], _PTR)
_sig("tensor_addcmul", [_PTR, _PTR, _PTR, _F], _PTR)
_sig("tensor_addcdiv", [_PTR, _PTR, _PTR, _F], _PTR)

# Reductions — full (return scalar)
_sig("tensor_sum_all", [_PTR], _F)
_sig("tensor_prod_all", [_PTR], _F)
_sig("tensor_mean_all", [_PTR], _F)
_sig("tensor_var_all", [_PTR, _BOOL], _F)
_sig("tensor_std_all", [_PTR, _BOOL], _F)
_sig("tensor_max_all", [_PTR], _F)
_sig("tensor_min_all", [_PTR], _F)
_sig("tensor_norm_all", [_PTR, _F], _F)
_sig("tensor_argmax_all", [_PTR], _U32)
_sig("tensor_argmin_all", [_PTR], _U32)

# Reductions — along dim (return tensor)
_sig("tensor_sum", [_PTR, _I32], _PTR)
_sig("tensor_mean", [_PTR, _I32], _PTR)
_sig("tensor_max_dim", [_PTR, _I32], _PTR)
_sig("tensor_min_dim", [_PTR, _I32], _PTR)

# Compare
for _name in ["eq", "ne", "lt", "le", "gt", "ge"]:
    _sig(f"tensor_{_name}", [_PTR, _PTR], _PTR)
_sig("tensor_maximum", [_PTR, _PTR], _PTR)
_sig("tensor_minimum", [_PTR, _PTR], _PTR)

# Linalg
_sig("tensor_dot", [_PTR, _PTR], _F)

# GPU prototypes
try:
    _sig("tensor_to_gpu", [_PTR], None)
    _sig("tensor_from_gpu", [_PTR], None)
    _sig("tensor_add_gpu", [_PTR, _PTR, _PTR], None)
    _sig("tensor_matmul_gpu", [_PTR, _PTR, _PTR], None)
    _sig("tensor_relu_gpu", [_PTR, _PTR], None)
    _sig("tensor_tanh_gpu", [_PTR, _PTR], None)
    _sig("tensor_gelu_gpu", [_PTR, _PTR], None)
except AttributeError:
    pass


# ============================================================
# Tensor class
# ============================================================

class Tensor:
    def __init__(self, data=None, shape=None, requires_grad=False, _ptr=None):
        if _ptr:
            self._ptr = _ptr
            self._owns_ptr = False
        elif data is not None:
            data = np.array(data, dtype=np.float32)
            c_shape = (ctypes.c_uint32 * len(data.shape))(*data.shape)
            self._ptr = _lib.tensor_create(len(data.shape), c_shape, requires_grad)
            self._owns_ptr = True
            ctypes.memmove(self._ptr.contents.data, data.ctypes.data, data.nbytes)
        elif shape is not None:
            if isinstance(shape, int):
                shape = (shape,)
            c_shape = (ctypes.c_uint32 * len(shape))(*shape)
            self._ptr = _lib.tensor_create(len(shape), c_shape, requires_grad)
            self._owns_ptr = True
        else:
            raise ValueError("Must provide data, shape, or _ptr")

    def __del__(self):
        if getattr(self, "_owns_ptr", False) and self._ptr:
            try:
                _lib.tensor_free(self._ptr)
                self._ptr = None
            except:
                pass

    def _make_like(self):
        """Create a new tensor with the same shape."""
        c_shape = (ctypes.c_uint32 * self._ptr.contents.dims)(*self.shape)
        ptr = _lib.tensor_create(self._ptr.contents.dims, c_shape, False)
        return ptr

    @property
    def shape(self):
        return tuple(self._ptr.contents.shape[:self._ptr.contents.dims])

    @property
    def ndim(self):
        return self._ptr.contents.dims

    @property
    def size(self):
        return self._ptr.contents.size

    @property
    def data(self):
        size = self._ptr.contents.size
        buf = ctypes.cast(self._ptr.contents.data, ctypes.POINTER(ctypes.c_float * size))
        return np.frombuffer(buf.contents, dtype=np.float32).reshape(self.shape)

    @property
    def grad(self):
        if not self._ptr.contents.grad:
            return None
        size = self._ptr.contents.size
        buf = ctypes.cast(self._ptr.contents.grad, ctypes.POINTER(ctypes.c_float * size))
        return np.frombuffer(buf.contents, dtype=np.float32).reshape(self.shape)

    @property
    def device(self):
        return self._ptr.contents.device

    # --- Autograd ---
    def backward(self):
        _lib.tensor_backward(self._ptr)

    def zero_grad(self):
        _lib.tensor_zero_grad(self._ptr)

    # --- Device ---
    def to_gpu(self):
        if hasattr(_lib, 'tensor_to_gpu'):
            _lib.tensor_to_gpu(self._ptr)
        return self

    def to_cpu(self):
        if hasattr(_lib, 'tensor_from_gpu'):
            _lib.tensor_from_gpu(self._ptr)
        return self

    # --- Arithmetic ---
    def __add__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Only Tensor additions supported")
        out_ptr = _lib.tensor_add(self._ptr, other._ptr)
        return Tensor(_ptr=out_ptr)

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Only Tensor subtractions supported")
        out_ptr = _lib.tensor_sub(self._ptr, other._ptr)
        return Tensor(_ptr=out_ptr)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            out_ptr = _lib.tensor_mul(self._ptr, other._ptr)
            return Tensor(_ptr=out_ptr)
        elif isinstance(other, (int, float)):
            # scalar mul via inplace on clone
            out = self.clone()
            _lib.tensor_mul_inplace(out._ptr, float(other))
            return out
        raise TypeError("Unsupported operand type")

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Only Tensor matmul supported")
        out_ptr = _lib.tensor_matmul(self._ptr, other._ptr)
        return Tensor(_ptr=out_ptr)

    def __neg__(self):
        out_ptr = _lib.tensor_neg(self._ptr)
        return Tensor(_ptr=out_ptr)

    def __abs__(self):
        out_ptr = _lib.tensor_abs_op(self._ptr)
        return Tensor(_ptr=out_ptr)

    # --- Clone ---
    def clone(self):
        t = Tensor(shape=self.shape)
        t._owns_ptr = True
        ctypes.memmove(t._ptr.contents.data, self._ptr.contents.data,
                       self.size * ctypes.sizeof(ctypes.c_float))
        return t

    # --- Activations (return new tensor) ---
    def relu(self):
        out_ptr = _lib.tensor_relu(self._ptr)
        return Tensor(_ptr=out_ptr)

    def sigmoid(self):
        out_ptr = _lib.tensor_sigmoid(self._ptr)
        return Tensor(_ptr=out_ptr)

    def tanh(self):
        out = self.clone()
        _lib.tensor_tanh_inplace(out._ptr)
        return out

    def gelu(self):
        out = self._make_like()
        _lib.tensor_gelu(out, self._ptr)
        return Tensor(_ptr=out)

    def leaky_relu(self, negative_slope=0.01):
        out = self._make_like()
        _lib.tensor_leaky_relu(out, self._ptr, negative_slope)
        return Tensor(_ptr=out)

    def elu(self, alpha=1.0):
        out = self._make_like()
        _lib.tensor_elu(out, self._ptr, alpha)
        return Tensor(_ptr=out)

    def selu(self):
        out = self._make_like()
        _lib.tensor_selu(out, self._ptr)
        return Tensor(_ptr=out)

    def swish(self):
        out = self._make_like()
        _lib.tensor_swish(out, self._ptr)
        return Tensor(_ptr=out)

    def mish(self):
        out = self._make_like()
        _lib.tensor_mish(out, self._ptr)
        return Tensor(_ptr=out)

    def hardswish(self):
        out = self._make_like()
        _lib.tensor_hardswish(out, self._ptr)
        return Tensor(_ptr=out)

    def hardsigmoid(self):
        out = self._make_like()
        _lib.tensor_hardsigmoid(out, self._ptr)
        return Tensor(_ptr=out)

    def hardtanh(self, min_val=-1.0, max_val=1.0):
        out = self._make_like()
        _lib.tensor_hardtanh(out, self._ptr, min_val, max_val)
        return Tensor(_ptr=out)

    def celu(self, alpha=1.0):
        out = self._make_like()
        _lib.tensor_celu(out, self._ptr, alpha)
        return Tensor(_ptr=out)

    def softsign(self):
        out = self._make_like()
        _lib.tensor_softsign(out, self._ptr)
        return Tensor(_ptr=out)

    def softplus(self, beta=1.0, threshold=20.0):
        out = self._make_like()
        _lib.tensor_softplus(out, self._ptr, beta, threshold)
        return Tensor(_ptr=out)

    def log_softmax(self):
        out = self._make_like()
        _lib.tensor_log_softmax(out, self._ptr)
        return Tensor(_ptr=out)

    def softmax(self):
        out = self.clone()
        _lib.tensor_softmax_inplace(out._ptr)
        return out

    def rrelu(self, lower=1.0/8, upper=1.0/3):
        out = self._make_like()
        _lib.tensor_rrelu(out, self._ptr, lower, upper)
        return Tensor(_ptr=out)

    def threshold(self, threshold, value):
        out = self._make_like()
        _lib.tensor_threshold(out, self._ptr, threshold, value)
        return Tensor(_ptr=out)

    # --- Math ops (return new tensor) ---
    def abs(self):
        return Tensor(_ptr=_lib.tensor_abs_op(self._ptr))

    def neg(self):
        return Tensor(_ptr=_lib.tensor_neg(self._ptr))

    def sign(self):
        return Tensor(_ptr=_lib.tensor_sign(self._ptr))

    def reciprocal(self):
        return Tensor(_ptr=_lib.tensor_reciprocal(self._ptr))

    def square(self):
        return Tensor(_ptr=_lib.tensor_square(self._ptr))

    def sqrt(self):
        return Tensor(_ptr=_lib.tensor_sqrt_op(self._ptr))

    def rsqrt(self):
        return Tensor(_ptr=_lib.tensor_rsqrt(self._ptr))

    def exp(self):
        return Tensor(_ptr=_lib.tensor_exp(self._ptr))

    def log(self):
        return Tensor(_ptr=_lib.tensor_log(self._ptr))

    def sin(self):
        return Tensor(_ptr=_lib.tensor_sin(self._ptr))

    def cos(self):
        return Tensor(_ptr=_lib.tensor_cos(self._ptr))

    def clamp(self, min_val, max_val):
        return Tensor(_ptr=_lib.tensor_clamp(self._ptr, min_val, max_val))

    def floor(self):
        return Tensor(_ptr=_lib.tensor_floor_op(self._ptr))

    def ceil(self):
        return Tensor(_ptr=_lib.tensor_ceil_op(self._ptr))

    def round(self):
        return Tensor(_ptr=_lib.tensor_round_op(self._ptr))

    # --- Reductions ---
    def sum(self, dim=None):
        if dim is None:
            return float(_lib.tensor_sum_all(self._ptr))
        return Tensor(_ptr=_lib.tensor_sum(self._ptr, dim))

    def mean(self, dim=None):
        if dim is None:
            return float(_lib.tensor_mean_all(self._ptr))
        return Tensor(_ptr=_lib.tensor_mean(self._ptr, dim))

    def var(self, unbiased=True):
        return float(_lib.tensor_var_all(self._ptr, unbiased))

    def std(self, unbiased=True):
        return float(_lib.tensor_std_all(self._ptr, unbiased))

    def max(self, dim=None):
        if dim is None:
            return float(_lib.tensor_max_all(self._ptr))
        return Tensor(_ptr=_lib.tensor_max_dim(self._ptr, dim))

    def min(self, dim=None):
        if dim is None:
            return float(_lib.tensor_min_all(self._ptr))
        return Tensor(_ptr=_lib.tensor_min_dim(self._ptr, dim))

    def argmax(self):
        return int(_lib.tensor_argmax_all(self._ptr))

    def argmin(self):
        return int(_lib.tensor_argmin_all(self._ptr))

    def norm(self, p=2.0):
        return float(_lib.tensor_norm_all(self._ptr, p))

    # --- Comparison ---
    def eq(self, other):
        return Tensor(_ptr=_lib.tensor_eq(self._ptr, other._ptr))

    def ne(self, other):
        return Tensor(_ptr=_lib.tensor_ne(self._ptr, other._ptr))

    def lt(self, other):
        return Tensor(_ptr=_lib.tensor_lt(self._ptr, other._ptr))

    def le(self, other):
        return Tensor(_ptr=_lib.tensor_le(self._ptr, other._ptr))

    def gt(self, other):
        return Tensor(_ptr=_lib.tensor_gt(self._ptr, other._ptr))

    def ge(self, other):
        return Tensor(_ptr=_lib.tensor_ge(self._ptr, other._ptr))

    # --- Linalg ---
    def dot(self, other):
        return float(_lib.tensor_dot(self._ptr, other._ptr))

    # --- Utility ---
    def fill_(self, value):
        _lib.tensor_fill(self._ptr, value)
        return self

    def randomize_(self):
        _lib.tensor_randomize(self._ptr)
        return self

    def numpy(self):
        """Return a copy as numpy array."""
        return self.data.copy()

    def __repr__(self):
        return f"rpl.Tensor({self.data}, requires_grad={self._ptr.contents.requires_grad})"

    def __len__(self):
        return self.shape[0] if self.ndim > 0 else 1

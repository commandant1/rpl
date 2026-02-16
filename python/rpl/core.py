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
        ("device", ctypes.c_int32),
        ("gpu_buffer", ctypes.c_uint32),
        ("is_leaf", ctypes.c_bool),
        ("parent1", ctypes.c_void_p),
        ("parent2", ctypes.c_void_p),
        ("backward_fn", ctypes.c_void_p),
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
_sig("tensor_outer", [_PTR, _PTR], _PTR)
_sig("tensor_cross", [_PTR, _PTR], _PTR)
_sig("tensor_mv", [_PTR, _PTR], _PTR)
_sig("tensor_eye", [_U32], _PTR)
_sig("tensor_trace", [_PTR], _F)
_sig("tensor_det", [_PTR], _F)
_sig("tensor_inverse", [_PTR], _PTR)
_sig("tensor_tril", [_PTR, _I32], _PTR)
_sig("tensor_triu", [_PTR, _I32], _PTR)
_sig("tensor_diag", [_PTR, _I32], _PTR)
_sig("tensor_cholesky", [_PTR], _PTR)
_sig("tensor_matrix_power", [_PTR, _I32], _PTR)
_sig("tensor_bmm", [_PTR, _PTR], _PTR)

# Manipulation
_sig("tensor_reshape", [_PTR, _U32, ctypes.POINTER(_U32)], _PTR)
_sig("tensor_squeeze", [_PTR], _PTR)
_sig("tensor_unsqueeze", [_PTR, _I32], _PTR)
_sig("tensor_flatten", [_PTR, _I32, _I32], _PTR)
_sig("tensor_t_op", [_PTR], _PTR)
_sig("tensor_cat", [ctypes.POINTER(_PTR), _U32, _I32], _PTR)
_sig("tensor_stack", [ctypes.POINTER(_PTR), _U32, _I32], _PTR)
_sig("tensor_chunk", [_PTR, _U32, _I32, ctypes.POINTER(_U32)], ctypes.POINTER(_PTR))
_sig("tensor_clone", [_PTR], _PTR)
_sig("tensor_flip", [_PTR, ctypes.POINTER(_I32), _U32], _PTR)
_sig("tensor_roll", [_PTR, _I32, _I32], _PTR)
_sig("tensor_narrow", [_PTR, _I32, _U32, _U32], _PTR)
_sig("tensor_index_select", [_PTR, _I32, ctypes.POINTER(_U32), _U32], _PTR)
_sig("tensor_where_cond", [_PTR, _PTR, _PTR], _PTR)
_sig("tensor_tile", [_PTR, ctypes.POINTER(_U32), _U32], _PTR)

# FFT
_sig("tensor_fft", [_PTR], _PTR)
_sig("tensor_ifft", [_PTR], _PTR)

# Random / Creation
_sig("rpl_manual_seed", [ctypes.c_uint64], None)
_sig("tensor_rand", [_U32, ctypes.POINTER(_U32)], _PTR)
_sig("tensor_randn", [_U32, ctypes.POINTER(_U32)], _PTR)
_sig("tensor_zeros", [_U32, ctypes.POINTER(_U32)], _PTR)
_sig("tensor_ones", [_U32, ctypes.POINTER(_U32)], _PTR)
_sig("tensor_arange", [_F, _F, _F], _PTR)
_sig("tensor_linspace", [_F, _F, _U32], _PTR)
_sig("tensor_randperm", [_U32], _PTR)
_sig("tensor_zeros_like", [_PTR], _PTR)
_sig("tensor_ones_like", [_PTR], _PTR)

# Utility
_sig("tensor_numel", [_PTR], _U32)
_sig("tensor_is_floating_point", [_PTR], _BOOL)
_sig("tensor_hann_window", [_U32], _PTR)
_sig("tensor_hamming_window", [_U32], _PTR)
_sig("tensor_bincount", [_PTR, _U32], _PTR)
_sig("tensor_histc", [_PTR, _U32, _F, _F], _PTR)
_sig("tensor_broadcast_to", [_PTR, _U32, ctypes.POINTER(_U32)], _PTR)
_sig("tensor_convolve", [_PTR, _PTR], _PTR)
_sig("tensor_interp", [_PTR, _PTR, _PTR], _PTR)
_sig("tensor_trapezoid", [_PTR, _F], _F)

# Missing math
_sig("tensor_nan_to_num", [_PTR, _F, _F, _F], _PTR)
_sig("tensor_lerp", [_PTR, _PTR, _F], _PTR)
_sig("tensor_addcmul", [_PTR, _PTR, _PTR, _F], _PTR)
_sig("tensor_addcdiv", [_PTR, _PTR, _PTR, _F], _PTR)
_sig("tensor_div", [_PTR, _PTR], _PTR)
_sig("tensor_sub", [_PTR, _PTR], _PTR)

# Missing reduce
_sig("tensor_prod_all", [_PTR], _F)
_sig("tensor_median_all", [_PTR], _F)
_sig("tensor_count_nonzero_all", [_PTR], _U32)
_sig("tensor_cumsum", [_PTR, _I32], _PTR)
_sig("tensor_diff", [_PTR, _I32], _PTR)
_sig("tensor_all", [_PTR], _BOOL)
_sig("tensor_any", [_PTR], _BOOL)
_sig("tensor_nansum_all", [_PTR], _F)
_sig("tensor_nanmean_all", [_PTR], _F)
_sig("tensor_nanmax_all", [_PTR], _F)
_sig("tensor_nanmin_all", [_PTR], _F)
_sig("tensor_dist", [_PTR, _PTR, _F], _F)
_sig("tensor_argmax_dim", [_PTR, _I32], _PTR)

# Missing compare
_sig("tensor_logical_and", [_PTR, _PTR], _PTR)
_sig("tensor_logical_or", [_PTR, _PTR], _PTR)
_sig("tensor_logical_not", [_PTR], _PTR)
_sig("tensor_isnan_op", [_PTR], _PTR)
_sig("tensor_isinf_op", [_PTR], _PTR)
_sig("tensor_isfinite_op", [_PTR], _PTR)
_sig("tensor_sort_op", [_PTR, _I32, _BOOL, ctypes.POINTER(_PTR)], _PTR)
_sig("tensor_equal", [_PTR, _PTR], _BOOL)
_sig("tensor_allclose", [_PTR, _PTR, _F, _F], _BOOL)
_sig("tensor_unique", [_PTR, ctypes.POINTER(_U32)], _PTR)

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

    def outer(self, other):
        return Tensor(_ptr=_lib.tensor_outer(self._ptr, other._ptr))

    def cross(self, other):
        return Tensor(_ptr=_lib.tensor_cross(self._ptr, other._ptr))

    def mv(self, vec):
        return Tensor(_ptr=_lib.tensor_mv(self._ptr, vec._ptr))

    def trace(self):
        return float(_lib.tensor_trace(self._ptr))

    def det(self):
        return float(_lib.tensor_det(self._ptr))

    def inverse(self):
        return Tensor(_ptr=_lib.tensor_inverse(self._ptr))

    def tril(self, diagonal=0):
        return Tensor(_ptr=_lib.tensor_tril(self._ptr, diagonal))

    def triu(self, diagonal=0):
        return Tensor(_ptr=_lib.tensor_triu(self._ptr, diagonal))

    def diag(self, diagonal=0):
        return Tensor(_ptr=_lib.tensor_diag(self._ptr, diagonal))

    def cholesky(self):
        return Tensor(_ptr=_lib.tensor_cholesky(self._ptr))

    def matrix_power(self, n):
        return Tensor(_ptr=_lib.tensor_matrix_power(self._ptr, n))

    def bmm(self, other):
        return Tensor(_ptr=_lib.tensor_bmm(self._ptr, other._ptr))

    # --- Manipulation ---
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        c_shape = (ctypes.c_uint32 * len(shape))(*shape)
        return Tensor(_ptr=_lib.tensor_reshape(self._ptr, len(shape), c_shape))

    def squeeze(self):
        return Tensor(_ptr=_lib.tensor_squeeze(self._ptr))

    def unsqueeze(self, dim):
        return Tensor(_ptr=_lib.tensor_unsqueeze(self._ptr, dim))

    def flatten(self, start_dim=0, end_dim=-1):
        return Tensor(_ptr=_lib.tensor_flatten(self._ptr, start_dim, end_dim))

    @property
    def T(self):
        return Tensor(_ptr=_lib.tensor_t_op(self._ptr))

    def flip(self, dims):
        if isinstance(dims, int):
            dims = [dims]
        c_dims = (ctypes.c_int32 * len(dims))(*dims)
        return Tensor(_ptr=_lib.tensor_flip(self._ptr, c_dims, len(dims)))

    def roll(self, shift, dim=0):
        return Tensor(_ptr=_lib.tensor_roll(self._ptr, shift, dim))

    def narrow(self, dim, start, length):
        return Tensor(_ptr=_lib.tensor_narrow(self._ptr, dim, start, length))

    def index_select(self, dim, indices):
        c_idx = (ctypes.c_uint32 * len(indices))(*indices)
        return Tensor(_ptr=_lib.tensor_index_select(self._ptr, dim, c_idx, len(indices)))

    def tile(self, reps):
        if isinstance(reps, int):
            reps = [reps]
        c_reps = (ctypes.c_uint32 * len(reps))(*reps)
        return Tensor(_ptr=_lib.tensor_tile(self._ptr, c_reps, len(reps)))

    # --- Additional Math ---
    def tan(self):
        return Tensor(_ptr=_lib.tensor_tan(self._ptr))

    def asin(self):
        return Tensor(_ptr=_lib.tensor_asin(self._ptr))

    def acos(self):
        return Tensor(_ptr=_lib.tensor_acos(self._ptr))

    def atan(self):
        return Tensor(_ptr=_lib.tensor_atan(self._ptr))

    def sinh(self):
        return Tensor(_ptr=_lib.tensor_sinh(self._ptr))

    def cosh(self):
        return Tensor(_ptr=_lib.tensor_cosh(self._ptr))

    def exp2(self):
        return Tensor(_ptr=_lib.tensor_exp2(self._ptr))

    def expm1(self):
        return Tensor(_ptr=_lib.tensor_expm1(self._ptr))

    def log2(self):
        return Tensor(_ptr=_lib.tensor_log2(self._ptr))

    def log10(self):
        return Tensor(_ptr=_lib.tensor_log10(self._ptr))

    def log1p(self):
        return Tensor(_ptr=_lib.tensor_log1p(self._ptr))

    def frac(self):
        return Tensor(_ptr=_lib.tensor_frac(self._ptr))

    def cbrt(self):
        return Tensor(_ptr=_lib.tensor_cbrt(self._ptr))

    def erf(self):
        return Tensor(_ptr=_lib.tensor_erf(self._ptr))

    def erfc(self):
        return Tensor(_ptr=_lib.tensor_erfc(self._ptr))

    def nan_to_num(self, nan=0.0, posinf=1e10, neginf=-1e10):
        return Tensor(_ptr=_lib.tensor_nan_to_num(self._ptr, nan, posinf, neginf))

    def lerp(self, other, weight):
        return Tensor(_ptr=_lib.tensor_lerp(self._ptr, other._ptr, weight))

    def trunc(self):
        return Tensor(_ptr=_lib.tensor_trunc_op(self._ptr))

    # --- Additional Reductions ---
    def prod(self):
        return float(_lib.tensor_prod_all(self._ptr))

    def median(self):
        return float(_lib.tensor_median_all(self._ptr))

    def count_nonzero(self):
        return int(_lib.tensor_count_nonzero_all(self._ptr))

    def cumsum(self, dim=0):
        return Tensor(_ptr=_lib.tensor_cumsum(self._ptr, dim))

    def diff(self, dim=0):
        return Tensor(_ptr=_lib.tensor_diff(self._ptr, dim))

    def all(self):
        return bool(_lib.tensor_all(self._ptr))

    def any(self):
        return bool(_lib.tensor_any(self._ptr))

    def nansum(self):
        return float(_lib.tensor_nansum_all(self._ptr))

    def nanmean(self):
        return float(_lib.tensor_nanmean_all(self._ptr))

    def dist(self, other, p=2.0):
        return float(_lib.tensor_dist(self._ptr, other._ptr, p))

    def argmax_dim(self, dim):
        return Tensor(_ptr=_lib.tensor_argmax_dim(self._ptr, dim))

    # --- Additional Compare ---
    def logical_and(self, other):
        return Tensor(_ptr=_lib.tensor_logical_and(self._ptr, other._ptr))

    def logical_or(self, other):
        return Tensor(_ptr=_lib.tensor_logical_or(self._ptr, other._ptr))

    def logical_not(self):
        return Tensor(_ptr=_lib.tensor_logical_not(self._ptr))

    def isnan(self):
        return Tensor(_ptr=_lib.tensor_isnan_op(self._ptr))

    def isinf(self):
        return Tensor(_ptr=_lib.tensor_isinf_op(self._ptr))

    def isfinite(self):
        return Tensor(_ptr=_lib.tensor_isfinite_op(self._ptr))

    def maximum(self, other):
        return Tensor(_ptr=_lib.tensor_maximum(self._ptr, other._ptr))

    def minimum(self, other):
        return Tensor(_ptr=_lib.tensor_minimum(self._ptr, other._ptr))

    def sort(self, dim=0, descending=False):
        idx_ptr = ctypes.POINTER(RTensor)()
        sorted_ptr = _lib.tensor_sort_op(self._ptr, dim, descending, ctypes.byref(idx_ptr))
        sorted_t = Tensor(_ptr=sorted_ptr)
        idx_t = Tensor(_ptr=idx_ptr)
        return sorted_t, idx_t

    def equal(self, other):
        return bool(_lib.tensor_equal(self._ptr, other._ptr))

    def allclose(self, other, rtol=1e-5, atol=1e-8):
        return bool(_lib.tensor_allclose(self._ptr, other._ptr, rtol, atol))

    def unique(self):
        count = ctypes.c_uint32()
        ptr = _lib.tensor_unique(self._ptr, ctypes.byref(count))
        return Tensor(_ptr=ptr), int(count.value)

    # --- FFT ---
    def fft(self):
        return Tensor(_ptr=_lib.tensor_fft(self._ptr))

    def ifft(self):
        return Tensor(_ptr=_lib.tensor_ifft(self._ptr))

    # --- Utility ---
    def fill_(self, value):
        _lib.tensor_fill(self._ptr, value)
        return self

    def randomize_(self):
        _lib.tensor_randomize(self._ptr)
        return self

    def numel(self):
        return int(_lib.tensor_numel(self._ptr))

    def is_floating_point(self):
        return bool(_lib.tensor_is_floating_point(self._ptr))

    def numpy(self):
        """Return a copy as numpy array."""
        return self.data.copy()

    def __repr__(self):
        return f"rpl.Tensor({self.data}, requires_grad={self._ptr.contents.requires_grad})"

    def __len__(self):
        return self.shape[0] if self.ndim > 0 else 1

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return Tensor(_ptr=_lib.tensor_div(self._ptr, other._ptr))
        raise TypeError("Only Tensor divisions supported")

# ============================================================
# Module-level factory functions
# ============================================================

def manual_seed(seed):
    _lib.rpl_manual_seed(seed)

def rand(*shape):
    c_shape = (ctypes.c_uint32 * len(shape))(*shape)
    return Tensor(_ptr=_lib.tensor_rand(len(shape), c_shape))

def randn(*shape):
    c_shape = (ctypes.c_uint32 * len(shape))(*shape)
    return Tensor(_ptr=_lib.tensor_randn(len(shape), c_shape))

def zeros(*shape):
    c_shape = (ctypes.c_uint32 * len(shape))(*shape)
    return Tensor(_ptr=_lib.tensor_zeros(len(shape), c_shape))

def ones(*shape):
    c_shape = (ctypes.c_uint32 * len(shape))(*shape)
    return Tensor(_ptr=_lib.tensor_ones(len(shape), c_shape))

def arange(start, end, step=1.0):
    return Tensor(_ptr=_lib.tensor_arange(start, end, step))

def linspace(start, end, steps):
    return Tensor(_ptr=_lib.tensor_linspace(start, end, steps))

def randperm(n):
    return Tensor(_ptr=_lib.tensor_randperm(n))

def eye(n):
    return Tensor(_ptr=_lib.tensor_eye(n))

def cat(tensors, dim=0):
    arr = (ctypes.POINTER(RTensor) * len(tensors))(*[t._ptr for t in tensors])
    return Tensor(_ptr=_lib.tensor_cat(arr, len(tensors), dim))

def stack(tensors, dim=0):
    arr = (ctypes.POINTER(RTensor) * len(tensors))(*[t._ptr for t in tensors])
    return Tensor(_ptr=_lib.tensor_stack(arr, len(tensors), dim))

def where(cond, x, y):
    return Tensor(_ptr=_lib.tensor_where_cond(cond._ptr, x._ptr, y._ptr))

def hann_window(size):
    return Tensor(_ptr=_lib.tensor_hann_window(size))

def hamming_window(size):
    return Tensor(_ptr=_lib.tensor_hamming_window(size))


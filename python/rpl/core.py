import ctypes
import os
import numpy as np

# Load the shared library
lib_path = os.path.join(os.path.dirname(__file__), "../../build/librpl.so")
try:
    _lib = ctypes.CDLL(lib_path)
except OSError:
    # Try local search if path above fails
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
    ]

# Function prototypes
_lib.tensor_create.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.c_bool]
_lib.tensor_create.restype = ctypes.POINTER(RTensor)

_lib.tensor_free.argtypes = [ctypes.POINTER(RTensor)]
_lib.tensor_free.restype = None

_lib.tensor_fill.argtypes = [ctypes.POINTER(RTensor), ctypes.c_float]
_lib.tensor_fill.restype = None

_lib.tensor_matmul.argtypes = [ctypes.POINTER(RTensor), ctypes.POINTER(RTensor)]
_lib.tensor_matmul.restype = ctypes.POINTER(RTensor)

_lib.tensor_add_out.argtypes = [ctypes.POINTER(RTensor), ctypes.POINTER(RTensor), ctypes.POINTER(RTensor)]
_lib.tensor_add_out.restype = None

_lib.tensor_add.argtypes = [ctypes.POINTER(RTensor), ctypes.POINTER(RTensor)]
_lib.tensor_add.restype = ctypes.POINTER(RTensor)

_lib.tensor_mul_out.argtypes = [ctypes.POINTER(RTensor), ctypes.POINTER(RTensor), ctypes.POINTER(RTensor)]
_lib.tensor_mul_out.restype = None

_lib.tensor_mul.argtypes = [ctypes.POINTER(RTensor), ctypes.POINTER(RTensor)]
_lib.tensor_mul.restype = ctypes.POINTER(RTensor)

_lib.tensor_backward.argtypes = [ctypes.POINTER(RTensor)]
_lib.tensor_backward.restype = None

_lib.tensor_zero_grad.argtypes = [ctypes.POINTER(RTensor)]
_lib.tensor_zero_grad.restype = None

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
            # Copy data
            ctypes.memmove(self._ptr.contents.data, data.ctypes.data, data.nbytes)
        elif shape is not None:
            c_shape = (ctypes.c_uint32 * len(shape))(*shape)
            self._ptr = _lib.tensor_create(len(shape), c_shape, requires_grad)
            self._owns_ptr = True
        else:
            raise ValueError("Must provide data or shape")

    def __del__(self):
        if getattr(self, "_owns_ptr", False) and self._ptr:
            try:
                _lib.tensor_free(self._ptr)
                self._ptr = None
            except:
                pass

    @property
    def shape(self):
        return tuple(self._ptr.contents.shape[:self._ptr.contents.dims])

    @property
    def data(self):
        # Convert to numpy array without copying
        size = self._ptr.contents.size
        buffer = ctypes.cast(self._ptr.contents.data, ctypes.POINTER(ctypes.c_float * size))
        return np.frombuffer(buffer.contents, dtype=np.float32).reshape(self.shape)

    @property
    def grad(self):
        if not self._ptr.contents.grad:
            return None
        size = self._ptr.contents.size
        buffer = ctypes.cast(self._ptr.contents.grad, ctypes.POINTER(ctypes.c_float * size))
        return np.frombuffer(buffer.contents, dtype=np.float32).reshape(self.shape)

    def backward(self):
        _lib.tensor_backward(self._ptr)

    def zero_grad(self):
        _lib.tensor_zero_grad(self._ptr)

    def __add__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Only Tensor additions supported")
        out_ptr = _lib.tensor_add(self._ptr, other._ptr)
        return Tensor(_ptr=out_ptr)

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Only Tensor matmul supported")
        out_ptr = _lib.tensor_matmul(self._ptr, other._ptr)
        return Tensor(_ptr=out_ptr)

    def __repr__(self):
        return f"rpl.Tensor({self.data}, requires_grad={self._ptr.contents.requires_grad})"


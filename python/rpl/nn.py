import ctypes
from .core import _lib, Tensor, RTensor

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

# Activations
_lib.tensor_relu.argtypes = [ctypes.POINTER(RTensor)]
_lib.tensor_relu.restype = ctypes.POINTER(RTensor)

_lib.tensor_sigmoid.argtypes = [ctypes.POINTER(RTensor)]
_lib.tensor_sigmoid.restype = ctypes.POINTER(RTensor)

_lib.tensor_relu_inplace.argtypes = [ctypes.POINTER(RTensor)]
_lib.tensor_sigmoid_inplace.argtypes = [ctypes.POINTER(RTensor)]

class ReLU:
    def __call__(self, x):
        out_ptr = _lib.tensor_relu(x._ptr)
        return Tensor(_ptr=out_ptr)

class Sigmoid:
    def __call__(self, x):
        out_ptr = _lib.tensor_sigmoid(x._ptr)
        return Tensor(_ptr=out_ptr)

class Tanh:
    def __call__(self, x):
        _lib.tensor_tanh_inplace(x._ptr)
        return x

# Loss Functions
_lib.tensor_mse_loss.argtypes = [ctypes.POINTER(RTensor), ctypes.POINTER(RTensor)]
_lib.tensor_mse_loss.restype = ctypes.POINTER(RTensor)

_lib.mse_loss.argtypes = [ctypes.POINTER(RTensor), ctypes.POINTER(RTensor)]
_lib.mse_loss.restype = ctypes.c_float

def mse_loss(input, target):
    out_ptr = _lib.tensor_mse_loss(input._ptr, target._ptr)
    return Tensor(_ptr=out_ptr)

def cross_entropy_loss(input, target):
    return _lib.cross_entropy_loss(input._ptr, target._ptr)

# Add other layers as needed (Conv2d, etc.)

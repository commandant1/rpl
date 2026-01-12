import ctypes
from .core import _lib, RTensor

class ROptimizer(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("learning_rate", ctypes.c_float),
        # ... other members omitted for simplicity or bound via pointers
    ]

_lib.optimizer_sgd_create.argtypes = [
    ctypes.POINTER(ctypes.POINTER(RTensor)), 
    ctypes.c_uint32, 
    ctypes.c_float, 
    ctypes.c_float, 
    ctypes.c_float, 
    ctypes.c_float, 
    ctypes.c_bool
]
_lib.optimizer_sgd_create.restype = ctypes.POINTER(ROptimizer)

_lib.optimizer_step.argtypes = [ctypes.POINTER(ROptimizer)]
_lib.optimizer_step.restype = None

_lib.optimizer_zero_grad.argtypes = [ctypes.POINTER(ROptimizer)]
_lib.optimizer_zero_grad.restype = None

_lib.optimizer_free.argtypes = [ctypes.POINTER(ROptimizer)]
_lib.optimizer_free.restype = None

class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        self._params = params # Keep refs to params
        self._param_ptrs = (ctypes.POINTER(RTensor) * len(params))()
        for i, p in enumerate(params):
            self._param_ptrs[i] = p._ptr
        # parameters, num_params, lr, momentum, dampening, weight_decay, nesterov
        self._ptr = _lib.optimizer_sgd_create(self._param_ptrs, len(params), lr, momentum, 0.0, weight_decay, False)

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            _lib.optimizer_free(self._ptr)
            self._ptr = None

    def step(self):
        _lib.optimizer_step(self._ptr)

    def zero_grad(self):
        _lib.optimizer_zero_grad(self._ptr)

    def __del__(self):
        # _lib.optimizer_free(self._ptr)
        pass

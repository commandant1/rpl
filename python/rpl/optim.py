import ctypes
import sys
from .core import _lib, RTensor

class ROptimizer(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("learning_rate", ctypes.c_float),
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

_lib.optimizer_adam_create.argtypes = [
    ctypes.POINTER(ctypes.POINTER(RTensor)),
    ctypes.c_uint32,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float
]
_lib.optimizer_adam_create.restype = ctypes.POINTER(ROptimizer)

_lib.optimizer_adamw_create.argtypes = [
    ctypes.POINTER(ctypes.POINTER(RTensor)),
    ctypes.c_uint32,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float
]
_lib.optimizer_adamw_create.restype = ctypes.POINTER(ROptimizer)

_lib.optimizer_lamb_create.argtypes = [
    ctypes.POINTER(ctypes.POINTER(RTensor)),
    ctypes.c_uint32,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float
]
_lib.optimizer_lamb_create.restype = ctypes.POINTER(ROptimizer)

_lib.optimizer_step.argtypes = [ctypes.POINTER(ROptimizer)]
_lib.optimizer_step.restype = None

_lib.optimizer_zero_grad.argtypes = [ctypes.POINTER(ROptimizer)]
_lib.optimizer_zero_grad.restype = None

_lib.optimizer_zero_grad_set_to_none.argtypes = [ctypes.POINTER(ROptimizer)]
_lib.optimizer_zero_grad_set_to_none.restype = None

_lib.optimizer_free.argtypes = [ctypes.POINTER(ROptimizer)]
_lib.optimizer_free.restype = None

class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        self._params = list(params) # Convert iterator/generator to list
        self._param_ptrs = (ctypes.POINTER(RTensor) * len(self._params))()
        for i, p in enumerate(self._params):
            self._param_ptrs[i] = p._ptr
        # parameters, num_params, lr, momentum, dampening, weight_decay, nesterov
        self._ptr = _lib.optimizer_sgd_create(self._param_ptrs, len(self._params), lr, momentum, 0.0, weight_decay, False)

    def __del__(self):
        try:
            if sys is None or sys.is_finalizing() or _lib is None:
                return
        except:
            return
        if hasattr(self, "_ptr") and self._ptr:
            try:
                _lib.optimizer_free(self._ptr)
            except:
                pass
            self._ptr = None

    def step(self):
        _lib.optimizer_step(self._ptr)

    def zero_grad(self, set_to_none=False):
        if set_to_none:
            _lib.optimizer_zero_grad_set_to_none(self._ptr)
        else:
            _lib.optimizer_zero_grad(self._ptr)


class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self._params = list(params)
        self._param_ptrs = (ctypes.POINTER(RTensor) * len(self._params))()
        for i, p in enumerate(self._params):
            self._param_ptrs[i] = p._ptr
        self._ptr = _lib.optimizer_adam_create(self._param_ptrs, len(self._params), lr, betas[0], betas[1], eps, weight_decay)

    def __del__(self):
        try:
            if sys is None or sys.is_finalizing() or _lib is None:
                return
        except:
            return
        if hasattr(self, "_ptr") and self._ptr:
            try:
                _lib.optimizer_free(self._ptr)
            except:
                pass
            self._ptr = None

    def step(self):
        _lib.optimizer_step(self._ptr)

    def zero_grad(self, set_to_none=False):
        if set_to_none:
            _lib.optimizer_zero_grad_set_to_none(self._ptr)
        else:
            _lib.optimizer_zero_grad(self._ptr)


class AdamW:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self._params = list(params)
        self._param_ptrs = (ctypes.POINTER(RTensor) * len(self._params))()
        for i, p in enumerate(self._params):
            self._param_ptrs[i] = p._ptr
        self._ptr = _lib.optimizer_adamw_create(self._param_ptrs, len(self._params), lr, betas[0], betas[1], eps, weight_decay)

    def __del__(self):
        try:
            if sys is None or sys.is_finalizing() or _lib is None:
                return
        except:
            return
        if hasattr(self, "_ptr") and self._ptr:
            try:
                _lib.optimizer_free(self._ptr)
            except:
                pass
            self._ptr = None

    def step(self):
        _lib.optimizer_step(self._ptr)

    def zero_grad(self, set_to_none=False):
        if set_to_none:
            _lib.optimizer_zero_grad_set_to_none(self._ptr)
        else:
            _lib.optimizer_zero_grad(self._ptr)


class LAMB:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01):
        self._params = list(params)
        self._param_ptrs = (ctypes.POINTER(RTensor) * len(self._params))()
        for i, p in enumerate(self._params):
            self._param_ptrs[i] = p._ptr
        self._ptr = _lib.optimizer_lamb_create(self._param_ptrs, len(self._params), lr, betas[0], betas[1], eps, weight_decay)

    def __del__(self):
        try:
            if sys is None or sys.is_finalizing() or _lib is None:
                return
        except:
            return
        if hasattr(self, "_ptr") and self._ptr:
            try:
                _lib.optimizer_free(self._ptr)
            except:
                pass
            self._ptr = None

    def step(self):
        _lib.optimizer_step(self._ptr)

    def zero_grad(self, set_to_none=False):
        if set_to_none:
            _lib.optimizer_zero_grad_set_to_none(self._ptr)
        else:
            _lib.optimizer_zero_grad(self._ptr)

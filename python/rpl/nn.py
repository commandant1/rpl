import ctypes
import sys
import numpy as np
from .core import _lib, Tensor, RTensor

# ============================================================
# Module Base Class
# ============================================================

class Module:
    def __init__(self):
        self.training = True

    def parameters(self):
        params = []
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor):
                if value._ptr.contents.requires_grad:
                    params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Tensor):
                        if item._ptr.contents.requires_grad:
                            params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def train(self, mode=True):
        self.training = mode
        for name, value in self.__dict__.items():
            if isinstance(value, Module):
                value.train(mode)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        item.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

# ============================================================
# Sequential Module
# ============================================================

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)
        for i, layer in enumerate(self.layers):
            setattr(self, f"layer_{i}", layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
            elif isinstance(layer, Tensor) and layer._ptr.contents.requires_grad:
                params.append(layer)
        return params

    def train(self, mode=True):
        self.training = mode
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train(mode)
        return self

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

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self._ptr = _lib.linear_create(in_features, out_features)
        self.weight = Tensor(_ptr=self._ptr.contents.weight, _borrow=True)
        self.bias = Tensor(_ptr=self._ptr.contents.bias, _borrow=True)
        self.in_features = in_features
        self.out_features = out_features

    def __del__(self):
        try:
            if sys is None or sys.is_finalizing() or _lib is None:
                return
        except:
            return
        if hasattr(self, "_ptr") and self._ptr:
            try:
                _lib.linear_free(self._ptr)
            except:
                pass
            self._ptr = None

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        out_ptr = _lib.linear_forward(self._ptr, x._ptr)
        return Tensor(_ptr=out_ptr)

# ============================================================
# Conv2D Layer
# ============================================================

class RConv2d(ctypes.Structure):
    _fields_ = [
        ("weight", ctypes.POINTER(RTensor)),
        ("bias", ctypes.POINTER(RTensor)),
        ("in_channels", ctypes.c_uint32),
        ("out_channels", ctypes.c_uint32),
        ("kernel_size", ctypes.c_uint32),
        ("stride", ctypes.c_uint32),
        ("padding", ctypes.c_uint32),
        ("dilation", ctypes.c_uint32),
        ("groups", ctypes.c_uint32),
    ]

_lib.conv2d_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
_lib.conv2d_create.restype = ctypes.POINTER(RConv2d)
_lib.conv2d_forward.argtypes = [ctypes.POINTER(RConv2d), ctypes.POINTER(RTensor)]
_lib.conv2d_forward.restype = ctypes.POINTER(RTensor)
_lib.conv2d_free.argtypes = [ctypes.POINTER(RConv2d)]
_lib.conv2d_free.restype = None

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self._ptr = _lib.conv2d_create(in_channels, out_channels, kernel_size, stride, padding)
        self.weight = Tensor(_ptr=self._ptr.contents.weight, _borrow=True)
        self.bias = Tensor(_ptr=self._ptr.contents.bias, _borrow=True)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __del__(self):
        try:
            if sys is None or sys.is_finalizing() or _lib is None:
                return
        except:
            return
        if hasattr(self, "_ptr") and self._ptr:
            try:
                _lib.conv2d_free(self._ptr)
            except:
                pass
            self._ptr = None

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        out_ptr = _lib.conv2d_forward(self._ptr, x._ptr)
        return Tensor(_ptr=out_ptr)

# ============================================================
# BatchNorm2D Layer
# ============================================================

class RBatchNorm2d(ctypes.Structure):
    _fields_ = [
        ("num_features", ctypes.c_uint32),
        ("weight", ctypes.POINTER(RTensor)),
        ("bias", ctypes.POINTER(RTensor)),
        ("running_mean", ctypes.POINTER(RTensor)),
        ("running_var", ctypes.POINTER(RTensor)),
        ("momentum", ctypes.c_float),
        ("eps", ctypes.c_float),
        ("training", ctypes.c_bool),
        ("num_batches_tracked", ctypes.c_uint32),
    ]

_lib.batchnorm2d_create.argtypes = [ctypes.c_uint32, ctypes.c_float, ctypes.c_float]
_lib.batchnorm2d_create.restype = ctypes.POINTER(RBatchNorm2d)
_lib.batchnorm2d_forward.argtypes = [ctypes.POINTER(RBatchNorm2d), ctypes.POINTER(RTensor)]
_lib.batchnorm2d_forward.restype = ctypes.POINTER(RTensor)
_lib.batchnorm2d_free.argtypes = [ctypes.POINTER(RBatchNorm2d)]
_lib.batchnorm2d_free.restype = None

class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self._ptr = _lib.batchnorm2d_create(num_features, momentum, eps)
        self.weight = Tensor(_ptr=self._ptr.contents.weight, _borrow=True)
        self.bias = Tensor(_ptr=self._ptr.contents.bias, _borrow=True)
        self.running_mean = Tensor(_ptr=self._ptr.contents.running_mean, _borrow=True)
        self.running_var = Tensor(_ptr=self._ptr.contents.running_var, _borrow=True)
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

    def __del__(self):
        try:
            if sys is None or sys.is_finalizing() or _lib is None:
                return
        except:
            return
        if hasattr(self, "_ptr") and self._ptr:
            try:
                _lib.batchnorm2d_free(self._ptr)
            except:
                pass
            self._ptr = None

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        self._ptr.contents.training = self.training
        out_ptr = _lib.batchnorm2d_forward(self._ptr, x._ptr)
        return Tensor(_ptr=out_ptr)

# ============================================================
# LayerNorm Layer
# ============================================================

class RLayerNorm(ctypes.Structure):
    _fields_ = [
        ("normalized_shape", ctypes.c_uint32),
        ("weight", ctypes.POINTER(RTensor)),
        ("bias", ctypes.POINTER(RTensor)),
        ("eps", ctypes.c_float),
    ]

_lib.layer_norm_create.argtypes = [ctypes.c_uint32, ctypes.c_float]
_lib.layer_norm_create.restype = ctypes.POINTER(RLayerNorm)
_lib.layer_norm_forward.argtypes = [ctypes.POINTER(RLayerNorm), ctypes.POINTER(RTensor)]
_lib.layer_norm_forward.restype = ctypes.POINTER(RTensor)
_lib.layer_norm_free.argtypes = [ctypes.POINTER(RLayerNorm)]
_lib.layer_norm_free.restype = None

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, (list, tuple)):
            normalized_shape = normalized_shape[-1]
        self._ptr = _lib.layer_norm_create(normalized_shape, eps)
        self.weight = Tensor(_ptr=self._ptr.contents.weight, _borrow=True)
        self.bias = Tensor(_ptr=self._ptr.contents.bias, _borrow=True)
        self.eps = eps

    def __del__(self):
        try:
            if sys is None or sys.is_finalizing() or _lib is None:
                return
        except:
            return
        if hasattr(self, "_ptr") and self._ptr:
            try:
                _lib.layer_norm_free(self._ptr)
            except:
                pass
            self._ptr = None

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        out_ptr = _lib.layer_norm_forward(self._ptr, x._ptr)
        return Tensor(_ptr=out_ptr)

# ============================================================
# MaxPool2D Layer
# ============================================================

class RMaxPool2d(ctypes.Structure):
    _fields_ = [
        ("kernel_size", ctypes.c_uint32),
        ("stride", ctypes.c_uint32),
        ("padding", ctypes.c_uint32),
    ]

_lib.maxpool2d_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
_lib.maxpool2d_create.restype = ctypes.POINTER(RMaxPool2d)
_lib.maxpool2d_forward.argtypes = [ctypes.POINTER(RMaxPool2d), ctypes.POINTER(RTensor)]
_lib.maxpool2d_forward.restype = ctypes.POINTER(RTensor)
_lib.maxpool2d_free.argtypes = [ctypes.POINTER(RMaxPool2d)]
_lib.maxpool2d_free.restype = None

class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self._ptr = _lib.maxpool2d_create(kernel_size, stride, padding)

    def __del__(self):
        try:
            if sys is None or sys.is_finalizing() or _lib is None:
                return
        except:
            return
        if hasattr(self, "_ptr") and self._ptr:
            try:
                _lib.maxpool2d_free(self._ptr)
            except:
                pass
            self._ptr = None

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        out_ptr = _lib.maxpool2d_forward(self._ptr, x._ptr)
        return Tensor(_ptr=out_ptr)

# ============================================================
# Dropout Layer
# ============================================================

class RDropout(ctypes.Structure):
    _fields_ = [
        ("p", ctypes.c_float),
        ("training", ctypes.c_bool),
    ]

_lib.dropout_create.argtypes = [ctypes.c_float]
_lib.dropout_create.restype = ctypes.POINTER(RDropout)
_lib.dropout_forward.argtypes = [ctypes.POINTER(RDropout), ctypes.POINTER(RTensor)]
_lib.dropout_forward.restype = ctypes.POINTER(RTensor)
_lib.dropout_free.argtypes = [ctypes.POINTER(RDropout)]
_lib.dropout_free.restype = None

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self._ptr = _lib.dropout_create(p)

    def __del__(self):
        try:
            if sys is None or sys.is_finalizing() or _lib is None:
                return
        except:
            return
        if hasattr(self, "_ptr") and self._ptr:
            try:
                _lib.dropout_free(self._ptr)
            except:
                pass
            self._ptr = None

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        self._ptr.contents.training = self.training
        out_ptr = _lib.dropout_forward(self._ptr, x._ptr)
        return Tensor(_ptr=out_ptr)

# ============================================================
# Embedding Layer
# ============================================================

class REmbedding(ctypes.Structure):
    _fields_ = [
        ("num_embeddings", ctypes.c_uint32),
        ("embedding_dim", ctypes.c_uint32),
        ("weight", ctypes.POINTER(RTensor)),
    ]

_lib.embedding_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
_lib.embedding_create.restype = ctypes.POINTER(REmbedding)
_lib.embedding_forward.argtypes = [ctypes.POINTER(REmbedding), ctypes.POINTER(RTensor)]
_lib.embedding_forward.restype = ctypes.POINTER(RTensor)
_lib.embedding_free.argtypes = [ctypes.POINTER(REmbedding)]
_lib.embedding_free.restype = None

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self._ptr = _lib.embedding_create(num_embeddings, embedding_dim)
        self.weight = Tensor(_ptr=self._ptr.contents.weight, _borrow=True)

    def __del__(self):
        try:
            if sys is None or sys.is_finalizing() or _lib is None:
                return
        except:
            return
        if hasattr(self, "_ptr") and self._ptr:
            try:
                _lib.embedding_free(self._ptr)
            except:
                pass
            self._ptr = None

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        out_ptr = _lib.embedding_forward(self._ptr, x._ptr)
        out = Tensor(_ptr=out_ptr)
        if x.ndim > 1:
            out_shape = list(x.shape) + [self._ptr.contents.embedding_dim]
            out = out.reshape(*out_shape)
        return out

# ============================================================
# Flatten Layer
# ============================================================

class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        return x.flatten(self.start_dim, self.end_dim)

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

class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return mse_loss(input, target)

class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        log_softmax = input.log_softmax()
        loss = - (target * log_softmax).sum(dim=1).sum(dim=0) / float(input.shape[0])
        return loss

# ============================================================
# Activation Modules
# ============================================================

class ReLU(Module):
    def forward(self, x):
        return x.relu()

class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()

class Tanh(Module):
    def forward(self, x):
        return x.tanh()

class GELU(Module):
    def forward(self, x):
        return x.gelu()

class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope
    def forward(self, x):
        return x.leaky_relu(self.negative_slope)

class ELU(Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        return x.elu(self.alpha)

class SELU(Module):
    def forward(self, x):
        return x.selu()

class Swish(Module):
    def forward(self, x):
        return x.swish()

SiLU = Swish

class Mish(Module):
    def forward(self, x):
        return x.mish()

class Hardswish(Module):
    def forward(self, x):
        return x.hardswish()

class Hardsigmoid(Module):
    def forward(self, x):
        return x.hardsigmoid()

class Hardtanh(Module):
    def __init__(self, min_val=-1.0, max_val=1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
    def forward(self, x):
        return x.hardtanh(self.min_val, self.max_val)

class CELU(Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        return x.celu(self.alpha)

class Softplus(Module):
    def __init__(self, beta=1.0, threshold=20.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
    def forward(self, x):
        return x.softplus(self.beta, self.threshold)

class Softsign(Module):
    def forward(self, x):
        return x.softsign()

class LogSoftmax(Module):
    def forward(self, x):
        return x.log_softmax()

class Softmax(Module):
    def forward(self, x):
        return x.softmax()

class PReLU(Module):
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.weight = Tensor([init] * num_parameters)
    def forward(self, x):
        out = x._make_like()
        _lib.tensor_prelu(out, x._ptr, self.weight._ptr)
        return Tensor(_ptr=out)

class RReLU(Module):
    def __init__(self, lower=1.0/8, upper=1.0/3):
        super().__init__()
        self.lower = lower
        self.upper = upper
    def forward(self, x):
        return x.rrelu(self.lower, self.upper)

class Threshold(Module):
    def __init__(self, threshold, value):
        super().__init__()
        self.threshold_val = threshold
        self.value = value
    def forward(self, x):
        return x.threshold(self.threshold_val, self.value)

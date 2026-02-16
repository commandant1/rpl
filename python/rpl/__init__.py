from .core import (
    Tensor,
    manual_seed, rand, randn, zeros, ones, arange, linspace, randperm,
    eye, cat, stack, where, hann_window, hamming_window,
)
from . import nn
from . import optim
from . import data
try:
    from . import sklearn
except ImportError:
    pass

__version__ = "0.1.0"

from .core import Tensor
from . import nn
from . import optim
from . import data
try:
    from . import sklearn
except ImportError:
    pass

__version__ = "0.1.0"

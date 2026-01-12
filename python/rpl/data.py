import ctypes
from .core import _lib, Tensor, RTensor

class RDataset(ctypes.Structure):
    pass # Opaque

class RDataLoader(ctypes.Structure):
    pass # Opaque

_lib.tensor_dataset_create.argtypes = [ctypes.POINTER(RTensor), ctypes.POINTER(RTensor)]
_lib.tensor_dataset_create.restype = ctypes.POINTER(RDataset)

_lib.dataloader_create.argtypes = [ctypes.POINTER(RDataset), ctypes.c_uint32, ctypes.c_bool, ctypes.c_bool, ctypes.c_uint32]
_lib.dataloader_create.restype = ctypes.POINTER(RDataLoader)

_lib.dataloader_free.argtypes = [ctypes.POINTER(RDataLoader)]
_lib.dataloader_free.restype = None

class TensorDataset:
    def __init__(self, data, targets):
        if not isinstance(data, Tensor): data = Tensor(data)
        if not isinstance(targets, Tensor): targets = Tensor(targets)
        self.data = data
        self.targets = targets
        self._ptr = _lib.tensor_dataset_create(data._ptr, targets._ptr)

class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self._ptr = _lib.dataloader_create(dataset._ptr, batch_size, shuffle, False, 0)

    def __del__(self):
        # _lib.dataloader_free(self._ptr)
        pass

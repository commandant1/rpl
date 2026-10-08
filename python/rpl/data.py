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

_lib.dataloader_start.argtypes = [ctypes.POINTER(RDataLoader)]
_lib.dataloader_start.restype = None

_lib.dataloader_next.argtypes = [
    ctypes.POINTER(RDataLoader),
    ctypes.POINTER(ctypes.POINTER(RTensor)),
    ctypes.POINTER(ctypes.POINTER(RTensor))
]
_lib.dataloader_next.restype = ctypes.c_bool

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
        try:
            import sys
            if sys is None or sys.is_finalizing() or _lib is None:
                return
        except:
            return
        if hasattr(self, "_ptr") and self._ptr:
            try:
                _lib.dataloader_free(self._ptr)
            except:
                pass
            self._ptr = None

    def __iter__(self):
        _lib.dataloader_start(self._ptr)
        return self

    def __next__(self):
        batch = ctypes.POINTER(RTensor)()
        labels = ctypes.POINTER(RTensor)()
        if not _lib.dataloader_next(self._ptr, ctypes.byref(batch), ctypes.byref(labels)):
            raise StopIteration
        return Tensor(_ptr=batch), Tensor(_ptr=labels)

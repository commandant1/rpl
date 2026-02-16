import ctypes
import os
import sys

# Mock MAX_DIMS if not imported
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

def print_offsets(cls):
    print(f"Offsets for {cls.__name__}:")
    for field in cls._fields_:
        name = field[0]
        attr = getattr(cls, name)
        print(f"  {name}: {attr.offset}")

if __name__ == "__main__":
    print_offsets(RTensor)

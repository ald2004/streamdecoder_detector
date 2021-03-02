import PytorchNvCodec as pnvc
from tritonclient.utils.cuda_shared_memory import CudaSharedMemoryException, _raise_if_error, _utf8, _raise_error
import numpy as np
import os
from ctypes import *

_ccudashm_lib = "ccudashm" if os.name == 'nt' else 'libcuda_shared.so'
# _ccudashm_path = pkg_resources.resource_filename(
#     'tritonclient.utils.cuda_shared_memory', _ccudashm_lib)

_ccudashm = cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.abspath(__file__)), _ccudashm_lib))

_ccudashm_shared_memory_region_set = _ccudashm.CudaSharedMemoryRegionSet
_ccudashm_shared_memory_region_set.restype = c_int
_ccudashm_shared_memory_region_set.argtypes = [
    c_void_p, c_uint64, c_uint64, c_void_p
]

_ccudashm_shared_memory_region_set_from_decoder = _ccudashm.CudaSharedMemoryRegionSet_from_decoder
_ccudashm_shared_memory_region_set_from_decoder.restype = c_int
_ccudashm_shared_memory_region_set_from_decoder.argtypes = [
    c_void_p, c_void_p, c_uint64, c_uint64
]


def pnvc_set_shared_memory_region(cuda_shm_handle, input_values):
    if not isinstance(input_values, (list, tuple)):
        _raise_error("input_values must be specified as a numpy array")
    for input_value in input_values:
        if not isinstance(input_value, (np.ndarray,)):
            _raise_error(
                "input_values must be specified as a list/tuple of numpy arrays"
            )

    offset_current = 0
    for input_value in input_values:
        input_value = np.ascontiguousarray(input_value).flatten()
        byte_size = input_value.size * input_value.itemsize
        _raise_if_error(
            c_int(_ccudashm_shared_memory_region_set(cuda_shm_handle, c_uint64(offset_current), \
                                                     c_uint64(byte_size), input_value.ctypes.data_as(c_void_p))))
        offset_current += byte_size
    return


def get_cuda_memory_from_vpf(cuda_shm_handle, ptr_vpf, byte_size):
    offset_current = 0
    _raise_if_error(
        c_int(
            _ccudashm_shared_memory_region_set_from_decoder(cuda_shm_handle, c_void_p(ptr_vpf), offset_current, byte_size)))
    return

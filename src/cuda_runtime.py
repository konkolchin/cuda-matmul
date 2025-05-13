from ctypes import *
import os

# Load CUDA runtime library
try:
    if os.name == 'nt':  # Windows
        cuda = CDLL('cudart64_12.dll')
    else:  # Linux
        cuda = CDLL('libcudart.so')
except OSError:
    cuda = None

def cudaGetDeviceCount():
    """Get the number of CUDA-capable devices."""
    if cuda is None:
        raise RuntimeError("CUDA runtime library not found")
    
    count = c_int()
    result = cuda.cudaGetDeviceCount(byref(count))
    if result != 0:  # cudaSuccess = 0
        raise RuntimeError(f"CUDA error: {result}")
    return count.value 
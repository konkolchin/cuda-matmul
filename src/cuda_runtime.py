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

# Define CUDA device properties structure
class cudaDeviceProp(Structure):
    _fields_ = [
        ("name", c_char * 256),
        ("totalGlobalMem", c_size_t),
        ("sharedMemPerBlock", c_size_t),
        ("regsPerBlock", c_int),
        ("warpSize", c_int),
        ("maxThreadsPerBlock", c_int),
        ("maxThreadsDim", c_int * 3),
        ("maxGridSize", c_int * 3),
        ("clockRate", c_int),
        ("totalConstMem", c_size_t),
        ("major", c_int),
        ("minor", c_int),
        ("textureAlignment", c_size_t),
        ("deviceOverlap", c_int),
        ("multiProcessorCount", c_int),
        ("kernelExecTimeoutEnabled", c_int),
        ("integrated", c_int),
        ("canMapHostMemory", c_int),
        ("computeMode", c_int),
        ("maxTexture1D", c_int),
        ("maxTexture2D", c_int * 2),
        ("maxTexture3D", c_int * 3),
        ("maxTexture2DArray", c_int * 3),
        ("surfaceAlignment", c_size_t),
        ("concurrentKernels", c_int),
        ("ECCEnabled", c_int),
        ("pciBusID", c_int),
        ("pciDeviceID", c_int),
        ("pciDomainID", c_int),
        ("tccDriver", c_int),
        ("asyncEngineCount", c_int),
        ("unifiedAddressing", c_int),
        ("memoryClockRate", c_int),
        ("memoryBusWidth", c_int),
        ("l2CacheSize", c_int),
        ("maxThreadsPerMultiProcessor", c_int),
        ("streamPrioritiesSupported", c_int),
        ("globalL1CacheSupported", c_int),
        ("localL1CacheSupported", c_int),
        ("sharedMemPerMultiprocessor", c_size_t),
        ("regsPerMultiprocessor", c_int),
        ("managedMemory", c_int),
        ("isMultiGpuBoard", c_int),
        ("multiGpuBoardGroupID", c_int),
        ("hostNativeAtomicSupported", c_int),
        ("singleToDoublePrecisionPerfRatio", c_int),
        ("pageableMemoryAccess", c_int),
        ("concurrentManagedAccess", c_int),
        ("computePreemptionSupported", c_int),
        ("canUseHostPointerForRegisteredMem", c_int),
        ("cooperativeLaunch", c_int),
        ("cooperativeMultiDeviceLaunch", c_int),
        ("sharedMemPerBlockOptin", c_size_t),
        ("pageableMemoryAccessUsesHostPageTables", c_int),
        ("directManagedMemAccessFromHost", c_int),
        ("maxBlocksPerMultiProcessor", c_int),
        ("accessPolicyMaxWindowSize", c_int),
        ("reservedSharedMemPerBlock", c_size_t),
        ("hostRegisterSupported", c_int),
        ("sparseCudaArraySupported", c_int),
        ("hostRegisterReadOnlySupported", c_int),
        ("timelineSemaphoreInteropSupported", c_int),
        ("memoryPoolsSupported", c_int),
        ("gpuDirectRDMASupported", c_int),
        ("gpuDirectRDMAFlushWritesOptions", c_uint),
        ("gpuDirectRDMAWritesOrdering", c_int),
        ("memoryPoolSupportedHandleTypes", c_uint),
        ("deferredMappingCudaArraySupported", c_int),
        ("ipcEventSupported", c_int),
        ("clusterLaunch", c_int),
        ("unifiedFunctionPointers", c_int),
        ("reserved", c_int * 64)
    ]

def cudaGetDeviceCount():
    """Get the number of CUDA-capable devices."""
    if cuda is None:
        raise RuntimeError("CUDA runtime library not found")
    
    count = c_int()
    result = cuda.cudaGetDeviceCount(byref(count))
    if result != 0:  # cudaSuccess = 0
        raise RuntimeError(f"CUDA error: {result}")
    return count.value

def cudaGetDeviceProperties(device):
    """Get properties of a CUDA device."""
    if cuda is None:
        raise RuntimeError("CUDA runtime library not found")
    
    props = cudaDeviceProp()
    result = cuda.cudaGetDeviceProperties(byref(props), device)
    if result != 0:  # cudaSuccess = 0
        raise RuntimeError(f"CUDA error: {result}")
    return props 
#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Host function to launch the kernel
extern "C" void launchVectorAdd(const float* a, const float* b, float* c, int n) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    vectorAdd<<<numBlocks, blockSize>>>(a, b, c, n);
    cudaDeviceSynchronize();
}

} // extern "C" 
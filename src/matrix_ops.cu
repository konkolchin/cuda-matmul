#include "matrix_ops.h"
#include <cuda_runtime.h>
#include <cstring>
#include <stdio.h>

// Define block sizes for shared memory
#define SMALL_BLOCK 16
#define BIG_BLOCK 32

// Kernel for small block size (16x16)
__global__ void matmul_small_block(const float* a, const float* b, float* c,
                                 int m, int k, int n) {
    // Shared memory for tiles
    __shared__ float s_a[SMALL_BLOCK][SMALL_BLOCK];
    __shared__ float s_b[SMALL_BLOCK][SMALL_BLOCK];
    
    // Thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Accumulator for the dot product
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (k + SMALL_BLOCK - 1) / SMALL_BLOCK; t++) {
        // Load tiles into shared memory
        if (row < m && t * SMALL_BLOCK + threadIdx.x < k) {
            s_a[threadIdx.y][threadIdx.x] = a[row * k + t * SMALL_BLOCK + threadIdx.x];
        } else {
            s_a[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < n && t * SMALL_BLOCK + threadIdx.y < k) {
            s_b[threadIdx.y][threadIdx.x] = b[(t * SMALL_BLOCK + threadIdx.y) * n + col];
        } else {
            s_b[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Synchronize to ensure all threads have loaded their data
        __syncthreads();
        
        // Compute partial dot product
        if (row < m && col < n) {
            for (int i = 0; i < SMALL_BLOCK; i++) {
                sum += s_a[threadIdx.y][i] * s_b[i][threadIdx.x];
            }
        }
        
        // Synchronize to ensure all threads have finished using shared memory
        __syncthreads();
    }
    
    // Write result
    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

// Kernel for big block size (32x32)
__global__ void matmul_big_block(const float* a, const float* b, float* c,
                               int m, int k, int n) {
    // Shared memory for tiles
    __shared__ float s_a[BIG_BLOCK][BIG_BLOCK];
    __shared__ float s_b[BIG_BLOCK][BIG_BLOCK];
    
    // Thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Accumulator for the dot product
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (k + BIG_BLOCK - 1) / BIG_BLOCK; t++) {
        // Load tiles into shared memory
        if (row < m && t * BIG_BLOCK + threadIdx.x < k) {
            s_a[threadIdx.y][threadIdx.x] = a[row * k + t * BIG_BLOCK + threadIdx.x];
        } else {
            s_a[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < n && t * BIG_BLOCK + threadIdx.y < k) {
            s_b[threadIdx.y][threadIdx.x] = b[(t * BIG_BLOCK + threadIdx.y) * n + col];
        } else {
            s_b[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Synchronize to ensure all threads have loaded their data
        __syncthreads();
        
        // Compute partial dot product
        if (row < m && col < n) {
            for (int i = 0; i < BIG_BLOCK; i++) {
                sum += s_a[threadIdx.y][i] * s_b[i][threadIdx.x];
            }
        }
        
        // Synchronize to ensure all threads have finished using shared memory
        __syncthreads();
    }
    
    // Write result
    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

void matrix_multiply_gpu(const float* a, const float* b, float* c,
                        int m, int k, int n, int block_size) {
    float *d_a, *d_b, *d_c;
    
    // Allocate device memory
    cudaMalloc(&d_a, m * k * sizeof(float));
    cudaMalloc(&d_b, k * n * sizeof(float));
    cudaMalloc(&d_c, m * n * sizeof(float));
    
    // Copy input data to device
    cudaMemcpy(d_a, a, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, k * n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Initialize output matrix with zeros
    cudaMemset(d_c, 0, m * n * sizeof(float));
    
    // Define block and grid dimensions based on block size
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
                 (m + blockDim.y - 1) / blockDim.y);
    
    // Launch appropriate kernel based on block size
    if (block_size == SMALL_BLOCK) {
        matmul_small_block<<<gridDim, blockDim>>>(d_a, d_b, d_c, m, k, n);
    } else if (block_size == BIG_BLOCK) {
        matmul_big_block<<<gridDim, blockDim>>>(d_a, d_b, d_c, m, k, n);
    }
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
    
    // Copy result back to host
    cudaMemcpy(c, d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

extern "C" void matrix_multiply(const float* a, const float* b, float* c,
                              int m, int k, int n, bool useGpu, int block_size) {
    if (useGpu) {
        matrix_multiply_gpu(a, b, c, m, k, n, block_size);
    } else {
        matrix_multiply_cpu(a, b, c, m, k, n);
    }
}
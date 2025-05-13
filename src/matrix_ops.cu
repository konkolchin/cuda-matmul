#include "matrix_ops.h"
#include <cuda_runtime.h>
#include <cstring>
#include <stdio.h>

// Define tile size for shared memory
#define TILE_SIZE 16

__global__ void matrix_multiply_kernel(const float* a, const float* b, float* c,
                                     int m, int k, int n) {
    // Shared memory for tiles
    __shared__ float s_a[TILE_SIZE][TILE_SIZE];
    __shared__ float s_b[TILE_SIZE][TILE_SIZE];
    
    // Thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Accumulator for the dot product
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < m && t * TILE_SIZE + threadIdx.x < k) {
            s_a[threadIdx.y][threadIdx.x] = a[row * k + t * TILE_SIZE + threadIdx.x];
        } else {
            s_a[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < n && t * TILE_SIZE + threadIdx.y < k) {
            s_b[threadIdx.y][threadIdx.x] = b[(t * TILE_SIZE + threadIdx.y) * n + col];
        } else {
            s_b[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Synchronize to ensure all threads have loaded their data
        __syncthreads();
        
        // Compute partial dot product
        if (row < m && col < n) {
            for (int i = 0; i < TILE_SIZE; i++) {
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
                        int m, int k, int n) {
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
    
    // Define block and grid dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
                 (m + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    matrix_multiply_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, m, k, n);
    
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
                              int m, int k, int n, bool useGpu) {
    if (useGpu) {
        matrix_multiply_gpu(a, b, c, m, k, n);
    } else {
        matrix_multiply_cpu(a, b, c, m, k, n);
    }
}
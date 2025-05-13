#pragma once

#include <vector>

extern "C" {
    // CPU implementation of matrix multiplication
    void matrix_multiply_cpu(const float* a, const float* b, float* c, 
                            int m, int k, int n);

    // Matrix multiplication function that can work on both CPU and GPU
    void matrix_multiply(const float* a, const float* b, float* c, 
                        int m, int k, int n, bool useGpu);
}
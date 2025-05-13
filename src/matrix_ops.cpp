#include "matrix_ops.h"
#include <cstring>

void matrix_multiply_cpu(const float* a, const float* b, float* c, 
                        int m, int k, int n) {
    // Initialize output matrix with zeros
    std::memset(c, 0, m * n * sizeof(float));
    
    // Perform matrix multiplication
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
} 
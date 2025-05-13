#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "matrix_ops.h"

namespace py = pybind11;

py::array_t<float> matrix_multiply_wrapper(py::array_t<float> a, 
                                         py::array_t<float> b,
                                         bool useGpu) {
    // Get input arrays
    py::buffer_info a_buf = a.request();
    py::buffer_info b_buf = b.request();
    
    // Check input dimensions
    if (a_buf.ndim != 2 || b_buf.ndim != 2) {
        throw std::runtime_error("Input arrays must be 2D");
    }
    
    int m = a_buf.shape[0];
    int k = a_buf.shape[1];
    int n = b_buf.shape[1];
    
    if (k != b_buf.shape[0]) {
        throw std::runtime_error("Matrix dimensions do not match for multiplication");
    }
    
    // Create output array
    auto result = py::array_t<float>({m, n});
    py::buffer_info result_buf = result.request();
    
    // Get raw pointers
    float* a_ptr = static_cast<float*>(a_buf.ptr);
    float* b_ptr = static_cast<float*>(b_buf.ptr);
    float* result_ptr = static_cast<float*>(result_buf.ptr);
    
    // Call the matrix multiplication function
    matrix_multiply(a_ptr, b_ptr, result_ptr, m, k, n, useGpu);
    
    return result;
}

PYBIND11_MODULE(cuda_ops, m) {
    m.doc() = "Matrix multiplication module with CPU and GPU support";
    
    m.def("matrix_multiply", &matrix_multiply_wrapper, 
          "Multiply two matrices using CPU or GPU",
          py::arg("a").noconvert(),
          py::arg("b").noconvert(),
          py::arg("use_gpu") = false);
} 
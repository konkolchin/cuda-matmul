import unittest
import numpy as np
import time
import cuda_ops
import cuda_runtime

def has_gpu():
    try:
        cuda_runtime.cudaGetDeviceCount()
        return True
    except:
        return False

class TestMatrixOps(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        np.random.seed(42)
        self.has_gpu = has_gpu()
        
    def test_small_matrices(self):
        # Test with small matrices
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        
        # Expected result
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        
        # Test CPU implementation
        result_cpu = cuda_ops.matrix_multiply(a, b, use_gpu=False)
        np.testing.assert_array_almost_equal(result_cpu, expected)
        
        # Test GPU implementation if available
        if self.has_gpu:
            result_gpu = cuda_ops.matrix_multiply(a, b, use_gpu=True)
            np.testing.assert_array_almost_equal(result_gpu, expected)
        
    def test_large_matrices(self):
        # Test with larger matrices
        m, k, n = 100, 100, 100
        a = np.random.rand(m, k).astype(np.float32)
        b = np.random.rand(k, n).astype(np.float32)
        
        # Expected result using numpy
        expected = np.matmul(a, b)
        
        # Test CPU implementation
        result_cpu = cuda_ops.matrix_multiply(a, b, use_gpu=False)
        np.testing.assert_array_almost_equal(result_cpu, expected, decimal=4)
        
        # Test GPU implementation if available
        if self.has_gpu:
            result_gpu = cuda_ops.matrix_multiply(a, b, use_gpu=True)
            np.testing.assert_array_almost_equal(result_gpu, expected, decimal=4)
        
    def test_invalid_dimensions(self):
        # Test with incompatible matrix dimensions
        a = np.random.rand(2, 3).astype(np.float32)
        b = np.random.rand(4, 2).astype(np.float32)
        
        with self.assertRaises(RuntimeError):
            cuda_ops.matrix_multiply(a, b, use_gpu=False)
            
    def test_performance(self):
        # Test performance with large matrices
        sizes = [(100, 100), (500, 500), (1000, 1000)]
        
        print("\nPerformance Test Results:")
        print("Size\t\tCPU Time (s)\tGPU Time (s)\tSpeedup")
        print("-" * 50)
        
        for m, n in sizes:
            k = m  # Square matrices for simplicity
            a = np.random.rand(m, k).astype(np.float32)
            b = np.random.rand(k, n).astype(np.float32)
            
            # Warm-up runs
            for _ in range(3):
                cuda_ops.matrix_multiply(a, b, use_gpu=False)
                if self.has_gpu:
                    cuda_ops.matrix_multiply(a, b, use_gpu=True)
            
            # CPU timing
            start = time.time()
            for _ in range(5):
                cuda_ops.matrix_multiply(a, b, use_gpu=False)
            cpu_time = (time.time() - start) / 5
            
            # GPU timing if available
            if self.has_gpu:
                start = time.time()
                for _ in range(5):
                    cuda_ops.matrix_multiply(a, b, use_gpu=True)
                gpu_time = (time.time() - start) / 5
                speedup = cpu_time / gpu_time
                print(f"{m}x{n}\t\t{cpu_time:.4f}\t\t{gpu_time:.4f}\t\t{speedup:.2f}x")
            else:
                print(f"{m}x{n}\t\t{cpu_time:.4f}\t\tN/A\t\tN/A")

if __name__ == '__main__':
    unittest.main() 
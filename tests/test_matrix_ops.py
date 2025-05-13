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
        self.block_sizes = [16, 32]  # Define block sizes to test
        
    def test_small_matrices(self):
        # Test with small matrices
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        
        # Expected result
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        
        # Test CPU implementation
        result_cpu = cuda_ops.matrix_multiply(a, b, use_gpu=False)
        np.testing.assert_array_almost_equal(result_cpu, expected)
        
        # Test GPU implementation with both block sizes if available
        if self.has_gpu:
            for block_size in self.block_sizes:
                result_gpu = cuda_ops.matrix_multiply(a, b, use_gpu=True, block_size=block_size)
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
        
        # Test GPU implementation with both block sizes if available
        if self.has_gpu:
            for block_size in self.block_sizes:
                result_gpu = cuda_ops.matrix_multiply(a, b, use_gpu=True, block_size=block_size)
                np.testing.assert_array_almost_equal(result_gpu, expected, decimal=4)
    
    def test_very_large_matrices(self):
        # Test with very large matrices (2048x2048)
        m, k, n = 2048, 2048, 2048
        a = np.random.rand(m, k).astype(np.float32)
        b = np.random.rand(k, n).astype(np.float32)
        
        # Expected result using numpy
        expected = np.matmul(a, b)
        
        # Test GPU implementation with both block sizes if available
        if self.has_gpu:
            for block_size in self.block_sizes:
                result_gpu = cuda_ops.matrix_multiply(a, b, use_gpu=True, block_size=block_size)
                np.testing.assert_array_almost_equal(result_gpu, expected, decimal=4)
            
    def test_invalid_dimensions(self):
        # Test with incompatible matrix dimensions
        a = np.random.rand(2, 3).astype(np.float32)
        b = np.random.rand(4, 2).astype(np.float32)
        
        with self.assertRaises(RuntimeError):
            cuda_ops.matrix_multiply(a, b, use_gpu=False)
            
        if self.has_gpu:
            for block_size in self.block_sizes:
                with self.assertRaises(RuntimeError):
                    cuda_ops.matrix_multiply(a, b, use_gpu=True, block_size=block_size)
            
    def test_performance(self):
        # Test performance with large matrices
        sizes = [(1024, 1024), (2048, 2048)]  # Test both sizes
        
        print("\nPerformance Test Results:")
        print("Matrix Size\tCPU Time (s)\tGPU 16x16 (s)\tGPU 32x32 (s)\tSpeedup 16x16\tSpeedup 32x32")
        print("-" * 80)
        
        for m, n in sizes:
            k = m  # Square matrices for simplicity
            a = np.random.rand(m, k).astype(np.float32)
            b = np.random.rand(k, n).astype(np.float32)
            
            # Warm-up runs
            for _ in range(3):
                cuda_ops.matrix_multiply(a, b, use_gpu=False)
                if self.has_gpu:
                    for block_size in self.block_sizes:
                        cuda_ops.matrix_multiply(a, b, use_gpu=True, block_size=block_size)
            
            # CPU timing
            start = time.time()
            for _ in range(5):
                cuda_ops.matrix_multiply(a, b, use_gpu=False)
            cpu_time = (time.time() - start) / 5
            
            # GPU timing if available
            if self.has_gpu:
                gpu_times = []
                speedups = []
                
                for block_size in self.block_sizes:
                    start = time.time()
                    for _ in range(5):
                        cuda_ops.matrix_multiply(a, b, use_gpu=True, block_size=block_size)
                    gpu_time = (time.time() - start) / 5
                    gpu_times.append(gpu_time)
                    speedups.append(cpu_time / gpu_time)
                
                print(f"{m}x{n}\t\t{cpu_time:.4f}\t\t{gpu_times[0]:.4f}\t\t{gpu_times[1]:.4f}\t\t{speedups[0]:.2f}x\t\t{speedups[1]:.2f}x")
            else:
                print(f"{m}x{n}\t\t{cpu_time:.4f}\t\tN/A\t\tN/A\t\tN/A\t\tN/A")

if __name__ == '__main__':
    unittest.main() 
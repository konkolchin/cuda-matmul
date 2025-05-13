import unittest
import numpy as np
import time
import cuda_ops
import cuda_runtime
import pytest
import logging
import os

# Configure logging
logger = logging.getLogger(__name__)

def has_gpu():
    """Check if CUDA is available and log the environment details."""
    try:
        # Log CUDA environment variables
        logger.info(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
        logger.info(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
        
        # Try to get device count
        device_count = cuda_runtime.cudaGetDeviceCount()
        logger.info(f"Found {device_count} CUDA device(s)")
        
        if device_count > 0:
            # Get device properties for the first device
            props = cuda_runtime.cudaGetDeviceProperties(0)
            logger.info(f"Device 0: {props.name.decode() if hasattr(props, 'name') else 'Unknown'}")
            return True
        else:
            logger.warning("No CUDA devices found")
            return False
    except Exception as e:
        logger.error(f"Error checking CUDA availability: {str(e)}")
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
            
    @pytest.mark.performance
    def test_performance(self):
        """Test and report performance of matrix multiplication with different block sizes."""
        # Test performance with large matrices
        sizes = [(1024, 1024), (2048, 2048)]  # Test both sizes
        
        # Create a table header for the results
        header = "Matrix Size\tCPU Time (s)\tGPU 16x16 (s)\tGPU 32x32 (s)\tSpeedup 16x16\tSpeedup 32x32"
        separator = "-" * 80
        
        # Log the header using pytest's logging
        logger.info("\nPerformance Test Results:")
        logger.info(header)
        logger.info(separator)
        
        results = []
        for m, n in sizes:
            k = m  # Square matrices for simplicity
            logger.info(f"\nTesting {m}x{n} matrices...")
            
            # Generate matrices
            logger.info("Generating test matrices...")
            a = np.random.rand(m, k).astype(np.float32)
            b = np.random.rand(k, n).astype(np.float32)
            
            # Warm-up runs (reduced to 2)
            logger.info("Performing warm-up runs...")
            for i in range(2):
                logger.info(f"Warm-up run {i+1}/2")
                cuda_ops.matrix_multiply(a, b, use_gpu=False)
                if self.has_gpu:
                    for block_size in self.block_sizes:
                        cuda_ops.matrix_multiply(a, b, use_gpu=True, block_size=block_size)
            
            # CPU timing (reduced to 3 runs)
            logger.info("Measuring CPU performance...")
            start = time.time()
            for i in range(3):
                logger.info(f"CPU run {i+1}/3")
                cuda_ops.matrix_multiply(a, b, use_gpu=False)
            cpu_time = (time.time() - start) / 3
            logger.info(f"Average CPU time: {cpu_time:.4f}s")
            
            # GPU timing if available
            if self.has_gpu:
                gpu_times = []
                speedups = []
                
                for block_size in self.block_sizes:
                    logger.info(f"Measuring GPU performance with {block_size}x{block_size} blocks...")
                    start = time.time()
                    for i in range(3):  # Reduced to 3 runs
                        logger.info(f"GPU run {i+1}/3")
                        cuda_ops.matrix_multiply(a, b, use_gpu=True, block_size=block_size)
                    gpu_time = (time.time() - start) / 3
                    gpu_times.append(gpu_time)
                    speedups.append(cpu_time / gpu_time)
                    logger.info(f"Average GPU time ({block_size}x{block_size}): {gpu_time:.4f}s")
                
                result = f"{m}x{n}\t\t{cpu_time:.4f}\t\t{gpu_times[0]:.4f}\t\t{gpu_times[1]:.4f}\t\t{speedups[0]:.2f}x\t\t{speedups[1]:.2f}x"
                logger.info(result)
                
                # Store results for potential test assertions
                results.append({
                    'size': f"{m}x{n}",
                    'cpu_time': cpu_time,
                    'gpu_times': gpu_times,
                    'speedups': speedups
                })
                
                # Add test assertions to verify performance
                self.assertGreater(speedups[0], 1.0, f"16x16 block size should be faster than CPU for {m}x{n} matrices")
                self.assertGreater(speedups[1], 1.0, f"32x32 block size should be faster than CPU for {m}x{n} matrices")
            else:
                result = f"{m}x{n}\t\t{cpu_time:.4f}\t\tN/A\t\tN/A\t\tN/A\t\tN/A"
                logger.info(result)
                logger.info("Skipping GPU tests - no GPU available")

if __name__ == '__main__':
    # Configure logging for when running directly
    logging.basicConfig(level=logging.INFO)
    unittest.main() 
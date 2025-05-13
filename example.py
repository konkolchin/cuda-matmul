import numpy as np
import cuda_ops
import time

def run_example():
    print("Matrix Multiplication Example")
    print("============================")
    
    # Create test matrices
    print("\n1. Creating test matrices...")
    a = np.array([[1, 2, 3],
                  [4, 5, 6]], dtype=np.float32)
    b = np.array([[7, 8],
                  [9, 10],
                  [11, 12]], dtype=np.float32)
    
    print("Matrix A:")
    print(a)
    print("\nMatrix B:")
    print(b)
    
    # CPU multiplication
    print("\n2. Running CPU multiplication...")
    start_time = time.time()
    result_cpu = cuda_ops.matrix_multiply(a, b, use_gpu=False)
    cpu_time = time.time() - start_time
    
    print("CPU Result:")
    print(result_cpu)
    print(f"CPU Time: {cpu_time:.6f} seconds")
    
    # GPU multiplication
    print("\n3. Running GPU multiplication...")
    start_time = time.time()
    result_gpu = cuda_ops.matrix_multiply(a, b, use_gpu=True)
    gpu_time = time.time() - start_time
    
    print("GPU Result:")
    print(result_gpu)
    print(f"GPU Time: {gpu_time:.6f} seconds")
    
    # Verify results
    print("\n4. Verifying results...")
    if np.allclose(result_cpu, result_gpu):
        print("✓ Results match!")
    else:
        print("✗ Results don't match!")
    
    # Performance test with larger matrices
    print("\n5. Performance test with larger matrices...")
    sizes = [(100, 100), (500, 500), (1000, 1000)]
    
    print("\nMatrix Size | CPU Time (s) | GPU Time (s) | Speedup")
    print("-" * 55)
    
    for m, n in sizes:
        k = m  # Square matrices for simplicity
        a = np.random.rand(m, k).astype(np.float32)
        b = np.random.rand(k, n).astype(np.float32)
        
        # CPU timing
        start_time = time.time()
        cuda_ops.matrix_multiply(a, b, use_gpu=False)
        cpu_time = time.time() - start_time
        
        # GPU timing
        start_time = time.time()
        cuda_ops.matrix_multiply(a, b, use_gpu=True)
        gpu_time = time.time() - start_time
        
        speedup = cpu_time / gpu_time
        print(f"{m}x{n}\t\t{cpu_time:.4f}\t\t{gpu_time:.4f}\t\t{speedup:.2f}x")

if __name__ == "__main__":
    run_example() 
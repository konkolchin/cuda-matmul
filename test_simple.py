import sys
import os

# Add the build directory to Python path
build_dir = os.path.join(os.path.dirname(__file__), 'build', 'lib')
sys.path.append(build_dir)

import numpy as np
import cuda_ops

# Create small test matrices
a = np.array([[1, 2], [3, 4]], dtype=np.float32)
b = np.array([[5, 6], [7, 8]], dtype=np.float32)

print("Matrix A:")
print(a)
print("\nMatrix B:")
print(b)

# Try CPU multiplication
print("\nTrying CPU multiplication...")
try:
    result = cuda_ops.matrix_multiply(a, b, use_gpu=False)
    print("CPU Result:")
    print(result)
except Exception as e:
    print(f"CPU Error: {e}")

# Try GPU multiplication
print("\nTrying GPU multiplication...")
try:
    result = cuda_ops.matrix_multiply(a, b, use_gpu=True)
    print("GPU Result:")
    print(result)
except Exception as e:
    print(f"GPU Error: {e}") 
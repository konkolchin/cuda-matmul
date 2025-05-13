# CUDA Matrix Operations Library

This package contains the CUDA matrix operations library and its Python bindings.

## Quick Start

1. Install the package:
```bash
pip install -r requirements.txt
pip install -e .
```

2. Use it in your code:
```python
import numpy as np
import cuda_ops

# Create two matrices
a = np.array([[1, 2], [3, 4]], dtype=np.float32)
b = np.array([[5, 6], [7, 8]], dtype=np.float32)

# Multiply them (on CPU or GPU)
result = cuda_ops.matrix_multiply(a, b, use_gpu=True)
print(result)  # [[19 22]
               #  [43 50]]
```

That's it! The library handles all the CUDA setup automatically.

## Contents
- lib/libcuda_ops.so: The main shared library
- src/: Source files for the library
- tests/: Test files demonstrating usage
- build_scripts/: Build scripts for different platforms
- setup.py: Python package setup file
- requirements.txt: Python dependencies

## Building
1. For Linux (using Docker):
   cd build_scripts
   ./build_docker.bat

2. For Windows:
   cd build_scripts
   ./build.bat

## Installation
1. Install Python dependencies:
   pip install -r requirements.txt

2. Install the package:
   pip install -e .

3. Run tests to verify installation:
   python tests/test_matrix_ops.py

## Usage
Here's a simple example of how to use the library:

```python
import numpy as np
import cuda_ops

# Create matrices
a = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=np.float32)
b = np.array([[7, 8],
              [9, 10],
              [11, 12]], dtype=np.float32)

# CPU multiplication
result_cpu = cuda_ops.matrix_multiply(a, b, use_gpu=False)

# GPU multiplication
result_gpu = cuda_ops.matrix_multiply(a, b, use_gpu=True)
```

## Running the Example

To run the example script that demonstrates both CPU and GPU performance:

```bash
python example.py
```

This will:
1. Show a simple matrix multiplication example
2. Compare CPU and GPU results
3. Run performance tests with different matrix sizes

## Performance

The GPU implementation uses shared memory and optimized memory access patterns for better performance. You should see significant speedup for larger matrices.

## Troubleshooting

If you encounter any issues:

1. Make sure CUDA is properly installed
2. Check that your GPU is CUDA-compatible
3. Verify that the NVIDIA drivers are up to date
4. Ensure the library is properly installed with `pip install -e .`

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
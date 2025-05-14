# CUDA Matrix Operations Library

A high-performance CUDA-accelerated matrix operations library with Python bindings. This library provides efficient implementations of common matrix operations using NVIDIA GPUs.

## Features

- Matrix multiplication (GEMM) operations
- Optimized CUDA kernels for GPU acceleration
- Python bindings for easy integration
- Cross-platform support (Linux and Windows)
- Comprehensive test suite

## Requirements

- CUDA Toolkit 11.4 or later (tested with CUDA 11.4.4)
- Python 3.8 or later (tested with Python 3.8.10)
- NVIDIA GPU with compute capability 3.5 or higher
- GCC/G++ compiler (Linux) or Visual Studio (Windows)
- CMake 3.10 or later

### Hardware Requirements
- NVIDIA GPU with at least 2GB of VRAM
- Minimum compute capability: 3.5
- Recommended compute capability: 6.0 or higher for optimal performance

### Software Requirements
- CUDA Toolkit 11.4 or later
  - Driver version must be compatible with the installed CUDA Toolkit
  - For CUDA 11.4, NVIDIA driver version 450.80.02 or later is required
- Python 3.8 or later
  - NumPy 1.19.0 or later
  - PyCUDA 2021.1 or later
- Build tools:
  - Linux: GCC/G++ 7.0 or later
  - Windows: Visual Studio 2019 or later with C++ build tools
  - CMake 3.10 or later

## Installation

### Linux (Ubuntu/Debian)

1. Install CUDA Toolkit:
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/cuda-repo-ubuntu2004-11-4-local_11.4.4-470.82.01-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-4-local_11.4.4-470.82.01-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
```

2. Install build dependencies:
```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential python3-dev
```

3. Clone and build the library:
```bash
git clone https://github.com/konkolchin/cuda-matmul.git
cd cuda-matmul
chmod +x build.sh
./build.sh
```

4. Install Python package:
```bash
pip install -r requirements.txt
pip install -e .
```

### Windows

1. Install CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)
2. Install build dependencies:
   - Install Visual Studio 2019 or later with C++ development tools
   - Install CMake from [cmake.org](https://cmake.org/download/)
   - Add both to your system PATH

3. Clone and build the library:
```powershell
git clone https://github.com/konkolchin/cuda-matmul.git
cd cuda-matmul
.\build.bat
```

4. Install Python package:
```powershell
pip install -r requirements.txt
pip install -e .
```

## Building from Source

### Manual Build (Linux)

If you prefer to build manually instead of using the build script:

```bash
# Create and enter build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
make -j4

# Install Python package
cd ..
pip install -e .
```

### Manual Build (Windows)

If you prefer to build manually instead of using the build script:

```powershell
# Create and enter build directory
mkdir build; cd build

# Configure with CMake
cmake ..

# Build
cmake --build . --config Release

# Install Python package
cd ..
pip install -e .
```

### Build Options

You can customize the build with the following CMake options:

- `-DCMAKE_BUILD_TYPE=Release|Debug` - Set build type (default: Release)
- `-DCUDA_ARCH=75` - Set target CUDA architecture (default: auto-detect)
- `-DPYTHON_EXECUTABLE=/path/to/python` - Specify Python interpreter
- `-DBUILD_TESTS=ON|OFF` - Build test suite (default: ON)

Example:
```bash
cmake -DCMAKE_BUILD_TYPE=Debug -DCUDA_ARCH=75 ..
```

## Usage

```python
import numpy as np
from cuda_matrix_ops import matrix_multiply

# Create input matrices
a = np.random.rand(1000, 1000).astype(np.float32)
b = np.random.rand(1000, 1000).astype(np.float32)

# Perform matrix multiplication
c = matrix_multiply(a, b)

# Verify result
expected = np.matmul(a, b)
print(f"Max difference: {np.max(np.abs(c - expected))}")
```

## Development

### Project Structure

```
cuda-matmul/
├── src/                    # Source files
│   ├── matrix_ops.cu      # CUDA kernels
│   ├── matrix_ops.h       # Header file
│   ├── bindings.cpp       # Python bindings
│   └── cuda_ops.cu        # CUDA operations
├── tests/                  # Test files
│   └── test_matrix_ops.py # Python tests
├── CMakeLists.txt         # CMake configuration
├── build.sh              # Linux build script
├── build.bat             # Windows build script
└── setup.py              # Python package setup
```

### Running Tests

```bash
# From the project root
cd tests
python -m pytest test_matrix_ops.py -v
```

### Building from Source

1. Ensure CUDA Toolkit is installed
2. Run the appropriate build script:
   - Linux: `./build.sh`
   - Windows: `.\build.bat`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NVIDIA for CUDA Toolkit
- PyBind11 for Python bindings
- CMake for build system 
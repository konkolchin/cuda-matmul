#!/bin/bash

# Get the absolute path of the project directory
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create build directory with proper permissions
rm -rf build
mkdir -p build
chmod 755 build

# Configure and build
cd build
cmake "$PROJECT_DIR"
make -j4

# Install Python package
cd "$PROJECT_DIR"
pip install -e .

# Create tests directory if it doesn't exist
mkdir -p tests
chmod 755 tests

# Copy the built library to the tests directory
cp build/lib/libcuda_ops.so tests/
chmod 755 tests/libcuda_ops.so

# Create a symbolic link to the library in the tests directory
cd tests
ln -sf libcuda_ops.so cuda_ops.so

# Return to root directory
cd "$PROJECT_DIR"

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful! The shared library is in build/lib/"
    ls -l build/lib/
else
    echo "Build failed!"
    exit 1
fi 
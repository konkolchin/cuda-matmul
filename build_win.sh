#!/bin/bash

# Create build directory
mkdir -p build
cd build

# Configure with MinGW
cmake -G "MinGW Makefiles" ..

# Build
mingw32-make -j4

# Install Python package
cd ..
pip install -e .

# Create tests directory if it doesn't exist
mkdir -p tests

# Copy the built library to the tests directory
cp build/lib/libcuda_ops.so tests/

# Create a symbolic link to the library in the tests directory
cd tests
ln -sf libcuda_ops.so cuda_ops.so

# Return to root directory
cd ..

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful! The shared library is in build/lib/"
    ls -l build/lib/
else
    echo "Build failed!"
    exit 1
fi 








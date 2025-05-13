@echo off

REM Clean build directory
rmdir /s /q build
mkdir build

REM Build Docker image
docker build -t cuda-so-builder .

REM Create a container and copy the .so file
docker run --rm -v %cd%:/app cuda-so-builder bash -c "cd /app && rm -rf build && mkdir build && cd build && cmake .. && make -j4"

echo Build successful! The shared library should be in build/lib/
dir build\lib\libcuda_ops.so 





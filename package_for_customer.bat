@echo off
echo Creating distribution package for customer...

REM Create distribution directory
set DIST_DIR=cuda_ops_dist
rmdir /s /q %DIST_DIR%
mkdir %DIST_DIR%
mkdir %DIST_DIR%\lib
mkdir %DIST_DIR%\src
mkdir %DIST_DIR%\tests

REM Copy the shared library
copy build\lib\libcuda_ops.so %DIST_DIR%\lib\

REM Copy source files
copy src\*.h %DIST_DIR%\src\
copy src\*.cpp %DIST_DIR%\src\
copy src\*.cu %DIST_DIR%\src\

REM Copy Python files
copy setup.py %DIST_DIR%\
copy requirements.txt %DIST_DIR%\
copy tests\*.py %DIST_DIR%\tests\

REM Create README
echo Creating README...
(
echo CUDA Matrix Operations Library
echo =============================
echo.
echo This package contains the CUDA matrix operations library and its Python bindings.
echo.
echo Contents:
echo - lib/libcuda_ops.so: The main shared library
echo - src/: Source files for the library
echo - tests/: Test files demonstrating usage
echo - setup.py: Python package setup file
echo - requirements.txt: Python dependencies
echo.
echo Installation:
echo 1. Install Python dependencies:
echo    pip install -r requirements.txt
echo.
echo 2. Install the package:
echo    pip install -e .
echo.
echo 3. Run tests to verify installation:
echo    python tests/test_matrix_ops.py
echo.
echo Usage:
echo import cuda_ops
echo.
echo # Example: Matrix multiplication
echo import numpy as np
echo a = np.array([[1, 2], [3, 4]], dtype=np.float32)
echo b = np.array([[5, 6], [7, 8]], dtype=np.float32)
echo result = cuda_ops.matrix_multiply(a, b, use_gpu=True)
) > %DIST_DIR%\README.md

echo Creating ZIP archive...
powershell Compress-Archive -Path %DIST_DIR% -DestinationPath cuda_ops_dist.zip -Force

echo Done! Distribution package created as cuda_ops_dist.zip 
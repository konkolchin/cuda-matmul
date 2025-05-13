@echo off

REM Create build directory
if not exist build mkdir build
cd build

REM Configure with MinGW
cmake -G "MinGW Makefiles" ..

REM Build
mingw32-make -j4

REM Install Python package
cd ..
pip install -e .

REM Create tests directory if it doesn't exist
if not exist tests mkdir tests

REM Copy the built library to the tests directory
copy build\lib\libcuda_ops.so tests\

REM Return to root directory
cd ..

REM Check if build was successful
if %ERRORLEVEL% EQU 0 (
    echo Build successful! The shared library is in build/lib/
    dir build\lib\libcuda_ops.so
) else (
    echo Build failed!
    exit /b 1
) 
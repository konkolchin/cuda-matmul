@echo off
echo Creating complete package...

REM Create distribution directory
set DIST_DIR=cuda_matrix_ops
rmdir /s /q %DIST_DIR%
mkdir %DIST_DIR%
mkdir %DIST_DIR%\lib
mkdir %DIST_DIR%\src
mkdir %DIST_DIR%\tests
mkdir %DIST_DIR%\build_scripts

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

REM Copy build scripts
copy build*.bat %DIST_DIR%\build_scripts\
copy build*.sh %DIST_DIR%\build_scripts\
copy CMakeLists.txt %DIST_DIR%\build_scripts\
copy Dockerfile %DIST_DIR%\build_scripts\

REM Copy README
copy README.md %DIST_DIR%\

echo Creating ZIP archive...
powershell Compress-Archive -Path %DIST_DIR% -DestinationPath cuda_matrix_ops.zip -Force

echo Done! Complete package created as cuda_matrix_ops.zip 
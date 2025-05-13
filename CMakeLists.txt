cmake_minimum_required(VERSION 3.10)
project(cuda_ops CUDA CXX)

# Enable CUDA
enable_language(CUDA)

# Set CUDA specific flags
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Detect CUDA architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Default to compute capability 7.5 if not specified
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Force MinGW for .so building on Windows
if(WIN32)
    set(CMAKE_C_COMPILER "gcc")
    set(CMAKE_CXX_COMPILER "g++")
    set(CMAKE_SHARED_LIBRARY_PREFIX "lib")
    set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
endif()

# Find Python
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)

# Download pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Create shared library
add_library(cuda_ops SHARED
    src/cuda_ops.cu
    src/matrix_ops.cpp
    src/matrix_ops.cu
    src/bindings.cpp
)

# Set output directory
set_target_properties(cuda_ops PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    PREFIX "lib"
    CUDA_SEPARABLE_COMPILATION ON
)

# Include directories
target_include_directories(cuda_ops PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${Python_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(cuda_ops PRIVATE
    ${Python_LIBRARIES}
    cudart
    pybind11::module
)

# Print CUDA configuration
message(STATUS "CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}") 
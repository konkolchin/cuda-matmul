FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Install required packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Build the project
RUN mkdir -p build && cd build && \
    cmake .. && \
    make -j4

# The .so file will be in /app/build/lib/libcuda_ops.so 
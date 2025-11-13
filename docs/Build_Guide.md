# Point3D Interpolation Library - Build Guide

## System Requirements

### Minimum Requirements
- **Operating System**: Linux, Windows, or macOS
- **Compiler**: C++17 compatible compiler
  - GCC 7.0+ (Linux)
  - Clang 6.0+ (macOS/Linux)
  - MSVC 2019+ (Windows)
- **Build System**: CMake 3.18+
- **Memory**: 2GB RAM minimum, 8GB recommended

### Optional Requirements (for GPU acceleration)
- **CUDA Toolkit**: 11.0+ (https://developer.nvidia.com/cuda-toolkit)
- **NVIDIA GPU**: Compute Capability 6.0+ (Pascal architecture or newer)
- **GPU Memory**: 2GB+ VRAM recommended

## Quick Start Build

### Linux/macOS

```bash
# Clone the repository
git clone <repository-url>
cd point3d_interp

# Create build directory
mkdir build && cd build

# Configure with default options (GPU enabled)
cmake ..

# Build
make -j$(nproc)

# Run tests
ctest --output-on-failure

# Install (optional)
sudo make install
```

### Windows

```batch
# Using Visual Studio Developer Command Prompt
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019"
cmake --build . --config Release

# Run tests
ctest --output-on-failure
```

## Detailed Build Options

### CMake Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Release | Build type: Debug, Release, RelWithDebInfo |
| `USE_DOUBLE_PRECISION` | OFF | Use double precision floating point |
| `BUILD_TESTS` | ON | Build unit tests |
| `BUILD_EXAMPLES` | ON | Build example programs |
| `CMAKE_CUDA_ARCHITECTURES` | 60;70;75;80;86 | Target CUDA architectures |

### Build Examples

#### Basic Release Build
```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
```

#### Debug Build with Tests
```bash
cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON ..
make -j8
ctest --output-on-failure
```

#### CPU-only Build (no CUDA)
```bash
cmake -DUSE_DOUBLE_PRECISION=ON -DBUILD_EXAMPLES=OFF ..
make -j8
```

#### Specify CUDA Architecture
```bash
cmake -DCMAKE_CUDA_ARCHITECTURES="75;80;86" ..
make -j8
```

## Troubleshooting

### Common Build Issues

#### CUDA Not Found
**Error:** `CUDA_TOOLKIT_ROOT_DIR not found`

**Solution:**
```bash
# Set CUDA path explicitly
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda ..

# Or install CUDA toolkit
# Ubuntu/Debian:
sudo apt install nvidia-cuda-toolkit

# CentOS/RHEL:
sudo yum install cuda-toolkit
```

#### Compiler Too Old
**Error:** `C++17 required but compiler doesn't support it`

**Solutions:**
- **GCC**: Upgrade to GCC 7+ or use devtoolset (CentOS)
- **Clang**: Upgrade to Clang 6+
- **macOS**: Use Homebrew GCC: `brew install gcc`

#### CMake Version Too Old
**Error:** `CMake 3.18 required`

**Solution:**
```bash
# Ubuntu/Debian:
sudo apt install cmake

# Or build from source:
wget https://github.com/Kitware/CMake/releases/download/v3.26.0/cmake-3.26.0.tar.gz
tar -xzf cmake-3.26.0.tar.gz
cd cmake-3.26.0
./bootstrap && make && sudo make install
```

### GPU-Related Issues

#### No GPU Available
The library automatically falls back to CPU-only mode if:
- No NVIDIA GPU detected
- CUDA initialization fails
- GPU memory allocation fails

Check GPU status:
```bash
nvidia-smi
```

#### CUDA Architecture Mismatch
**Error:** `Unsupported CUDA architecture`

**Solution:** Update `CMAKE_CUDA_ARCHITECTURES` to match your GPU:
```bash
# Find your GPU architecture
nvidia-smi --query-gpu=compute_cap --format=csv

# Set appropriate architectures
cmake -DCMAKE_CUDA_ARCHITECTURES="70;75;80" ..
```

### Memory Issues

#### Out of Memory During Build
**Solution:** Reduce parallel jobs:
```bash
make -j2  # Instead of -j8
```

#### GPU Memory Insufficient
The library handles large datasets by falling back to CPU. For very large datasets, ensure adequate system RAM.

## Testing

### Run All Tests
```bash
cd build
ctest --output-on-failure
```

### Run Specific Tests
```bash
# Run accuracy tests
./tests/unit_tests --gtest_filter="*Accuracy*"

# Run API tests
./tests/unit_tests --gtest_filter="*API*"
```

### Performance Benchmarks
```bash
cd build
./tests/performance_benchmark
```

## Installation

### System-wide Installation
```bash
cd build
sudo make install
```

This installs:
- Headers: `/usr/local/include/point3d_interp/`
- Library: `/usr/local/lib/libpoint3d_interp.a`
- CMake config: `/usr/local/lib/cmake/point3d_interp/`

### Using Installed Library
```cmake
find_package(point3d_interp REQUIRED)
target_link_libraries(your_target point3d_interp)
```

## Cross-Compilation

### ARM64/Linux
```bash
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/arm64-toolchain.cmake ..
```

### Windows Cross-Compilation (from Linux)
```bash
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/mingw-w64-toolchain.cmake ..
```

## Development Setup

### IDE Integration

#### Visual Studio Code
```json
{
    "cmake.configureSettings": {
        "CMAKE_BUILD_TYPE": "Debug",
        "BUILD_TESTS": true
    }
}
```

#### CLion
- Open project root directory
- CLion will detect CMakeLists.txt automatically

### Code Formatting
```bash
# Install clang-format
sudo apt install clang-format

# Format all source files
find . -name "*.cpp" -o -name "*.hpp" -o -name "*.cu" -o -name "*.cuh" | xargs clang-format -i
```

## Performance Optimization

### Compiler Flags
```bash
# GCC optimization
CXXFLAGS="-O3 -march=native -ffast-math" cmake ..

# CUDA optimization
CMAKE_CUDA_FLAGS="-use_fast_math --expt-relaxed-constexpr" cmake ..
```

### Profiling
```bash
# CPU profiling
valgrind --tool=callgrind ./your_app

# GPU profiling (requires CUDA)
nvprof ./your_app
```

## Packaging

### Create Debian Package
```bash
cd build
cpack -G DEB
```

### Create RPM Package
```bash
cd build
cpack -G RPM
```

### Create Source Archive
```bash
cd build
cpack -G TGZ
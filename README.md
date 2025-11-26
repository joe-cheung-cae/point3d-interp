# Point3D Interpolation Library

A high-performance 3D magnetic field data interpolation library with GPU-accelerated tricubic Hermite interpolation.

## Features

- 🚀 **High Performance**: CUDA-based GPU acceleration with optimized kernels, supports millions of interpolation queries per second
- 📊 **Accurate**: Implements tricubic Hermite interpolation algorithm with complete gradient computation for regular grids, IDW interpolation for unstructured data with KD-tree spatial indexing
- 🔧 **Easy to Use**: Clean C++ API interface, supports single-point and batch queries with automatic data type detection
- 🏗️ **Flexible**: Supports regular grid (tricubic Hermite) and unstructured point cloud data (IDW with configurable extrapolation strategies)
- 💪 **Reliable**: Comprehensive error handling, boundary checking, and extrapolation strategies for out-of-bounds queries
- 🔄 **Compatible**: Automatic CPU/GPU switching with spatial indexing and performance optimizations
- 📊 **Visualization Ready**: Built-in Paraview VTK export for data visualization and analysis
- 🧩 **Extensible Architecture**: Modular design with abstract interfaces, factory patterns, and plugin support for easy algorithm extension
- ✅ **Production Ready**: All tests pass (25/25, 100% success rate), fully documented, and ready for production use

## Quick Start

### Build Requirements

- C++17 compatible compiler (GCC 7+, Clang 6+, MSVC 2019+)
- CMake 3.18+
- CUDA Toolkit 11.0+ (optional, for GPU acceleration)
- NVIDIA GPU (compute capability 6.0+) (optional)

### Build Steps

```bash
# Clone the repository
git clone <repository-url>
cd point3d_interp

# Create build directory
mkdir build && cd build

# Configure (GPU support enabled by default)
cmake ..

# Or disable GPU support
cmake -DUSE_DOUBLE_PRECISION=OFF ..

# Build
make -j8

# Run example
./examples/basic_usage
```

### Basic Usage

```cpp
#include "point3d_interp/interpolator_api.h"
#include <iostream>

int main() {
    using namespace p3d;

    // Create interpolator (auto-detects GPU by default)
    MagneticFieldInterpolator interp;

    // Load data from CSV file
    ErrorCode err = interp.LoadFromCSV("magnetic_field_data.csv");
    if (err != ErrorCode::Success) {
        std::cerr << "Load failed: " << ErrorCodeToString(err) << std::endl;
        return 1;
    }

    // Single-point interpolation
    Point3D query_point(1.5, 2.3, 0.8);
    InterpolationResult result;

    err = interp.Query(query_point, result);
    if (err == ErrorCode::Success && result.valid) {
        std::cout << "Magnetic field: (" << result.data.Bx << ", "
                  << result.data.By << ", " << result.data.Bz << ")" << std::endl;
    }

    // Batch interpolation
    std::vector<Point3D> query_points = {/* ... */};
    std::vector<InterpolationResult> results(query_points.size());

    err = interp.QueryBatch(query_points.data(), results.data(), query_points.size());

    return 0;
}
```

### Data Export for Visualization

Export interpolation data to Paraview VTK format for advanced visualization and analysis:

```cpp
#include "point3d_interp/interpolator_api.h"
#include <vector>

int main() {
    using namespace p3d;

    MagneticFieldInterpolator interp;
    interp.LoadFromCSV("magnetic_field_data.csv");

    // Export input sampling points with field data
    ErrorCode err = MagneticFieldInterpolator::ExportInputPoints(
        ExportFormat::ParaviewVTK, "input_points.vtk");
    if (err != ErrorCode::Success) {
        std::cerr << "Export failed: " << ErrorCodeToString(err) << std::endl;
        return 1;
    }

    // Perform interpolation queries
    std::vector<Point3D> query_points = {
        {1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}, {3.0, 3.0, 3.0}
    };
    std::vector<InterpolationResult> results;
    interp.QueryBatch(query_points, results);

    // Export interpolated results
    err = MagneticFieldInterpolator::ExportOutputPoints(
        ExportFormat::ParaviewVTK, query_points, results, "output_points.vtk");

    return 0;
}
```

### Direct CUDA Kernel Access (Advanced)

For maximum performance and integration with existing CUDA applications, you can access the interpolation kernel directly:

```cpp
#include "point3d_interp/interpolator_api.h"
#include <cuda_runtime.h>

int main() {
    using namespace p3d;

    MagneticFieldInterpolator interp(true);  // GPU enabled
    interp.LoadFromCSV("data.csv");

    // Get GPU device pointers
    const Point3D* d_grid_points = interp.GetDeviceGridPoints();
    const MagneticFieldData* d_field_data = interp.GetDeviceFieldData();

    // Allocate your own GPU memory for queries and results
    Point3D* d_query_points;
    InterpolationResult* d_results;
    cudaMalloc(&d_query_points, num_queries * sizeof(Point3D));
    cudaMalloc(&d_results, num_queries * sizeof(InterpolationResult));

    // Copy query points to GPU
    cudaMemcpy(d_query_points, host_queries, num_queries * sizeof(Point3D), cudaMemcpyHostToDevice);

    // Get optimal kernel configuration
    KernelConfig config;
    interp.GetOptimalKernelConfig(num_queries, config);

    // Launch kernel directly
    TricubicHermiteInterpolationKernel<<<config.grid_x, config.grid_y, config.grid_z, config.block_x, config.block_y, config.block_z>>>(
        d_query_points, d_field_data, interp.GetGridParams(), d_results, num_queries);

    // Copy results back
    cudaMemcpy(host_results, d_results, num_queries * sizeof(InterpolationResult), cudaMemcpyDeviceToHost);

    return 0;
}
```

## Data Format

### CSV File Format

First line is header, each subsequent line contains 15 fields:

```csv
x,y,z,Bx,By,Bz,dBx_dx,dBx_dy,dBx_dz,dBy_dx,dBy_dy,dBy_dz,dBz_dx,dBz_dy,dBz_dz
0.0,0.0,0.0,0.123,-0.456,0.789,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09
1.0,0.0,0.0,0.125,-0.454,0.791,0.011,0.021,0.031,0.041,0.051,0.061,0.071,0.081,0.091
...
```

- `x,y,z`: 3D spatial coordinates
- `Bx,By,Bz`: Magnetic field vector components
- `dBx_dx,dBx_dy,dBx_dz`: Derivatives of Bx with respect to x, y, z
- `dBy_dx,dBy_dy,dBy_dz`: Derivatives of By with respect to x, y, z
- `dBz_dx,dBz_dy,dBz_dz`: Derivatives of Bz with respect to x, y, z

### Data Requirements

**Regular Grid Data:**
- Data points must be on a regular 3D grid
- Grid spacing can be non-uniform but must be regular
- Supports 3x3x3 to arbitrary size grids
- Automatic grid parameter detection, no manual configuration needed

**Unstructured Point Cloud Data:**
- Data points can be at arbitrary 3D positions
- No grid structure required
- Uses inverse distance weighting (IDW) interpolation with GPU acceleration
- Automatic detection when loading non-regular data

## API Reference

### MagneticFieldInterpolator

Main interface class providing data loading and interpolation functionality.

#### Constructor

```cpp
MagneticFieldInterpolator(bool use_gpu = true, int device_id = 0,
                         InterpolationMethod method = InterpolationMethod::TricubicHermite,
                         ExtrapolationMethod extrapolation_method = ExtrapolationMethod::None);
```

**Parameters:**
- `use_gpu`: Whether to use GPU acceleration (default: true)
- `device_id`: CUDA device ID (default: 0)
- `method`: Interpolation method (default: TricubicHermite)
- `extrapolation_method`: Extrapolation method for out-of-bounds queries in unstructured data (default: None)

#### Methods

- `ErrorCode LoadFromCSV(const std::string& filepath)`: Load data from CSV file
- `ErrorCode LoadFromMemory(const Point3D*, const MagneticFieldData*, size_t)`: Load data from memory
- `ErrorCode Query(const Point3D&, InterpolationResult&)`: Single-point interpolation query
- `ErrorCode QueryBatch(const Point3D*, InterpolationResult*, size_t)`: Batch interpolation query
- `ErrorCode QueryBatch(const std::vector<Point3D>&, std::vector<InterpolationResult>&)`: Batch interpolation query with vectors
- `InterpolationResult QueryEx(const Point3D&)`: Single-point interpolation query (throws on error)
- `std::vector<InterpolationResult> QueryBatchEx(const std::vector<Point3D>&)`: Batch interpolation query with vectors (throws on error)
- `const GridParams& GetGridParams() const`: Get grid parameters
- `bool IsDataLoaded() const`: Check if data is loaded
- `size_t GetDataPointCount() const`: Get number of data points

### Data Structures

- `Point3D`: 3D point structure (x, y, z)
- `MagneticFieldData`: Magnetic field data structure (Bx, By, Bz, dBx_dx, dBx_dy, dBx_dz, dBy_dx, dBy_dy, dBy_dz, dBz_dx, dBz_dy, dBz_dz)
- `InterpolationResult`: Interpolation result (data, valid)
- `GridParams`: Grid parameters (origin, spacing, dimensions, bounds)
- `InterpolationMethod`: Interpolation method enum (TricubicHermite, IDW)
- `ExtrapolationMethod`: Extrapolation method enum (None, NearestNeighbor, LinearExtrapolation)

## Performance Characteristics

### Benchmark Results (Latest)

#### Structured Data (Regular Grids)
| Configuration | Data Points | Query Points | CPU Time | GPU Time | Speedup |
|---------------|-------------|--------------|----------|----------|---------|
| 10x10x10 grid | 1,000 | 10,000 | 1.06ms | 0.72ms | 1.47x |
| 20x20x20 grid | 8,000 | 10,000 | 1.06ms | 0.70ms | 1.51x |
| 30x30x30 grid | 27,000 | 10,000 | 0.85ms | 0.78ms | 1.09x |
| 50x50x50 grid | 125,000 | 10,000 | 1.63ms | 0.90ms | 1.81x |

#### Unstructured Data (Point Clouds)
| Configuration | Data Points | Query Points | CPU Time | GPU Time | Speedup |
|---------------|-------------|--------------|----------|----------|---------|
| 1,000 points | 1,000 | 10,000 | 72.03ms | 0.80ms | 89.81x |
| 8,000 points | 8,000 | 10,000 | 579.45ms | 1.57ms | 369.54x |
| 27,000 points | 27,000 | 10,000 | 2002.41ms | 0.97ms | 2068.60x |
| 125,000 points | 125,000 | 10,000 | 9457.22ms | 2.08ms | 4548.93x |

*Test Environment: Intel i7-13700K + NVIDIA GeForce RTX 3050 Ti Laptop GPU*

### Optimization Recommendations

1. **Batch Queries**: Prefer `QueryBatch()` over looping `Query()` calls
2. **Memory Layout**: Ensure query point arrays are contiguous
3. **GPU Memory**: Avoid frequent CPU-GPU data transfers
4. **Precision Choice**: Single precision (float) usually sufficient, 2x faster than double precision

## Architecture Design

The library uses a modular, extensible architecture based on abstract interfaces and factory patterns:

### Core Components

```
point3d_interp/
├── include/          # Public headers
│   └── point3d_interp/
│       ├── interpolator_api.h       # Main API (Pimpl pattern)
│       ├── types.h                  # Data type definitions
│       ├── error_codes.h            # Error codes
│       ├── data_loader.h            # Data loader
│       ├── grid_structure.h         # Grid structure
│       ├── cpu_interpolator.h       # CPU interpolator
│       ├── unstructured_interpolator.h # Unstructured data interpolator
│       ├── kd_tree.h                # KD-tree spatial indexing
│       ├── spatial_grid.h           # GPU spatial grid
│       ├── memory_manager.h         # GPU memory management
│       ├── interpolator_interface.h # Abstract interpolator interface
│       ├── interpolator_adapters.h  # Adapter classes for existing interpolators
│       └── interpolator_factory.h   # Factory classes for interpolator creation
├── src/              # Implementation files
│   ├── interpolator_api.cu          # API implementation (CUDA)
│   ├── data_loader.cpp              # CSV parsing
│   ├── grid_structure.cpp           # Grid management
│   ├── cpu_interpolator.cpp         # CPU interpolation
│   ├── unstructured_interpolator.cpp # IDW interpolation
│   ├── kd_tree.cpp                  # KD-tree implementation
│   ├── spatial_grid.cpp             # GPU spatial grid
│   ├── cuda_interpolator.cu         # CUDA kernels
│   ├── memory_manager.cu            # GPU memory management
│   ├── interpolator_adapters.cpp    # Adapter implementations
│   └── interpolator_factory.cpp     # Factory implementations
├── examples/         # Example programs
├── tests/            # Unit tests
├── data/             # Sample data
└── docs/             # Documentation
```

### Design Patterns

#### Abstract Interface Pattern
- `IInterpolator`: Defines the contract for all interpolator implementations
- Enables polymorphic usage and decouples algorithm implementations from the API layer
- Supports both structured and unstructured data types

#### Adapter Pattern
- `CPUStructuredInterpolatorAdapter`: Adapts existing CPU structured grid interpolator
- `CPUUnstructuredInterpolatorAdapter`: Adapts existing CPU unstructured data interpolator
- `GPUStructuredInterpolatorAdapter`: Adapts GPU structured grid interpolator
- `GPUUnstructuredInterpolatorAdapter`: Adapts GPU unstructured data interpolator

#### Factory Pattern
- `InterpolatorFactory`: Creates appropriate interpolator based on data type and requirements
- `PluginInterpolatorFactory`: Supports dynamic loading of interpolation algorithms
- `GlobalInterpolatorFactory`: Manages factory instances globally

#### Strategy Pattern
- Automatic data type detection (regular grid vs unstructured)
- Dynamic algorithm selection based on data characteristics
- Configurable extrapolation strategies for out-of-bounds queries

### Extension Points

#### Adding New Interpolation Algorithms
```cpp
class MyCustomInterpolator : public IInterpolator {
public:
    InterpolationResult query(const Point3D& point) const override {
        // Implement your algorithm
    }
    // ... other interface methods
};
```

#### Registering Custom Factories
```cpp
class MyCustomFactory : public IInterpolatorFactory {
public:
    std::unique_ptr<IInterpolator> createInterpolator(...) override {
        // Return your custom interpolator
    }
};

// Register globally
GlobalInterpolatorFactory::instance().registerFactory(
    std::make_unique<MyCustomFactory>());
```

This architecture ensures the library remains maintainable, testable, and extensible while preserving backward compatibility.

## Build Options

### CMake Options

- `USE_DOUBLE_PRECISION`: Use double precision floating point (default: OFF)
- `BUILD_TESTS`: Build tests (default: ON)
- `BUILD_EXAMPLES`: Build examples (default: ON)

### Compilation Options

```bash
# Enable double precision
cmake -DUSE_DOUBLE_PRECISION=ON ..

# Build library only
cmake -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF ..

# Specify CUDA architectures
cmake -DCMAKE_CUDA_ARCHITECTURES="75;80;86" ..
```

## Error Handling

The library uses error codes instead of exceptions for error handling:

- `Success`: Operation successful
- `FileNotFound`: File not found
- `InvalidFileFormat`: Invalid file format
- `InvalidGridData`: Invalid grid data
- `CudaError`: CUDA-related error
- `QueryOutOfBounds`: Query point out of bounds

## License

[License Information]

## Contributing

Welcome to submit Issues and Pull Requests!

## Contact Information

[Contact Information]
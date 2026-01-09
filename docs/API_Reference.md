# Point3D Interpolation Library - API Reference

## Overview

The Point3D Interpolation Library provides a high-performance C++ API for 3D magnetic field data interpolation. It supports both regular grid data (using tricubic Hermite interpolation with optional GPU acceleration) and unstructured point cloud data (using Hermite Moving Least Squares interpolation for superior accuracy or inverse distance weighting interpolation as fallback).

## Core Classes

### MagneticFieldInterpolator

The main interface class for magnetic field interpolation. Uses a modular architecture with abstract interfaces and factory patterns for extensibility.

The main interface class for magnetic field interpolation.

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

##### Data Loading

```cpp
void LoadFromCSV(const std::string& filepath);
```

Loads magnetic field data from a CSV file. Throws `std::runtime_error` on failure.

**Parameters:**
- `filepath`: Path to the CSV file

**Throws:** `std::runtime_error` if the file cannot be loaded or parsed

```cpp
void LoadFromMemory(const Point3D* points, const MagneticFieldData* field_data, size_t count);
```

Loads magnetic field data from memory arrays. Automatically detects whether the data forms a regular grid or is an unstructured point cloud. Throws `std::runtime_error` on failure.

**Parameters:**
- `points`: Array of 3D coordinates
- `field_data`: Array of magnetic field data
- `count`: Number of data points

**Throws:** `std::runtime_error` if the data is invalid

**Note:** For regular grid data, GPU acceleration is available. For unstructured data, Hermite Moving Least Squares (HMLS) provides superior accuracy using gradient information, with IDW as fallback.

##### Query Methods

```cpp
void Query(const Point3D& query_point, InterpolationResult& result);
```

Performs single-point interpolation. Throws `std::runtime_error` on failure.

**Parameters:**
- `query_point`: The point to interpolate at
- `result`: Output interpolation result

**Throws:** `std::runtime_error` if interpolation fails

```cpp
void QueryBatch(const Point3D* query_points, InterpolationResult* results, size_t count);
```

Performs batch interpolation for multiple points. Throws `std::runtime_error` on failure.

**Parameters:**
- `query_points`: Array of query points
- `results`: Array to store results
- `count`: Number of query points

**Throws:** `std::runtime_error` if interpolation fails

##### Information Methods

```cpp
const GridParams& GetGridParams() const;
```

Returns the grid parameters of the loaded data.

**Returns:** Grid parameters structure

```cpp
bool IsDataLoaded() const;
```

Checks if data has been loaded.

**Returns:** true if data is loaded

```cpp
size_t GetDataPointCount() const;
```

Returns the number of data points.

**Returns:** Number of data points

```cpp
std::vector<Point3D> GetCoordinates() const;
```

Returns the coordinates of loaded data points.

**Returns:** Vector of coordinate points

```cpp
std::vector<MagneticFieldData> GetFieldData() const;
```

Returns the magnetic field data of loaded data points.

**Returns:** Vector of magnetic field data

##### Export Methods

```cpp
static void ExportInputPoints(const std::vector<Point3D>& coordinates,
                              const std::vector<MagneticFieldData>& field_data,
                              ExportFormat format, const std::string& filename);
```

Exports input sampling points with their magnetic field data to a visualization file. Throws `std::runtime_error` on failure.

**Parameters:**
- `coordinates`: Input coordinates vector
- `field_data`: Magnetic field data vector
- `format`: Export format (currently supports ParaviewVTK)
- `filename`: Output filename

**Throws:** `std::runtime_error` if export fails

**Note:** Exports coordinates and magnetic field data including derivatives.

```cpp
static void ExportOutputPoints(ExportFormat format, const std::vector<Point3D>& query_points,
                               const std::vector<InterpolationResult>& results, const std::string& filename);
```

Exports output interpolation points with their results to a visualization file. Throws `std::runtime_error` on failure.

**Parameters:**
- `format`: Export format (currently supports ParaviewVTK)
- `query_points`: Query points used for interpolation
- `results`: Interpolation results
- `filename`: Output filename

**Throws:** `std::runtime_error` if export fails

**Note:** Exports query points with interpolated magnetic field data, derivatives, and validity flags.

##### Direct CUDA Kernel Access Methods

```cpp
const Point3D* GetDeviceGridPoints() const;
```

Returns GPU device pointer to grid coordinates for direct CUDA kernel access.

**Returns:** Device pointer to grid points, nullptr if not available

```cpp
const MagneticFieldData* GetDeviceFieldData() const;
```

Returns GPU device pointer to field data for direct CUDA kernel access.

**Returns:** Device pointer to field data, nullptr if not available

```cpp
const GridParams* GetDeviceGridParams() const;
```

Returns GPU device pointer to grid parameters for direct CUDA kernel access.

**Returns:** Device pointer to grid parameters, nullptr if not available

```cpp
void LaunchInterpolationKernel(const Point3D* d_query_points,
                               InterpolationResult* d_results,
                               size_t count,
                               cudaStream_t stream = 0);
```

Launches the interpolation kernel directly with custom device pointers. Throws `std::runtime_error` on failure.

**Parameters:**
- `d_query_points`: Device pointer to query points array
- `d_results`: Device pointer to results array
- `count`: Number of query points
- `stream`: CUDA stream for asynchronous execution

**Throws:** `std::runtime_error` if kernel launch fails

```cpp
void GetOptimalKernelConfig(size_t query_count, KernelConfig& config) const;
```

Gets optimal kernel launch configuration for given query count.

**Parameters:**
- `query_count`: Number of query points
- `config`: Output kernel configuration

```cpp
void GetLastKernelTime(float& kernel_time_ms) const;
```

Gets the execution time of the last GPU kernel call in milliseconds. Throws `std::runtime_error` if no timing data is available.

**Parameters:**
- `kernel_time_ms`: Output kernel execution time

**Throws:** `std::runtime_error` if timing data is not available

**Note:** Only valid after a GPU QueryBatch call. Returns the time spent in GPU kernel execution only, excluding memory transfers.

## Data Structures

### Point3D

Represents a 3D point with x, y, z coordinates.

```cpp
struct Point3D {
    Real x, y, z;

    Point3D(Real x = 0, Real y = 0, Real z = 0);
};
```

### MagneticFieldData

Contains magnetic field data and its spatial derivatives at a point. For regular grid data, derivatives are computed using tricubic Hermite interpolation. For unstructured data using Hermite Moving Least Squares (HMLS), derivatives are computed as part of the interpolation. For IDW interpolation, derivatives are set to zero.

```cpp
struct MagneticFieldData {
    Real Bx;  // Magnetic field component in x direction
    Real By;  // Magnetic field component in y direction
    Real Bz;  // Magnetic field component in z direction
    // Spatial derivatives of Bx, By, Bz with respect to x, y, z (computed via tricubic interpolation)
    Real dBx_dx, dBx_dy, dBx_dz;
    Real dBy_dx, dBy_dy, dBy_dz;
    Real dBz_dx, dBz_dy, dBz_dz;
};
```

### InterpolationResult

Result of an interpolation query.

```cpp
struct InterpolationResult {
    MagneticFieldData data;  // Interpolated field data
    bool valid;              // Whether the result is valid
};
```

### GridParams

Parameters describing the interpolation grid.

```cpp
struct GridParams {
    Point3D origin;                    // Grid origin
    Point3D spacing;                   // Grid spacing (dx, dy, dz)
    std::array<uint32_t, 3> dimensions; // Grid dimensions (nx, ny, nz)
    Point3D min_bound;                 // Minimum bounds
    Point3D max_bound;                 // Maximum bounds
};
```

### ExtrapolationMethod

Enum specifying the extrapolation strategy for points outside the data bounds (applicable to unstructured data).

```cpp
enum class ExtrapolationMethod {
    None,              // No special extrapolation (IDW interpolates naturally)
    NearestNeighbor,   // Use value of nearest data point
    LinearExtrapolation // Linear extrapolation (placeholder, uses nearest for now)
};
```

#### Extrapolation Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `None` | IDW interpolation continues naturally outside data bounds | Default behavior, suitable when data covers the query region |
| `NearestNeighbor` | Returns the magnetic field values of the closest data point | Simple and fast extrapolation for points far from data |
| `LinearExtrapolation` | Performs linear extrapolation using gradient estimation from nearest neighbors | Advanced extrapolation for smoother transitions outside data bounds |

### ExportFormat

Enum specifying the visualization format for data export.

```cpp
enum class ExportFormat {
    ParaviewVTK,  // VTK legacy format for Paraview
    Tecplot       // Reserved for future implementation
};
```

### InterpolationMethod

Enum specifying the interpolation algorithm to use.

```cpp
enum class InterpolationMethod {
    Trilinear,          // Simple trilinear interpolation (regular grids only)
    TricubicHermite,    // Tricubic Hermite interpolation with gradient computation (regular grids)
    IDW,                // Inverse Distance Weighting (unstructured data)
    HermiteMLS          // Hermite Moving Least Squares (unstructured data, superior accuracy)
};
```

#### Interpolation Methods

| Method | Data Type | Description | Accuracy | Performance |
|--------|-----------|-------------|----------|-------------|
| `Trilinear` | Regular Grid | Simple linear interpolation in 3D | Basic | Fast |
| `TricubicHermite` | Regular Grid | Cubic interpolation with computed gradients | High | Medium |
| `IDW` | Unstructured | Inverse distance weighting | Medium | Fast |
| `HermiteMLS` | Unstructured | Moving least squares using function values and gradients | Very High | Medium |

## Extensibility Interfaces

The library provides abstract interfaces for extending interpolation capabilities.

### IInterpolator

Abstract base class defining the interface for all interpolator implementations.

```cpp
class IInterpolator {
public:
    virtual ~IInterpolator() = default;

    virtual InterpolationResult query(const Point3D& point) const = 0;
    virtual std::vector<InterpolationResult> queryBatch(const std::vector<Point3D>& points) const = 0;
    virtual bool supportsGPU() const = 0;
    virtual DataStructureType getDataType() const = 0;
    virtual InterpolationMethod getMethod() const = 0;
    virtual ExtrapolationMethod getExtrapolationMethod() const = 0;
    virtual size_t getDataCount() const = 0;
    virtual void getBounds(Point3D& min_bound, Point3D& max_bound) const = 0;
    virtual GridParams getGridParams() const = 0;
    virtual std::vector<Point3D> getCoordinates() const = 0;
    virtual std::vector<MagneticFieldData> getFieldData() const = 0;
};
```

### IInterpolatorFactory

Abstract factory interface for creating interpolator instances.

```cpp
class IInterpolatorFactory {
public:
    virtual ~IInterpolatorFactory() = default;

    virtual std::unique_ptr<IInterpolator> createInterpolator(
        DataStructureType dataType,
        InterpolationMethod method,
        const std::vector<Point3D>& coordinates,
        const std::vector<MagneticFieldData>& fieldData,
        ExtrapolationMethod extrapolation,
        bool useGPU) = 0;

    virtual bool supports(DataStructureType dataType, InterpolationMethod method, bool useGPU) const = 0;
};
```

### InterpolatorFactory

Concrete factory implementation providing the standard interpolation algorithms.

```cpp
class InterpolatorFactory : public IInterpolatorFactory {
public:
    std::unique_ptr<IInterpolator> createInterpolator(
        DataStructureType dataType,
        InterpolationMethod method,
        const std::vector<Point3D>& coordinates,
        const std::vector<MagneticFieldData>& fieldData,
        ExtrapolationMethod extrapolation,
        bool useGPU) override;

    bool supports(DataStructureType dataType, InterpolationMethod method, bool useGPU) const override;
};
```

### PluginInterpolatorFactory

Factory supporting dynamic loading of interpolation algorithms via plugins.

```cpp
class PluginInterpolatorFactory : public IInterpolatorFactory {
public:
    void registerPlugin(std::unique_ptr<IInterpolatorFactory> plugin);

    std::unique_ptr<IInterpolator> createInterpolator(
        DataStructureType dataType,
        InterpolationMethod method,
        const std::vector<Point3D>& coordinates,
        const std::vector<MagneticFieldData>& fieldData,
        ExtrapolationMethod extrapolation,
        bool useGPU) override;

    bool supports(DataStructureType dataType, InterpolationMethod method, bool useGPU) const override;
};
```

### GlobalInterpolatorFactory

Global factory registry for managing interpolator factories.

```cpp
class GlobalInterpolatorFactory {
public:
    static GlobalInterpolatorFactory& instance();

    void registerFactory(std::unique_ptr<IInterpolatorFactory> factory);

    std::unique_ptr<IInterpolator> createInterpolator(
        DataStructureType dataType,
        InterpolationMethod method,
        const std::vector<Point3D>& coordinates,
        const std::vector<MagneticFieldData>& fieldData,
        ExtrapolationMethod extrapolation = ExtrapolationMethod::None,
        bool useGPU = false);
};
```

### DataStructureType

Enum specifying the type of data structure being interpolated.

```cpp
enum class DataStructureType {
    RegularGrid,    // Structured grid data (supports tricubic Hermite)
    Unstructured    // Unstructured point cloud data (supports HMLS and IDW)
};
```

## Error Handling

The library uses modern C++ exceptions for error handling. All API methods throw `std::runtime_error` with descriptive error messages when errors occur. Client code should wrap API calls in try-catch blocks:

```cpp
try {
    MagneticFieldInterpolator interp;
    interp.LoadFromCSV("data.csv");
    InterpolationResult result;
    interp.Query(Point3D(1.0, 1.0, 1.0), result);
} catch (const std::runtime_error& e) {
    std::cerr << "Interpolation error: " << e.what() << std::endl;
}
```

Common error conditions that may throw exceptions:
- File not found or cannot be read
- Invalid CSV file format or corrupted data
- Invalid or inconsistent grid/mesh data
- Memory allocation failures
- CUDA-related errors (when GPU acceleration is enabled)
- Query points outside valid data bounds
- CUDA not available or device errors
- Invalid parameters passed to functions

## Usage Examples

### Basic Usage

```cpp
#include "point3d_interp/interpolator_api.h"
#include <stdexcept>

int main() {
    try {
        // Create interpolator
        p3d::MagneticFieldInterpolator interp;

        // Load data (throws on error)
        interp.LoadFromCSV("data.csv");

        // Query single point (throws on error)
        p3d::Point3D query(1.5, 2.3, 0.8);
        p3d::InterpolationResult result;

        interp.Query(query, result);
        if (result.valid) {
            std::cout << "Bx = " << result.data.Bx << std::endl;
        }

        return 0;
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
```

### Hermite Moving Least Squares (HMLS) Usage

For unstructured data with superior accuracy using gradient information:

```cpp
#include "point3d_interp/interpolator_api.h"
#include <stdexcept>

int main() {
    try {
        // Create HMLS interpolator for unstructured data
        p3d::MagneticFieldInterpolator interp(true, 0, p3d::InterpolationMethod::HermiteMLS);

        // Load unstructured data (throws on error)
        interp.LoadFromCSV("unstructured_data.csv");

        // Query single point with HMLS (throws on error)
        p3d::Point3D query(1.5, 2.3, 0.8);
        p3d::InterpolationResult result;

        interp.Query(query, result);
        if (result.valid) {
            std::cout << "HMLS Bx = " << result.data.Bx << std::endl;
            std::cout << "HMLS dBx/dx = " << result.data.dBx_dx << std::endl;
        }

        return 0;
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
```

### Batch Queries

```cpp
#include "point3d_interp/interpolator_api.h"
#include <vector>
#include <stdexcept>

int main() {
    try {
        p3d::MagneticFieldInterpolator interp;
        interp.LoadFromCSV("data.csv");

        // Prepare query points
        std::vector<p3d::Point3D> queries = {
            {1.0, 1.0, 1.0},
            {2.0, 2.0, 2.0},
            {3.0, 3.0, 3.0}
        };

        std::vector<p3d::InterpolationResult> results(queries.size());

        // Batch query (throws on error)
        interp.QueryBatch(queries.data(), results.data(), queries.size());

        return 0;
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
```

### Memory-Based Loading

```cpp
#include "point3d_interp/interpolator_api.h"
#include <vector>
#include <stdexcept>

int main() {
    try {
        // Prepare data in memory
        std::vector<p3d::Point3D> points = {
            {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
            {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}
        };

        std::vector<p3d::MagneticFieldData> field_data(points.size());
        // Fill field_data...

        p3d::MagneticFieldInterpolator interp;
        interp.LoadFromMemory(
            points.data(),
            field_data.data(),
            points.size()
        );

        return 0;
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
```

### Direct CUDA Kernel Access

For advanced users who need maximum performance or integration with existing CUDA applications, the library provides direct access to the CUDA interpolation kernel:

```cpp
#include "point3d_interp/interpolator_api.h"
#include <cuda_runtime.h>
#include <stdexcept>

int main() {
    try {
        p3d::MagneticFieldInterpolator interp(true);  // GPU enabled
        interp.LoadFromCSV("data.csv");

        // Get GPU device pointers
        const p3d::Point3D* d_grid_points = interp.GetDeviceGridPoints();
        const p3d::MagneticFieldData* d_field_data = interp.GetDeviceFieldData();

        if (!d_grid_points || !d_field_data) {
            throw std::runtime_error("GPU not available or data not loaded");
        }

        // Allocate GPU memory for your queries and results
        p3d::Point3D* d_query_points;
        p3d::InterpolationResult* d_results;
        cudaMalloc(&d_query_points, num_queries * sizeof(p3d::Point3D));
        cudaMalloc(&d_results, num_queries * sizeof(p3d::InterpolationResult));

        // Copy your query points to GPU
        cudaMemcpy(d_query_points, host_query_points,
                   num_queries * sizeof(p3d::Point3D), cudaMemcpyHostToDevice);

        // Get optimal kernel launch configuration
        p3d::KernelConfig config;
        interp.GetOptimalKernelConfig(num_queries, config);

        // Get grid parameters
        p3d::GridParams grid_params = interp.GetGridParams();

        // Launch the interpolation kernel directly (throws on error)
        interp.LaunchInterpolationKernel(d_query_points, d_results, num_queries);

        // Copy results back to host
        cudaMemcpy(host_results, d_results,
                   num_queries * sizeof(p3d::InterpolationResult), cudaMemcpyDeviceToHost);

        // Clean up
        cudaFree(d_query_points);
        cudaFree(d_results);

        return 0;
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
```

### Exporting Data for Visualization

```cpp
#include "point3d_interp/interpolator_api.h"
#include <vector>
#include <stdexcept>

int main() {
    try {
        p3d::MagneticFieldInterpolator interp;
        interp.LoadFromCSV("data.csv");

        // Get loaded data for export
        auto coordinates = interp.GetCoordinates();
        auto field_data = interp.GetFieldData();

        // Export input sampling points (throws on error)
        p3d::MagneticFieldInterpolator::ExportInputPoints(
            coordinates, field_data, p3d::ExportFormat::ParaviewVTK, "input_points.vtk");

        // Perform some queries
        std::vector<p3d::Point3D> queries = {
            {1.0, 1.0, 1.0},
            {2.0, 2.0, 2.0}
        };
        std::vector<p3d::InterpolationResult> results;
        interp.QueryBatch(queries, results);

        // Export output interpolation points (throws on error)
        p3d::MagneticFieldInterpolator::ExportOutputPoints(
            p3d::ExportFormat::ParaviewVTK, queries, results, "output_points.vtk");

        return 0;
    } catch (const std::runtime_error& e) {
        std::cerr << "Export error: " << e.what() << std::endl;
        return 1;
    }
}
```

**Note:** The exported VTK files can be opened in Paraview for visualization. Input points contain the original magnetic field data with derivatives, while output points contain interpolated values with validity information.

**Note:** Direct CUDA kernel access requires CUDA programming knowledge and manual memory management. The kernel function `TricubicHermiteInterpolationKernel` is declared in the public header for external use.

## Thread Safety

The `MagneticFieldInterpolator` class is not thread-safe. Create separate instances for concurrent use.

## Performance Considerations

- Use `QueryBatch()` for multiple queries instead of looping with `Query()`
- GPU acceleration provides 1.5-5x speedup for structured data and 10-100x speedup for unstructured HMLS data
- For unstructured data, Hermite Moving Least Squares (HMLS) provides 2-5x better accuracy than IDW but with higher computational cost (O(k³) where k is neighbors per query)
- IDW interpolation with KD-tree spatial indexing (O(log N) complexity) provides fast performance for large datasets
- Memory layout should be contiguous for optimal performance
- Avoid frequent CPU-GPU data transfers
- For large unstructured datasets (>1000 points), KD-tree spatial indexing is automatically used for optimal query performance
- Batch processing with coalesced memory access maximizes GPU utilization
- Choose HMLS for accuracy-critical applications, IDW for performance-critical applications

## Extending the Library

The library's modular architecture allows for easy extension with custom interpolation algorithms.

### Adding a Custom Interpolation Algorithm

```cpp
#include "point3d_interp/interpolator_interface.h"

class MyCustomInterpolator : public p3d::IInterpolator {
public:
    MyCustomInterpolator(const std::vector<p3d::Point3D>& coords,
                        const std::vector<p3d::MagneticFieldData>& data)
        : coordinates_(coords), field_data_(data) {}

    p3d::InterpolationResult query(const p3d::Point3D& point) const override {
        // Implement your custom interpolation algorithm
        p3d::InterpolationResult result;
        // ... your algorithm here ...
        return result;
    }

    std::vector<p3d::InterpolationResult> queryBatch(
        const std::vector<p3d::Point3D>& points) const override {
        std::vector<p3d::InterpolationResult> results;
        for (const auto& point : points) {
            results.push_back(query(point));
        }
        return results;
    }

    // Implement other required interface methods
    bool supportsGPU() const override { return false; }
    p3d::DataStructureType getDataType() const override { return p3d::DataStructureType::Unstructured; }
    p3d::InterpolationMethod getMethod() const override { return p3d::InterpolationMethod::IDW; } // Or custom enum value
    p3d::ExtrapolationMethod getExtrapolationMethod() const override { return p3d::ExtrapolationMethod::None; }
    size_t getDataCount() const override { return coordinates_.size(); }
    void getBounds(p3d::Point3D& min_bound, p3d::Point3D& max_bound) const override {
        // Calculate bounds
    }
    p3d::GridParams getGridParams() const override { return p3d::GridParams(); }
    std::vector<p3d::Point3D> getCoordinates() const override { return coordinates_; }
    std::vector<p3d::MagneticFieldData> getFieldData() const override { return field_data_; }

private:
    std::vector<p3d::Point3D> coordinates_;
    std::vector<p3d::MagneticFieldData> field_data_;
};
```

### Creating a Custom Factory

```cpp
#include "point3d_interp/interpolator_factory.h"

class MyCustomFactory : public p3d::IInterpolatorFactory {
public:
    std::unique_ptr<p3d::IInterpolator> createInterpolator(
        p3d::DataStructureType dataType,
        p3d::InterpolationMethod method,
        const std::vector<p3d::Point3D>& coordinates,
        const std::vector<p3d::MagneticFieldData>& fieldData,
        p3d::ExtrapolationMethod extrapolation,
        bool useGPU) override {

        if (method == p3d::InterpolationMethod::IDW && !useGPU) { // Your custom condition
            return std::make_unique<MyCustomInterpolator>(coordinates, fieldData);
        }
        return nullptr;
    }

    bool supports(p3d::DataStructureType dataType, p3d::InterpolationMethod method, bool useGPU) const override {
        return method == p3d::InterpolationMethod::IDW && !useGPU;
    }
};
```

### Registering Custom Algorithms

```cpp
#include "point3d_interp/interpolator_factory.h"

int main() {
    // Register your custom factory globally
    p3d::GlobalInterpolatorFactory::instance().registerFactory(
        std::make_unique<MyCustomFactory>());

    // Now the library will automatically use your custom interpolator
    // when appropriate conditions are met
    p3d::MagneticFieldInterpolator interp;
    interp.LoadFromCSV("data.csv");

    // Your custom algorithm will be used automatically
    p3d::InterpolationResult result;
    interp.Query(p3d::Point3D(1.0, 1.0, 1.0), result);

    return 0;
}
```

### Plugin System

For more advanced extensibility, use the plugin system:

```cpp
#include "point3d_interp/interpolator_factory.h"

int main() {
    // Create a plugin factory
    p3d::PluginInterpolatorFactory pluginFactory;

    // Register multiple algorithms
    pluginFactory.registerPlugin(std::make_unique<MyCustomFactory>());
    pluginFactory.registerPlugin(std::make_unique<AnotherCustomFactory>());

    // Use the plugin factory
    auto interpolator = pluginFactory.createInterpolator(
        p3d::DataStructureType::Unstructured,
        p3d::InterpolationMethod::IDW,
        coordinates, fieldData,
        p3d::ExtrapolationMethod::None,
        false);

    return 0;
}
```

This architecture ensures the library remains maintainable, testable, and easily extensible while preserving full backward compatibility.
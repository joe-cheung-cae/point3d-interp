# Point3D Interpolation Library - API Reference

## Overview

The Point3D Interpolation Library provides a high-performance C++ API for 3D magnetic field data interpolation with optional GPU acceleration.

## Core Classes

### MagneticFieldInterpolator

The main interface class for magnetic field interpolation.

#### Constructor

```cpp
MagneticFieldInterpolator(bool use_gpu = true, int device_id = 0);
```

**Parameters:**
- `use_gpu`: Whether to use GPU acceleration (default: true)
- `device_id`: CUDA device ID (default: 0)

#### Methods

##### Data Loading

```cpp
ErrorCode LoadFromCSV(const std::string& filepath);
```

Loads magnetic field data from a CSV file.

**Parameters:**
- `filepath`: Path to the CSV file

**Returns:** Error code indicating success or failure

```cpp
ErrorCode LoadFromMemory(const Point3D* points, const MagneticFieldData* field_data, size_t count);
```

Loads magnetic field data from memory arrays.

**Parameters:**
- `points`: Array of 3D coordinates
- `field_data`: Array of magnetic field data
- `count`: Number of data points

**Returns:** Error code

##### Query Methods

```cpp
ErrorCode Query(const Point3D& query_point, InterpolationResult& result);
```

Performs single-point interpolation.

**Parameters:**
- `query_point`: The point to interpolate at
- `result`: Output interpolation result

**Returns:** Error code

```cpp
ErrorCode QueryBatch(const Point3D* query_points, InterpolationResult* results, size_t count);
```

Performs batch interpolation for multiple points.

**Parameters:**
- `query_points`: Array of query points
- `results`: Array to store results
- `count`: Number of query points

**Returns:** Error code

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

Contains magnetic field data at a point.

```cpp
struct MagneticFieldData {
    Real Bx;  // Magnetic field component in x direction
    Real By;  // Magnetic field component in y direction
    Real Bz;  // Magnetic field component in z direction
    // Derivatives of Bx, By, Bz with respect to x, y, z
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

## Error Codes

The library uses error codes instead of exceptions for error handling.

| Error Code | Description |
|------------|-------------|
| `Success` | Operation completed successfully |
| `FileNotFound` | The specified file was not found |
| `FileReadError` | Error reading from file |
| `InvalidFileFormat` | The file format is invalid |
| `InvalidGridData` | The grid data is invalid or inconsistent |
| `MemoryAllocationError` | Memory allocation failed |
| `CudaError` | CUDA-related error |
| `InvalidParameter` | Invalid parameter passed to function |
| `DataNotLoaded` | No data has been loaded |
| `QueryOutOfBounds` | Query point is outside the valid range |
| `CudaNotAvailable` | CUDA is not available |
| `CudaDeviceError` | CUDA device error |

## Usage Examples

### Basic Usage

```cpp
#include "point3d_interp/api.h"

int main() {
    // Create interpolator
    p3d::MagneticFieldInterpolator interp;

    // Load data
    auto err = interp.LoadFromCSV("data.csv");
    if (err != p3d::ErrorCode::Success) {
        // Handle error
        return 1;
    }

    // Query single point
    p3d::Point3D query(1.5, 2.3, 0.8);
    p3d::InterpolationResult result;

    err = interp.Query(query, result);
    if (err == p3d::ErrorCode::Success && result.valid) {
        std::cout << "Bx = " << result.data.Bx << std::endl;
    }

    return 0;
}
```

### Batch Queries

```cpp
#include "point3d_interp/api.h"
#include <vector>

int main() {
    p3d::MagneticFieldInterpolator interp;
    interp.LoadFromCSV("data.csv");

    // Prepare query points
    std::vector<p3d::Point3D> queries = {
        {1.0, 1.0, 1.0},
        {2.0, 2.0, 2.0},
        {3.0, 3.0, 3.0}
    };

    std::vector<p3d::InterpolationResult> results(queries.size());

    // Batch query
    auto err = interp.QueryBatch(queries.data(), results.data(), queries.size());

    return 0;
}
```

### Memory-Based Loading

```cpp
#include "point3d_interp/api.h"
#include <vector>

int main() {
    // Prepare data in memory
    std::vector<p3d::Point3D> points = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}
    };

    std::vector<p3d::MagneticFieldData> field_data(points.size());
    // Fill field_data...

    p3d::MagneticFieldInterpolator interp;
    auto err = interp.LoadFromMemory(
        points.data(),
        field_data.data(),
        points.size()
    );

    return 0;
}
```

## Thread Safety

The `MagneticFieldInterpolator` class is not thread-safe. Create separate instances for concurrent use.

## Performance Considerations

- Use `QueryBatch()` for multiple queries instead of looping with `Query()`
- GPU acceleration provides significant speedup for large datasets
- Memory layout should be contiguous for optimal performance
- Avoid frequent CPU-GPU data transfers
# Hermite Moving Least Squares (HMLS) Implementation Plan

## Overview

This document outlines the plan to implement Hermite Moving Least Squares interpolation for both CPU and GPU in the point3d-interp library, following the existing architecture patterns.

## Background

### What is Hermite Moving Least Squares (HMLS)?

Hermite Moving Least Squares is a meshless interpolation method that:
1. **Moving Least Squares (MLS)**: Constructs local polynomial approximations using weighted least squares fitting
2. **Hermite Interpolation**: Incorporates both function values AND derivative information for improved accuracy
3. **Weighting Function**: Uses distance-based weights (e.g., Gaussian, polynomial) that decrease with distance from query point
4. **Local Approximation**: Builds polynomial approximations locally around each query point

### Why HMLS for This Project?

The project already has:
- **MagneticFieldData** structure with complete derivative information (∂Bx/∂x, ∂Bx/∂y, ∂Bx/∂z, etc.)
- **Unstructured data support** via KD-tree spatial indexing
- **GPU acceleration** infrastructure with CUDA kernels

HMLS will provide:
- **Higher accuracy** than simple IDW by utilizing gradient information
- **Smooth interpolation** with continuous derivatives
- **Better extrapolation** properties near boundaries
- **Compatibility** with existing unstructured data workflows

## Mathematical Foundation

### MLS Approximation

For a query point **q**, we approximate the field using a polynomial:

```
f(x) ≈ p(x) = Σ pᵢ φᵢ(x)
```

where φᵢ(x) are polynomial basis functions (e.g., 1, x, y, z, x², xy, ...) and pᵢ are coefficients.

### Hermite Constraints

The Hermite extension adds derivative constraints:
```
∂f/∂x ≈ ∂p/∂x
∂f/∂y ≈ ∂p/∂y
∂f/∂z ≈ ∂p/∂z
```

### Weighted Least Squares

Minimize the weighted error function:
```
E = Σᵢ wᵢ(||xᵢ - q||) [ (f(xᵢ) - p(xᵢ))² + 
     λ (∂f/∂x(xᵢ) - ∂p/∂x(xᵢ))² + 
     λ (∂f/∂y(xᵢ) - ∂p/∂y(xᵢ))² + 
     λ (∂f/∂z(xᵢ) - ∂p/∂z(xᵢ))² ]
```

where:
- wᵢ is the weight function (typically Gaussian or polynomial)
- λ is the regularization parameter balancing values vs derivatives
- xᵢ are the k-nearest neighbors from the data set

### Weight Functions

**Gaussian Weight**: `w(r) = exp(-r²/h²)`
**Polynomial Weight**: `w(r) = (1 - r/h)⁴₊ · (4r/h + 1)` (Wendland C² function)

where r = ||x - q|| and h is the support radius.

### Polynomial Basis

**Linear**: {1, x, y, z} - 4 terms
**Quadratic**: {1, x, y, z, x², xy, xz, y², yz, z²} - 10 terms
**Cubic**: Full cubic polynomial - 20 terms

## Architecture Design

### Class Structure

Following the existing pattern from [`UnstructuredInterpolator`](../include/point3d_interp/unstructured_interpolator.h):

```
HermiteMLSInterpolator
├── Constructor (coordinates, field_data, parameters)
├── Query methods (single, batch)
├── Private helper methods
│   ├── computeWeights()
│   ├── constructPolynomialBasis()
│   ├── buildLeastSquaresSystem()
│   ├── solveLeastSquares()
│   └── evaluatePolynomial()
└── Member variables
    ├── coordinates_
    ├── field_data_
    ├── kd_tree_ (for k-NN search)
    ├── parameters_ (polynomial order, weight function, support radius, etc.)
    └── bounds_
```

### Integration Points

Following the existing architecture in [`types.h`](../include/point3d_interp/types.h) and [`interpolator_interface.h`](../include/point3d_interp/interpolator_interface.h):

1. **Add to InterpolationMethod enum** (types.h):
   ```cpp
   enum class InterpolationMethod { 
       Trilinear, 
       TricubicHermite, 
       IDW,
       HermiteMLS  // NEW
   };
   ```

2. **Create Adapter Classes** (following [`interpolator_adapters.h`](../include/point3d_interp/interpolator_adapters.h)):
   - `CPUHermiteMLSInterpolatorAdapter`
   - `GPUHermiteMLSInterpolatorAdapter`

3. **Factory Registration** (following [`interpolator_factory.h`](../include/point3d_interp/interpolator_factory.h)):
   - Register in `DefaultInterpolatorFactory`
   - Support both CPU and GPU variants

## Implementation Steps

### Phase 1: Core Algorithm (CPU)

#### 1.1 Create Header File
**File**: `include/point3d_interp/hermite_mls_interpolator.h`

```cpp
class HermiteMLSInterpolator {
public:
    enum class WeightFunction { Gaussian, Wendland };
    enum class BasisOrder { Linear, Quadratic, Cubic };
    
    struct Parameters {
        BasisOrder basis_order = BasisOrder::Quadratic;
        WeightFunction weight_function = WeightFunction::Gaussian;
        Real support_radius = 2.0;
        Real derivative_weight = 1.0;  // λ parameter
        size_t max_neighbors = 20;
    };
    
    HermiteMLSInterpolator(
        const std::vector<Point3D>& coordinates,
        const std::vector<MagneticFieldData>& field_data,
        const Parameters& params = Parameters()
    );
    
    InterpolationResult query(const Point3D& query_point) const;
    std::vector<InterpolationResult> queryBatch(const std::vector<Point3D>& query_points) const;
    
private:
    // Weight function computation
    Real computeWeight(Real distance, Real support_radius) const;
    
    // Polynomial basis evaluation
    size_t getBasisSize() const;
    void evaluateBasis(const Point3D& point, const Point3D& center, std::vector<Real>& basis) const;
    void evaluateBasisDerivatives(const Point3D& point, const Point3D& center, 
                                  std::vector<Real>& dx, std::vector<Real>& dy, 
                                  std::vector<Real>& dz) const;
    
    // System construction and solving
    MagneticFieldData solveMLSSystem(const Point3D& query_point,
                                     const std::vector<size_t>& neighbor_indices,
                                     const std::vector<Real>& distances) const;
    
    // Member variables
    std::vector<Point3D> coordinates_;
    std::vector<MagneticFieldData> field_data_;
    Parameters params_;
    std::unique_ptr<KDTree> kd_tree_;
    Point3D min_bound_, max_bound_;
};
```

#### 1.2 Implement CPU Version
**File**: `src/hermite_mls_interpolator.cpp`

Key implementation details:
- Use Eigen or custom linear algebra for solving least squares systems
- Implement efficient matrix assembly using k-NN from KD-tree
- Support OpenMP parallelization for batch queries
- Implement both Gaussian and Wendland weight functions
- Support linear, quadratic, and cubic polynomial bases

### Phase 2: GPU Implementation

#### 2.1 CUDA Kernel Design
**File**: `src/cuda_hermite_mls_interpolator.cu`

Kernel structure:
```cuda
__global__ void hermiteMLSKernel(
    const Point3D* query_points,
    const Point3D* data_points,
    const MagneticFieldData* field_data,
    const uint32_t* cell_offsets,
    const uint32_t* cell_points,
    const SpatialGrid spatial_grid,
    InterpolationResult* results,
    size_t num_queries,
    HermiteMLSParameters params
);
```

GPU optimizations:
- Use shared memory for neighbor data
- Parallelize matrix assembly across threads
- Use cuBLAS/cuSOLVER for linear algebra operations OR custom QR decomposition
- Coalesced memory access patterns
- Spatial grid for efficient neighbor finding (already exists in project)

#### 2.2 GPU Memory Management
Following [`memory_manager.h`](../include/point3d_interp/memory_manager.h):
- Use `cuda::GpuMemory<T>` for device memory
- Transfer data points and field data to GPU
- Maintain spatial grid on device

### Phase 3: Adapter Implementation

#### 3.1 CPU Adapter
**File**: `src/interpolator_adapters.cu` (add to existing)

```cpp
class CPUHermiteMLSInterpolatorAdapter : public IInterpolator {
public:
    CPUHermiteMLSInterpolatorAdapter(
        std::unique_ptr<HermiteMLSInterpolator> interpolator,
        InterpolationMethod method,
        ExtrapolationMethod extrapolation
    );
    
    // Implement IInterpolator interface
    InterpolationResult query(const Point3D& point) const override;
    std::vector<InterpolationResult> queryBatch(const std::vector<Point3D>& points) const override;
    bool supportsGPU() const override { return false; }
    DataStructureType getDataType() const override { return DataStructureType::Unstructured; }
    // ... other interface methods
    
private:
    std::unique_ptr<HermiteMLSInterpolator> hmls_interpolator_;
    InterpolationMethod method_;
    ExtrapolationMethod extrapolation_;
};
```

#### 3.2 GPU Adapter
Similar structure but with GPU resources:
```cpp
class GPUHermiteMLSInterpolatorAdapter : public IInterpolator {
    // Similar to GPUUnstructuredInterpolatorAdapter
    cuda::GpuMemory<Point3D> d_points_;
    cuda::GpuMemory<MagneticFieldData> d_field_data_;
    SpatialGrid spatial_grid_;
    // ... GPU-specific members
};
```

### Phase 4: Factory Registration

**File**: `src/interpolator_factory.cu` (modify existing)

```cpp
std::unique_ptr<IInterpolator> DefaultInterpolatorFactory::createInterpolator(
    DataStructureType dataType,
    InterpolationMethod method,
    const std::vector<Point3D>& coordinates,
    const std::vector<MagneticFieldData>& fieldData,
    ExtrapolationMethod extrapolation,
    bool useGPU
) {
    // ... existing code ...
    
    if (method == InterpolationMethod::HermiteMLS && 
        dataType == DataStructureType::Unstructured) {
        
        HermiteMLSInterpolator::Parameters params;
        // Set default parameters or read from configuration
        
        auto hmls = std::make_unique<HermiteMLSInterpolator>(
            coordinates, fieldData, params
        );
        
        if (useGPU) {
            return std::make_unique<GPUHermiteMLSInterpolatorAdapter>(
                std::move(hmls), method, extrapolation
            );
        } else {
            return std::make_unique<CPUHermiteMLSInterpolatorAdapter>(
                std::move(hmls), method, extrapolation
            );
        }
    }
    
    // ... rest of code ...
}
```

### Phase 5: Testing

#### 5.1 Unit Tests
**File**: `tests/test_hermite_mls_interpolator.cpp`

Test cases:
1. **Basic functionality**
   - Constructor with valid data
   - Single point query
   - Batch query
   
2. **Accuracy tests**
   - Known polynomial functions (linear, quadratic)
   - Comparison with analytical gradients
   - Cross-validation against original data points
   
3. **Edge cases**
   - Minimum number of neighbors
   - Boundary points
   - Degenerate configurations
   
4. **Parameter variations**
   - Different polynomial orders
   - Different weight functions
   - Different support radii
   
5. **CPU vs GPU consistency**
   - Same results from CPU and GPU implementations
   - Numerical tolerance checks

#### 5.2 Accuracy Comparison Tests
**File**: `tests/test_hmls_accuracy.cpp`

Compare HMLS against:
- IDW interpolation
- Tricubic Hermite (for regular grids)
- Analytical test functions

Metrics:
- L2 error norm
- Maximum error
- Gradient accuracy

#### 5.3 Benchmark Tests
**Files**: `tests/benchmark_hmls_*.cpp`

Following existing benchmark structure:
- `benchmark_hmls_inside_domain_1000.cpp`
- `benchmark_hmls_inside_domain_8000.cpp`
- `benchmark_hmls_inside_domain_27000.cpp`
- `benchmark_hmls_inside_domain_125000.cpp`

Measure:
- CPU execution time
- GPU execution time
- Speedup factors
- Memory usage

### Phase 6: Documentation

#### 6.1 Update README.md
Add HMLS to features list and quick start example:

```cpp
// Create interpolator with HMLS method
MagneticFieldInterpolator interp(true, 0, InterpolationMethod::HermiteMLS);
interp.LoadFromCSV("unstructured_data.csv");

// Query
Point3D query_point(1.5, 2.3, 0.8);
InterpolationResult result;
interp.Query(query_point, result);
```

#### 6.2 Update API_Reference.md
Add comprehensive HMLS documentation:
- Algorithm description
- Parameter tuning guidelines
- Performance characteristics
- Use case recommendations

#### 6.3 Add Performance Guide Section
Document HMLS performance:
- Benchmark results
- Comparison with IDW
- Scalability with data size
- GPU acceleration benefits

## File Structure Summary

### New Files
```
include/point3d_interp/
├── hermite_mls_interpolator.h         # HMLS CPU interpolator header

src/
├── hermite_mls_interpolator.cpp       # HMLS CPU implementation
├── cuda_hermite_mls_interpolator.cu   # HMLS GPU kernel implementation

tests/
├── test_hermite_mls_interpolator.cpp  # Unit tests
├── test_hmls_accuracy.cpp             # Accuracy comparison tests
├── benchmark_hmls_inside_domain_1000.cpp
├── benchmark_hmls_inside_domain_8000.cpp
├── benchmark_hmls_inside_domain_27000.cpp
└── benchmark_hmls_inside_domain_125000.cpp

plans/
└── hmls_implementation_plan.md        # This document
```

### Modified Files
```
include/point3d_interp/
├── types.h                            # Add HermiteMLS to enum
├── interpolator_adapters.h            # Add HMLS adapter classes

src/
├── interpolator_adapters.cu           # Implement HMLS adapters
├── interpolator_factory.cu            # Register HMLS in factory
├── interpolator_api.cu                # Update API if needed

tests/
└── CMakeLists.txt                     # Add new test executables

CMakeLists.txt                         # Add new source files
README.md                              # Update documentation
docs/API_Reference.md                  # Add HMLS documentation
```

## Implementation Timeline

### Detailed Task Breakdown

1. ✅ **Understand HMLS algorithm** - Mathematical foundation documented above
2. ✅ **Design architecture** - Class structure and integration points defined
3. **Implement CPU version**
   - Create header file with class declaration
   - Implement polynomial basis functions
   - Implement weight functions
   - Implement least squares system construction
   - Implement query methods
   - Add unit tests for CPU implementation
4. **Implement GPU version**
   - Design CUDA kernel structure
   - Implement device functions for basis and weights
   - Implement GPU kernel with spatial grid
   - Add memory management
   - Add unit tests for GPU implementation
5. **Integration**
   - Add to InterpolationMethod enum
   - Create adapter classes
   - Register in factory
   - Update API
   - Integration tests
6. **Testing & Validation**
   - Comprehensive unit tests
   - Accuracy comparison tests
   - Benchmark tests
   - Cross-validation with existing methods
7. **Documentation**
   - Update README
   - Update API reference
   - Add usage examples
   - Document performance characteristics

## Configuration Parameters

### Recommended Default Values

```cpp
HermiteMLSInterpolator::Parameters defaults = {
    .basis_order = BasisOrder::Quadratic,      // Good balance of accuracy/performance
    .weight_function = WeightFunction::Gaussian, // Smooth, well-conditioned
    .support_radius = 2.0,                     // Project-specific, may need tuning
    .derivative_weight = 1.0,                  // Equal weight to values and derivatives
    .max_neighbors = 20                        // Sufficient for quadratic basis
};
```

### Parameter Tuning Guidelines

**basis_order**:
- Linear: Fast, lower accuracy, needs ~4-8 neighbors
- Quadratic: Balanced, needs ~10-20 neighbors
- Cubic: High accuracy, needs ~20-40 neighbors

**support_radius**:
- Too small: Insufficient neighbors, ill-conditioned system
- Too large: Over-smoothing, loss of local features
- Rule of thumb: 1.5-3.0 times average point spacing

**derivative_weight** (λ):
- < 1.0: Prioritize function values
- = 1.0: Equal importance
- > 1.0: Prioritize derivatives (better for gradient accuracy)

**max_neighbors**:
- Must be >= basis_size for well-conditioned system
- Recommended: 2-3× basis_size for robustness

## Expected Outcomes

### Accuracy Improvements
- 2-5× better accuracy than IDW for smooth fields
- Exact reproduction of polynomials up to the basis order
- Improved derivative accuracy compared to IDW

### Performance Characteristics
- CPU: O(k³) per query (k = max_neighbors, due to linear algebra)
- GPU: Massive parallelization for batch queries
- Expected GPU speedup: 10-100× for large batches (similar to IDW benchmarks)

### Memory Requirements
- CPU: Similar to IDW (coordinates + field_data + KD-tree)
- GPU: Additional space for spatial grid and temporary matrices

## References

1. Levin, D. (2003). "Mesh-Independent Surface Interpolation"
2. Wendland, H. (2004). "Scattered Data Approximation"
3. Lancaster, P. & Salkauskas, K. (1981). "Surfaces Generated by Moving Least Squares Methods"
4. Belytschko, T. et al. (1996). "Element-Free Galerkin Methods" - HMLS in computational mechanics

## Notes

- The existing [`KDTree`](../include/point3d_interp/kd_tree.h) class will be reused for efficient k-NN search
- The existing [`SpatialGrid`](../include/point3d_interp/spatial_grid.h) can be used for GPU neighbor finding
- Consider using Eigen library for CPU linear algebra (already used in similar projects)
- For GPU, implement custom QR decomposition or use cuSOLVER for batch operations
- Maintain backward compatibility - HMLS is additive, doesn't modify existing code

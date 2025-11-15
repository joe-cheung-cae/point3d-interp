# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2025-11-15] - IDW Algorithm Accuracy Verification and Point3D Enhancement

### Added
- **IDW Accuracy Tests**: Comprehensive test suite for inverse distance weighting interpolation accuracy
  - `AccuracyTestSimple`: Verifies exact IDW calculations with manually computed expected values
  - `AccuracyTestDifferentPower`: Tests IDW behavior with different power parameters (p=1, p=3)
  - `AccuracyTestKNearestNeighbors`: Validates k-nearest neighbor functionality with controlled test cases

- **Point3D Division Operator**: Added `operator/` for scalar division of Point3D coordinates
  - Enables expressions like `point / scalar` for coordinate-wise division
  - Maintains consistency with existing `operator*` for scalar multiplication

### Changed
- **Test Coverage**: Enhanced `tests/test_unstructured_interpolator.cpp` with rigorous numerical accuracy verification
- **Type System**: Extended Point3D arithmetic operations for better usability

### Technical Details
- **Modified Files**: `include/point3d_interp/types.h`, `tests/test_unstructured_interpolator.cpp`
- **New Tests**: 3 additional accuracy tests for IDW interpolation
- **Verification**: All tests pass with precise numerical validation

### Verified
- ✅ All existing tests continue to pass (100% backward compatibility)
- ✅ New accuracy tests validate IDW algorithm correctness
- ✅ Code compiles successfully on all platforms

## [2025-11-14] - Unstructured Data Support and GPU Acceleration

### Added
- **Unstructured Point Cloud Interpolation**: Added support for non-regular grid data using inverse distance weighting (IDW)
  - New `UnstructuredInterpolator` class for scattered 3D point cloud interpolation
  - IDW algorithm with configurable power parameter (default 2.0)
  - Support for k-nearest neighbors optimization for large datasets
  - Automatic detection of unstructured data during loading

- **GPU Acceleration for Unstructured Data**: Extended CUDA support to unstructured point clouds
  - New `IDWInterpolationKernel` CUDA kernel for parallel IDW computation
  - GPU memory management for unstructured data points and field values
  - Seamless integration with existing GPU infrastructure

- **Automatic Data Type Detection**: Enhanced API with intelligent data structure recognition
  - Automatic switching between regular grid and unstructured interpolation
  - Maintains backward compatibility with existing regular grid data
  - Unified API interface for both data types

### Changed
- **API Enhancement**: Extended `MagneticFieldInterpolator` to support both structured and unstructured data
  - Added `IDW` interpolation method to `InterpolationMethod` enum
  - Enhanced `LoadFromMemory` with automatic data type detection
  - Updated GPU memory management to handle both data types

- **Documentation**: Comprehensive updates across all documentation
  - README.md: Added unstructured data support description and GPU acceleration details
  - API_Reference.md: Updated performance considerations and data format specifications
  - development_plan.md: Added complete implementation details

### Technical Details
- **New Files**: `include/point3d_interp/unstructured_interpolator.h`, `src/unstructured_interpolator.cpp`
- **Modified Files**: `include/point3d_interp/types.h`, `src/api.cpp`, `src/cuda_interpolator.cu`, `CMakeLists.txt`
- **New Tests**: `tests/test_unstructured_interpolator.cpp`, GPU unstructured interpolation tests
- **GPU Support**: Added CUDA kernel for IDW interpolation with parallel query processing
- **Memory Management**: Extended GPU memory classes to support unstructured data allocation and transfer

### Verified
- ✅ Code compiles successfully on all platforms
- ✅ All existing tests pass (100% backward compatibility)
- ✅ New unstructured interpolation tests pass (14/14)
- ✅ GPU acceleration verified for both data types
- ✅ CPU/GPU consistency maintained for regular grids
- ✅ Automatic data type detection working correctly
- ✅ Performance benchmarks show expected GPU speedup

## [2025-11-14] - Interpolation Algorithm Enhancement

### Fixed
- **CPU Interpolator**: Fixed tricubic Hermite interpolation algorithm to properly compute spatial derivatives
  - Previously, derivatives in interpolation results were incorrectly set to 0
  - Added `hermiteDerivative` function for computing derivatives of Hermite polynomials
  - Rewrote `tricubicHermiteInterpolate` to compute derivatives at each interpolation step
  - Added proper scaling of derivatives by grid spacing to convert from grid to world coordinates
  - Updated class documentation from "trilinear" to "tricubic Hermite"

- **CUDA Interpolator**: Fixed incomplete tricubic Hermite interpolation implementation
  - Previously, only computed field values but used incorrect (zero) derivatives in interpolation steps
  - Added `__device__ HermiteDerivative` function for GPU Hermite derivative computation
  - Completely rewrote `TricubicHermiteInterpolate` to match CPU implementation with proper intermediate derivative calculations
  - Added linear interpolation of cross-derivatives and proper scaling to world coordinates
  - Ensured CUDA results now match CPU implementation exactly

### Changed
- **Documentation**: Updated README.md, API_Reference.md, and development_plan.md to reflect tricubic Hermite interpolation
- **API**: MagneticFieldData now provides correctly computed spatial derivatives

### Technical Details
- Modified files: `src/cpu_interpolator.cpp`, `include/point3d_interp/cpu_interpolator.h`, `src/cuda_interpolator.cu`
- Added complete derivative calculation logic for tricubic Hermite interpolation on both CPU and GPU
- Ensured derivatives are scaled appropriately for world coordinate system
- CUDA implementation now matches CPU exactly with proper intermediate derivative handling
- All existing tests pass with enhanced functionality

### Verified
- ✅ Code compiles successfully
- ✅ All CPU interpolator tests pass (7/7)
- ✅ All GPU interpolator tests pass (13/13)
- ✅ All accuracy tests pass (5/5)
- ✅ CPU/GPU consistency verified with zero error
- ✅ Backward compatibility maintained
- ✅ Documentation updated consistently
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
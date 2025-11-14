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

### Changed
- **Documentation**: Updated README.md, API_Reference.md, and development_plan.md to reflect tricubic Hermite interpolation
- **API**: MagneticFieldData now provides correctly computed spatial derivatives

### Technical Details
- Modified files: `src/cpu_interpolator.cpp`, `include/point3d_interp/cpu_interpolator.h`
- Added complete derivative calculation logic for tricubic Hermite interpolation
- Ensured derivatives are scaled appropriately for world coordinate system
- All existing tests pass with enhanced functionality

### Verified
- ✅ Code compiles successfully
- ✅ All CPU interpolator tests pass (7/7)
- ✅ Backward compatibility maintained
- ✅ Documentation updated consistently
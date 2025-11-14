#ifndef POINTER3D_INTERP_CPU_INTERPOLATOR_H
#define POINTER3D_INTERP_CPU_INTERPOLATOR_H

#include "types.h"
#include "grid_structure.h"
#include <vector>

namespace p3d {

/**
 * @brief CPU-side tricubic Hermite interpolator
 *
 * Provides CPU version of tricubic Hermite interpolation implementation for verifying GPU version correctness
 */
class CPUInterpolator {
  public:
    /**
     * @brief Constructor
     * @param grid Regular grid object
     */
    explicit CPUInterpolator(const RegularGrid3D& grid);

    ~CPUInterpolator();

    // Disable copy, allow move
    CPUInterpolator(const CPUInterpolator&)            = delete;
    CPUInterpolator& operator=(const CPUInterpolator&) = delete;
    CPUInterpolator(CPUInterpolator&&) noexcept;
    CPUInterpolator& operator=(CPUInterpolator&&) noexcept;

    /**
     * @brief Single point interpolation query
     * @param query_point Query point coordinates
     * @return Interpolation result
     */
    InterpolationResult query(const Point3D& query_point) const;

    /**
     * @brief Batch interpolation query
     * @param query_points Query point array
     * @return Interpolation result array
     */
    std::vector<InterpolationResult> queryBatch(const std::vector<Point3D>& query_points) const;

    /**
     * @brief Get grid reference
     * @return Constant reference to grid object
     */
    const RegularGrid3D& getGrid() const { return *grid_ptr_; }

  private:
    /**
     * @brief 1D Hermite interpolation
     * @param f0 Value at 0
     * @param f1 Value at 1
     * @param df0 Derivative at 0
     * @param df1 Derivative at 1
     * @param t Parameter (0-1)
     * @return Interpolated value
     */
    Real hermiteInterpolate(Real f0, Real f1, Real df0, Real df1, Real t) const;

    /**
     * @brief Perform tricubic Hermite interpolation calculation
     * @param vertex_data Magnetic field data of 8 vertices
     * @param tx Local coordinate in x direction (0-1)
     * @param ty Local coordinate in y direction (0-1)
     * @param tz Local coordinate in z direction (0-1)
     * @return Interpolation result
     */
    MagneticFieldData tricubicHermiteInterpolate(const MagneticFieldData vertex_data[8], Real tx, Real ty,
                                                 Real tz) const;

    /**
     * @brief Get cell vertex data
     * @param indices 8 vertex indices
     * @param vertex_data Output vertex data array
     */
    void getVertexData(const uint32_t indices[8], MagneticFieldData vertex_data[8]) const;

    /**
     * @brief 1D Hermite interpolation derivative
     * @param f0 Value at 0
     * @param f1 Value at 1
     * @param df0 Derivative at 0
     * @param df1 Derivative at 1
     * @param t Parameter (0-1)
     * @return Derivative of interpolated value
     */
    Real hermiteDerivative(Real f0, Real f1, Real df0, Real df1, Real t) const;

    const RegularGrid3D* grid_ptr_;  // Grid pointer
};

}  // namespace p3d

#endif  // POINTER3D_INTERP_CPU_INTERPOLATOR_H
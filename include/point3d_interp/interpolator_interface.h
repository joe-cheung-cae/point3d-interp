#ifndef POINTER3D_INTERP_INTERPOLATOR_INTERFACE_H
#define POINTER3D_INTERP_INTERPOLATOR_INTERFACE_H

#include "types.h"
#include <vector>
#include <memory>

namespace p3d {

/**
 * @brief Data structure type enumeration
 */
enum class DataStructureType { RegularGrid, Unstructured };

/**
 * @brief Abstract interpolator interface
 *
 * This interface defines the common contract for all interpolator implementations,
 * enabling polymorphic usage and decoupling algorithm implementations from the API layer.
 */
class IInterpolator {
public:
    virtual ~IInterpolator() = default;

    /**
     * @brief Single point interpolation query
     * @param point Query point coordinates
     * @return Interpolation result
     */
    virtual InterpolationResult query(const Point3D& point) const = 0;

    /**
     * @brief Batch interpolation query
     * @param points Query point array
     * @return Interpolation result array
     */
    virtual std::vector<InterpolationResult> queryBatch(const std::vector<Point3D>& points) const = 0;

    /**
     * @brief Check if this interpolator supports GPU acceleration
     * @return true if GPU acceleration is supported
     */
    virtual bool supportsGPU() const = 0;

    /**
     * @brief Get the data structure type this interpolator handles
     * @return Data structure type
     */
    virtual DataStructureType getDataType() const = 0;

    /**
     * @brief Get the interpolation method used
     * @return Interpolation method
     */
    virtual InterpolationMethod getMethod() const = 0;

    /**
     * @brief Get the extrapolation method used
     * @return Extrapolation method
     */
    virtual ExtrapolationMethod getExtrapolationMethod() const = 0;

    /**
     * @brief Get number of data points
     * @return Number of data points
     */
    virtual size_t getDataCount() const = 0;

    /**
     * @brief Get coordinate bounds
     * @param min_bound Output minimum bound
     * @param max_bound Output maximum bound
     */
    virtual void getBounds(Point3D& min_bound, Point3D& max_bound) const = 0;

    /**
     * @brief Get grid parameters (for structured data only)
     * @return Grid parameters, or default if not applicable
     */
    virtual GridParams getGridParams() const = 0;

    /**
     * @brief Get all coordinate points
     * @return Vector of coordinate points
     */
    virtual std::vector<Point3D> getCoordinates() const = 0;

    /**
     * @brief Get all magnetic field data
     * @return Vector of magnetic field data
     */
    virtual std::vector<MagneticFieldData> getFieldData() const = 0;
};

/**
 * @brief Abstract factory for creating interpolators
 */
class IInterpolatorFactory {
public:
    virtual ~IInterpolatorFactory() = default;

    /**
     * @brief Create an interpolator instance
     * @param dataType Data structure type
     * @param method Interpolation method
     * @param coordinates Data point coordinates
     * @param fieldData Magnetic field data
     * @param extrapolation Extrapolation method
     * @param useGPU Whether to use GPU acceleration
     * @return Unique pointer to interpolator instance
     */
    virtual std::unique_ptr<IInterpolator> createInterpolator(
        DataStructureType dataType,
        InterpolationMethod method,
        const std::vector<Point3D>& coordinates,
        const std::vector<MagneticFieldData>& fieldData,
        ExtrapolationMethod extrapolation,
        bool useGPU) = 0;

    /**
     * @brief Check if this factory supports the given configuration
     * @param dataType Data structure type
     * @param method Interpolation method
     * @param useGPU Whether GPU is requested
     * @return true if supported
     */
    virtual bool supports(DataStructureType dataType, InterpolationMethod method, bool useGPU) const = 0;
};

}  // namespace p3d

#endif  // POINTER3D_INTERP_INTERPOLATOR_INTERFACE_H
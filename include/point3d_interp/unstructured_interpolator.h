#pragma once

#include "types.h"
#include "kd_tree.h"
#include <vector>
#include <memory>

namespace p3d {

/**
 * @brief CPU-side inverse distance weighting (IDW) interpolator for unstructured point clouds
 *
 * Provides IDW interpolation implementation for scattered 3D data points
 */
class UnstructuredInterpolator {
  public:
    /**
     * @brief Constructor
     * @param coordinates Point coordinates
     * @param field_data Magnetic field data at each point
     * @param power IDW power parameter (default 2.0)
     * @param max_neighbors Maximum number of neighbors to consider (0 = all points)
     * @param extrapolation_method Method for extrapolation outside data bounds (default None)
     */
    UnstructuredInterpolator(const std::vector<Point3D>& coordinates, const std::vector<MagneticFieldData>& field_data,
                             Real power = 2.0, size_t max_neighbors = 0,
                             ExtrapolationMethod extrapolation_method = ExtrapolationMethod::None);

    ~UnstructuredInterpolator();

    // Disable copy, allow move
    UnstructuredInterpolator(const UnstructuredInterpolator&)            = delete;
    UnstructuredInterpolator& operator=(const UnstructuredInterpolator&) = delete;
    UnstructuredInterpolator(UnstructuredInterpolator&&) noexcept;
    UnstructuredInterpolator& operator=(UnstructuredInterpolator&&) noexcept;

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
     * @brief Get number of data points
     * @return Number of data points
     */
    size_t getDataCount() const { return coordinates_.size(); }

    /**
     * @brief Get IDW power parameter
     * @return Power parameter
     */
    Real getPower() const { return power_; }

    /**
     * @brief Get maximum neighbors parameter
     * @return Maximum neighbors
     */
    size_t getMaxNeighbors() const { return max_neighbors_; }

    /**
     * @brief Get extrapolation method
     * @return Extrapolation method
     */
    ExtrapolationMethod getExtrapolationMethod() const { return extrapolation_method_; }

    /**
     * @brief Get all coordinate points
     * @return Array of coordinate points
     */
    const std::vector<Point3D>& getCoordinates() const { return coordinates_; }

    /**
     * @brief Get all magnetic field data
     * @return Array of magnetic field data
     */
    const std::vector<MagneticFieldData>& getFieldData() const { return field_data_; }

    /**
     * @brief Get minimum bound of data
     * @return Minimum bound point
     */
    const Point3D& getMinBound() const { return min_bound_; }

    /**
     * @brief Get maximum bound of data
     * @return Maximum bound point
     */
    const Point3D& getMaxBound() const { return max_bound_; }

  private:
    /**
     * @brief Calculate Euclidean distance between two points
     * @param p1 First point
     * @param p2 Second point
     * @return Distance
     */
    Real distance(const Point3D& p1, const Point3D& p2) const;

    /**
     * @brief Check if point is inside bounding box
     * @param point Query point
     * @return True if inside bounds
     */
    bool isPointInsideBounds(const Point3D& point) const;

    /**
     * @brief Apply extrapolation for points outside bounds
     * @param query_point Query point
     * @return Extrapolated result
     */
    InterpolationResult extrapolate(const Point3D& query_point) const;

    /**
     * @brief Find k nearest neighbors (if max_neighbors > 0)
     * @param query_point Query point
     * @param indices Output array of neighbor indices
     * @param distances Output array of distances
     * @return Number of neighbors found
     */
    size_t findNeighbors(const Point3D& query_point, std::vector<size_t>& indices, std::vector<Real>& distances) const;

    std::vector<Point3D>           coordinates_;           // Point coordinates
    std::vector<MagneticFieldData> field_data_;            // Magnetic field data
    Real                           power_;                 // IDW power parameter
    size_t                         max_neighbors_;         // Maximum neighbors (0 = all)
    ExtrapolationMethod            extrapolation_method_;  // Extrapolation method
    Point3D                        min_bound_;             // Minimum bounds of data
    Point3D                        max_bound_;             // Maximum bounds of data
    std::unique_ptr<KDTree>        kd_tree_;               // KD-tree for spatial indexing
};

}  // namespace p3d

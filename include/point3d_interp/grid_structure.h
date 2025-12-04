#pragma once

#include "types.h"
#include <vector>
#include <memory>

P3D_NAMESPACE_BEGIN

/**
 * @brief Regular grid structure class
 *
 * Manages structural information of 3D regular grid, provides coordinate transformation and index calculation functions
 */
class RegularGrid3D {
  public:
    /**
     * @brief Constructor
     * @param params Grid parameters
     */
    explicit RegularGrid3D(const GridParams& params);

    /**
     * @brief Constructor (automatically build from data points)
     * @param coordinates Coordinate array
     * @param field_data Magnetic field data array
     */
    RegularGrid3D(const std::vector<Point3D>& coordinates, const std::vector<MagneticFieldData>& field_data);

    ~RegularGrid3D();

    // Disable copy, allow move
    RegularGrid3D(const RegularGrid3D&)            = delete;
    RegularGrid3D& operator=(const RegularGrid3D&) = delete;
    RegularGrid3D(RegularGrid3D&&) noexcept;
    RegularGrid3D& operator=(RegularGrid3D&&) noexcept;

    /**
     * @brief Convert world coordinates to grid coordinates
     * @param world_point World coordinate point
     * @return Grid coordinates
     */
    P3D_HOST_DEVICE
    Point3D worldToGrid(const Point3D& world_point) const;

    /**
     * @brief Convert grid coordinates to world coordinates
     * @param grid_point Grid coordinate point
     * @return World coordinates
     */
    P3D_HOST_DEVICE
    Point3D gridToWorld(const Point3D& grid_point) const;

    /**
     * @brief Get 8 vertex indices of grid cell containing the point
     * @param grid_coords Grid coordinates
     * @param indices Output array of 8 vertex indices
     * @return Whether within valid range
     */
    P3D_HOST_DEVICE
    bool getCellVertexIndices(const Point3D& grid_coords, uint32_t indices[8]) const;

    /**
     * @brief Get index of grid data in array
     * @param i Index in x direction
     * @param j Index in y direction
     * @param k Index in z direction
     * @return Array index
     */
    P3D_HOST_DEVICE
    uint32_t getDataIndex(uint32_t i, uint32_t j, uint32_t k) const;

    /**
     * @brief Check if grid coordinates are within valid range
     * @param grid_coords Grid coordinates
     * @return Whether valid
     */
    P3D_HOST_DEVICE
    bool isValidGridCoords(const Point3D& grid_coords) const;

    /**
     * @brief Get grid parameters
     * @return Grid parameters
     */
    const GridParams& getParams() const { return params_; }

    /**
     * @brief Get number of data points
     * @return Number of data points
     */
    size_t getDataCount() const;

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

  private:
    /**
     * @brief Build grid parameters from coordinate data
     * @param coordinates Coordinate array
     */
    void buildFromCoordinates(const std::vector<Point3D>& coordinates);

    /**
     * @brief Validate integrity of grid data
     * @return Whether valid
     */
    bool validateGridData() const;

  private:
    GridParams                     params_;       // Grid parameters
    std::vector<Point3D>           coordinates_;  // Coordinate data
    std::vector<MagneticFieldData> field_data_;   // Magnetic field data
};

P3D_NAMESPACE_END

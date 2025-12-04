#pragma once

#include "types.h"
#include <vector>

P3D_NAMESPACE_BEGIN

/**
 * @brief Build a spatial grid for efficient GPU neighbor finding
 *
 * Divides 3D space into a uniform grid and assigns points to cells for
 * fast spatial queries on GPU.
 *
 * @param points Array of 3D points
 * @param grid_resolutions Desired grid resolution in each dimension (0 = auto)
 * @return Built spatial grid
 */
SpatialGrid buildSpatialGrid(const std::vector<Point3D>&    points,
                             const std::array<uint32_t, 3>& grid_resolutions = {0, 0, 0});

/**
 * @brief Build spatial grid with custom bounds
 *
 * @param points Array of 3D points
 * @param min_bound Minimum bounds of the grid
 * @param max_bound Maximum bounds of the grid
 * @param grid_resolutions Desired grid resolution in each dimension (0 = auto)
 * @return Built spatial grid
 */
SpatialGrid buildSpatialGrid(const std::vector<Point3D>& points, const Point3D& min_bound, const Point3D& max_bound,
                             const std::array<uint32_t, 3>& grid_resolutions = {0, 0, 0});

P3D_NAMESPACE_END

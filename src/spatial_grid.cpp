#include "point3d_interp/spatial_grid.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace p3d {

namespace {

// Helper function to compute bounds from points
void computeBounds(const std::vector<Point3D>& points, Point3D& min_bound, Point3D& max_bound) {
    if (points.empty()) {
        min_bound = Point3D(0, 0, 0);
        max_bound = Point3D(0, 0, 0);
        return;
    }

    min_bound = points[0];
    max_bound = points[0];

    for (const auto& point : points) {
        min_bound.x = std::min(min_bound.x, point.x);
        min_bound.y = std::min(min_bound.y, point.y);
        min_bound.z = std::min(min_bound.z, point.z);
        max_bound.x = std::max(max_bound.x, point.x);
        max_bound.y = std::max(max_bound.y, point.y);
        max_bound.z = std::max(max_bound.z, point.z);
    }
}

// Helper function to determine optimal grid resolution
std::array<uint32_t, 3> determineGridResolution(const std::vector<Point3D>& points, const Point3D& min_bound,
                                                const Point3D&                 max_bound,
                                                const std::array<uint32_t, 3>& requested_resolutions) {
    std::array<uint32_t, 3> resolutions = requested_resolutions;

    // Compute data extents
    Point3D extent = max_bound - min_bound;

    // Target average points per cell
    const size_t target_points_per_cell = 8;

    // If any resolution is 0, auto-determine
    for (int i = 0; i < 3; ++i) {
        if (resolutions[i] == 0) {
            if (extent.x > 0 && extent.y > 0 && extent.z > 0) {
                // Estimate resolution based on cube root of total points / target density
                size_t total_cells = std::max(static_cast<size_t>(1), points.size() / target_points_per_cell);
                Real volume_ratio  = (extent.x * extent.y * extent.z) / (extent.x * extent.y * extent.z);  // normalized

                // Distribute cells proportionally to extent
                Real total_resolution = std::cbrt(static_cast<Real>(total_cells) / volume_ratio);
                resolutions[i] =
                    std::max(static_cast<uint32_t>(1),
                             static_cast<uint32_t>(total_resolution * std::pow(extent.x / extent.y, 1.0 / 3.0)));
            } else {
                resolutions[i] = 8;  // fallback
            }
        }
    }

    // Clamp to reasonable bounds
    for (int i = 0; i < 3; ++i) {
        resolutions[i] = std::max(static_cast<uint32_t>(1), std::min(resolutions[i], static_cast<uint32_t>(128)));
    }

    return resolutions;
}

}  // anonymous namespace

SpatialGrid buildSpatialGrid(const std::vector<Point3D>& points, const std::array<uint32_t, 3>& grid_resolutions) {
    Point3D min_bound, max_bound;
    computeBounds(points, min_bound, max_bound);
    return buildSpatialGrid(points, min_bound, max_bound, grid_resolutions);
}

SpatialGrid buildSpatialGrid(const std::vector<Point3D>& points, const Point3D& min_bound, const Point3D& max_bound,
                             const std::array<uint32_t, 3>& grid_resolutions) {
    SpatialGrid grid;

    if (points.empty()) {
        return grid;
    }

    // Set grid properties
    grid.origin     = min_bound;
    grid.dimensions = determineGridResolution(points, min_bound, max_bound, grid_resolutions);

    // Compute cell sizes
    Point3D extent   = max_bound - min_bound;
    grid.cell_size.x = extent.x / grid.dimensions[0];
    grid.cell_size.y = extent.y / grid.dimensions[1];
    grid.cell_size.z = extent.z / grid.dimensions[2];

    // Avoid division by zero
    if (grid.cell_size.x <= 0) grid.cell_size.x = 1.0f;
    if (grid.cell_size.y <= 0) grid.cell_size.y = 1.0f;
    if (grid.cell_size.z <= 0) grid.cell_size.z = 1.0f;

    size_t num_cells = grid.get_num_cells();
    grid.cell_offsets.resize(num_cells + 1, 0);

    // Count points per cell
    std::vector<size_t> cell_counts(num_cells, 0);
    for (size_t i = 0; i < points.size(); ++i) {
        int ix, iy, iz;
        grid.get_cell_coords(points[i], ix, iy, iz);
        size_t cell_idx = grid.get_cell_index(ix, iy, iz);
        cell_counts[cell_idx]++;
    }

    // Compute offsets
    for (size_t i = 1; i <= num_cells; ++i) {
        grid.cell_offsets[i] = grid.cell_offsets[i - 1] + cell_counts[i - 1];
    }

    // Reset counts for assignment
    std::fill(cell_counts.begin(), cell_counts.end(), 0);
    grid.cell_points.resize(points.size());

    // Assign points to cells
    size_t point_idx = 0;
    for (const auto& point : points) {
        int ix, iy, iz;
        grid.get_cell_coords(point, ix, iy, iz);
        size_t cell_idx          = grid.get_cell_index(ix, iy, iz);
        size_t offset            = grid.cell_offsets[cell_idx] + cell_counts[cell_idx];
        grid.cell_points[offset] = static_cast<uint32_t>(point_idx);
        cell_counts[cell_idx]++;
        point_idx++;
    }

    return grid;
}

}  // namespace p3d
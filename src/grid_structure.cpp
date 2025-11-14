#include "point3d_interp/grid_structure.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_set>

namespace p3d {

RegularGrid3D::RegularGrid3D(const GridParams& params) : params_(params) {
    if (params.dimensions[0] == 0 || params.dimensions[1] == 0 || params.dimensions[2] == 0) {
        throw std::invalid_argument("Invalid grid dimensions");
    }
    params_.update_bounds();

    // Pre-allocate memory
    size_t total_points = static_cast<size_t>(params_.dimensions[0] * params_.dimensions[1] * params_.dimensions[2]);

    coordinates_.reserve(total_points);
    field_data_.reserve(total_points);

    // Generate coordinate points
    for (uint32_t k = 0; k < params_.dimensions[2]; ++k) {
        for (uint32_t j = 0; j < params_.dimensions[1]; ++j) {
            for (uint32_t i = 0; i < params_.dimensions[0]; ++i) {
                Point3D coord(params_.origin.x + i * params_.spacing.x, params_.origin.y + j * params_.spacing.y,
                              params_.origin.z + k * params_.spacing.z);
                coordinates_.push_back(coord);
                field_data_.push_back(MagneticFieldData());  // Default value
            }
        }
    }
}

RegularGrid3D::RegularGrid3D(const std::vector<Point3D>& coordinates, const std::vector<MagneticFieldData>& field_data)
    : coordinates_(coordinates), field_data_(field_data) {
    if (coordinates.size() != field_data.size()) {
        throw std::invalid_argument("Coordinates and field data size mismatch");
    }

    buildFromCoordinates(coordinates);

    if (!validateGridData()) {
        throw std::invalid_argument("Invalid grid data");
    }
}

RegularGrid3D::~RegularGrid3D() = default;

RegularGrid3D::RegularGrid3D(RegularGrid3D&& other) noexcept
    : params_(other.params_), coordinates_(std::move(other.coordinates_)), field_data_(std::move(other.field_data_)) {}

RegularGrid3D& RegularGrid3D::operator=(RegularGrid3D&& other) noexcept {
    if (this != &other) {
        params_      = other.params_;
        coordinates_ = std::move(other.coordinates_);
        field_data_  = std::move(other.field_data_);
    }
    return *this;
}

P3D_HOST_DEVICE
Point3D RegularGrid3D::worldToGrid(const Point3D& world_point) const {
    return Point3D((world_point.x - params_.origin.x) / params_.spacing.x,
                   (world_point.y - params_.origin.y) / params_.spacing.y,
                   (world_point.z - params_.origin.z) / params_.spacing.z);
}

P3D_HOST_DEVICE
Point3D RegularGrid3D::gridToWorld(const Point3D& grid_point) const {
    return Point3D(params_.origin.x + grid_point.x * params_.spacing.x,
                   params_.origin.y + grid_point.y * params_.spacing.y,
                   params_.origin.z + grid_point.z * params_.spacing.z);
}

P3D_HOST_DEVICE
bool RegularGrid3D::getCellVertexIndices(const Point3D& grid_coords, uint32_t indices[8]) const {
    // Get cell starting indices (floor)
    int i0 = static_cast<int>(grid_coords.x);
    int j0 = static_cast<int>(grid_coords.y);
    int k0 = static_cast<int>(grid_coords.z);

    // Handle boundary case: if at the last grid point, use the previous cell
    if (i0 == static_cast<int>(params_.dimensions[0]) - 1) i0--;
    if (j0 == static_cast<int>(params_.dimensions[1]) - 1) j0--;
    if (k0 == static_cast<int>(params_.dimensions[2]) - 1) k0--;

    int i1 = i0 + 1;
    int j1 = j0 + 1;
    int k1 = k0 + 1;

    // Check bounds
    if (i0 < 0 || i1 >= static_cast<int>(params_.dimensions[0]) || j0 < 0 ||
        j1 >= static_cast<int>(params_.dimensions[1]) || k0 < 0 || k1 >= static_cast<int>(params_.dimensions[2])) {
        return false;
    }

    // Calculate indices of 8 vertices
    indices[0] = getDataIndex(i0, j0, k0);  // (i0, j0, k0)
    indices[1] = getDataIndex(i1, j0, k0);  // (i1, j0, k0)
    indices[2] = getDataIndex(i0, j1, k0);  // (i0, j1, k0)
    indices[3] = getDataIndex(i1, j1, k0);  // (i1, j1, k0)
    indices[4] = getDataIndex(i0, j0, k1);  // (i0, j0, k1)
    indices[5] = getDataIndex(i1, j0, k1);  // (i1, j0, k1)
    indices[6] = getDataIndex(i0, j1, k1);  // (i0, j1, k1)
    indices[7] = getDataIndex(i1, j1, k1);  // (i1, j1, k1)

    return true;
}

P3D_HOST_DEVICE
uint32_t RegularGrid3D::getDataIndex(uint32_t i, uint32_t j, uint32_t k) const {
    return i + j * params_.dimensions[0] + k * params_.dimensions[0] * params_.dimensions[1];
}

P3D_HOST_DEVICE
bool RegularGrid3D::isValidGridCoords(const Point3D& grid_coords) const {
    return (grid_coords.x >= 0 && grid_coords.x <= params_.dimensions[0] - 1) &&
           (grid_coords.y >= 0 && grid_coords.y <= params_.dimensions[1] - 1) &&
           (grid_coords.z >= 0 && grid_coords.z <= params_.dimensions[2] - 1);
}

size_t RegularGrid3D::getDataCount() const { return coordinates_.size(); }

void RegularGrid3D::buildFromCoordinates(const std::vector<Point3D>& coordinates) {
    if (coordinates.empty()) {
        throw std::invalid_argument("Empty coordinates");
    }

    // Collect all x, y, z coordinates
    std::unordered_set<Real> x_coords, y_coords, z_coords;

    for (const auto& coord : coordinates) {
        x_coords.insert(coord.x);
        y_coords.insert(coord.y);
        z_coords.insert(coord.z);
    }

    // Convert to ordered vectors
    std::vector<Real> x_unique(x_coords.begin(), x_coords.end());
    std::vector<Real> y_unique(y_coords.begin(), y_coords.end());
    std::vector<Real> z_unique(z_coords.begin(), z_coords.end());

    std::sort(x_unique.begin(), x_unique.end());
    std::sort(y_unique.begin(), y_unique.end());
    std::sort(z_unique.begin(), z_unique.end());

    // Check if it's a regular grid
    if (x_unique.size() < 1 || y_unique.size() < 1 || z_unique.size() < 1) {
        throw std::invalid_argument("Insufficient unique coordinates for grid");
    }

    // Calculate spacing
    auto calculate_spacing = [](const std::vector<Real>& coords) -> Real {
        if (coords.size() < 2) return 1.0f;  // Default spacing for single point
        Real spacing = coords[1] - coords[0];
        for (size_t i = 2; i < coords.size(); ++i) {
            Real current_spacing = coords[i] - coords[i - 1];
            if (std::abs(current_spacing - spacing) > 1e-6) {
                throw std::invalid_argument("Non-uniform grid spacing detected");
            }
        }
        return spacing;
    };

    Real dx = calculate_spacing(x_unique);
    Real dy = calculate_spacing(y_unique);
    Real dz = calculate_spacing(z_unique);

    // Set grid parameters
    params_.origin     = Point3D(x_unique[0], y_unique[0], z_unique[0]);
    params_.spacing    = Point3D(dx, dy, dz);
    params_.dimensions = {static_cast<uint32_t>(x_unique.size()), static_cast<uint32_t>(y_unique.size()),
                          static_cast<uint32_t>(z_unique.size())};
    params_.update_bounds();
}

bool RegularGrid3D::validateGridData() const {
    size_t expected_count = static_cast<size_t>(params_.dimensions[0] * params_.dimensions[1] * params_.dimensions[2]);

    if (coordinates_.size() != expected_count || field_data_.size() != expected_count) {
        return false;
    }

    // Validate each coordinate is correct
    for (uint32_t k = 0; k < params_.dimensions[2]; ++k) {
        for (uint32_t j = 0; j < params_.dimensions[1]; ++j) {
            for (uint32_t i = 0; i < params_.dimensions[0]; ++i) {
                uint32_t       index = getDataIndex(i, j, k);
                const Point3D& coord = coordinates_[index];

                Real expected_x = params_.origin.x + i * params_.spacing.x;
                Real expected_y = params_.origin.y + j * params_.spacing.y;
                Real expected_z = params_.origin.z + k * params_.spacing.z;

                if (std::abs(coord.x - expected_x) > 1e-6 || std::abs(coord.y - expected_y) > 1e-6 ||
                    std::abs(coord.z - expected_z) > 1e-6) {
                    return false;
                }
            }
        }
    }

    return true;
}

}  // namespace p3d
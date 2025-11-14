#include "point3d_interp/unstructured_interpolator.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace p3d {

UnstructuredInterpolator::UnstructuredInterpolator(const std::vector<Point3D>&           coordinates,
                                                   const std::vector<MagneticFieldData>& field_data, Real power,
                                                   size_t max_neighbors)
    : coordinates_(coordinates), field_data_(field_data), power_(power), max_neighbors_(max_neighbors) {
    if (coordinates.size() != field_data.size()) {
        throw std::invalid_argument("Coordinates and field data size mismatch");
    }
    if (coordinates.empty()) {
        throw std::invalid_argument("Empty data set");
    }
    if (power <= 0) {
        throw std::invalid_argument("Power parameter must be positive");
    }
}

UnstructuredInterpolator::~UnstructuredInterpolator() = default;

UnstructuredInterpolator::UnstructuredInterpolator(UnstructuredInterpolator&& other) noexcept
    : coordinates_(std::move(other.coordinates_)),
      field_data_(std::move(other.field_data_)),
      power_(other.power_),
      max_neighbors_(other.max_neighbors_) {}

UnstructuredInterpolator& UnstructuredInterpolator::operator=(UnstructuredInterpolator&& other) noexcept {
    if (this != &other) {
        coordinates_   = std::move(other.coordinates_);
        field_data_    = std::move(other.field_data_);
        power_         = other.power_;
        max_neighbors_ = other.max_neighbors_;
    }
    return *this;
}

InterpolationResult UnstructuredInterpolator::query(const Point3D& query_point) const {
    InterpolationResult result;
    result.valid = false;

    std::vector<size_t> neighbor_indices;
    std::vector<Real>   neighbor_distances;

    size_t num_neighbors = findNeighbors(query_point, neighbor_indices, neighbor_distances);

    if (num_neighbors == 0) {
        return result;  // No valid neighbors
    }

    // Calculate weights and weighted sum
    Real              weight_sum = 0.0;
    MagneticFieldData weighted_sum(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    for (size_t i = 0; i < num_neighbors; ++i) {
        size_t idx  = neighbor_indices[i];
        Real   dist = neighbor_distances[i];

        // Avoid division by zero
        if (dist < std::numeric_limits<Real>::epsilon()) {
            // Exact match - return the data directly
            result.data  = field_data_[idx];
            result.valid = true;
            return result;
        }

        Real weight = 1.0 / std::pow(dist, power_);
        weight_sum += weight;

        // Weight the field data (only Bx, By, Bz for IDW, derivatives remain 0)
        weighted_sum.Bx += field_data_[idx].Bx * weight;
        weighted_sum.By += field_data_[idx].By * weight;
        weighted_sum.Bz += field_data_[idx].Bz * weight;
    }

    // Normalize by weight sum
    if (weight_sum > 0) {
        weighted_sum.Bx /= weight_sum;
        weighted_sum.By /= weight_sum;
        weighted_sum.Bz /= weight_sum;

        // Derivatives are not computed for IDW (set to 0)
        result.data  = weighted_sum;
        result.valid = true;
    }

    return result;
}

std::vector<InterpolationResult> UnstructuredInterpolator::queryBatch(const std::vector<Point3D>& query_points) const {
    std::vector<InterpolationResult> results;
    results.reserve(query_points.size());

    for (const auto& point : query_points) {
        results.push_back(query(point));
    }

    return results;
}

Real UnstructuredInterpolator::distance(const Point3D& p1, const Point3D& p2) const {
    Real dx = p1.x - p2.x;
    Real dy = p1.y - p2.y;
    Real dz = p1.z - p2.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

size_t UnstructuredInterpolator::findNeighbors(const Point3D& query_point, std::vector<size_t>& indices,
                                               std::vector<Real>& distances) const {
    indices.clear();
    distances.clear();

    if (max_neighbors_ == 0) {
        // Use all points
        indices.reserve(coordinates_.size());
        distances.reserve(coordinates_.size());

        for (size_t i = 0; i < coordinates_.size(); ++i) {
            Real dist = distance(query_point, coordinates_[i]);
            indices.push_back(i);
            distances.push_back(dist);
        }
    } else {
        // Find k nearest neighbors
        // Simple implementation: calculate all distances and sort
        // TODO: Optimize with KD-tree for large datasets
        std::vector<std::pair<Real, size_t>> dist_idx_pairs;
        dist_idx_pairs.reserve(coordinates_.size());

        for (size_t i = 0; i < coordinates_.size(); ++i) {
            Real dist = distance(query_point, coordinates_[i]);
            dist_idx_pairs.emplace_back(dist, i);
        }

        // Sort by distance
        std::sort(dist_idx_pairs.begin(), dist_idx_pairs.end());

        // Take first max_neighbors
        size_t num_to_take = std::min(max_neighbors_, dist_idx_pairs.size());
        indices.reserve(num_to_take);
        distances.reserve(num_to_take);

        for (size_t i = 0; i < num_to_take; ++i) {
            distances.push_back(dist_idx_pairs[i].first);
            indices.push_back(dist_idx_pairs[i].second);
        }
    }

    return indices.size();
}

}  // namespace p3d
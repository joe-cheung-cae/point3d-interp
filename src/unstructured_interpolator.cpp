#include "point3d_interp/unstructured_interpolator.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace p3d {

UnstructuredInterpolator::UnstructuredInterpolator(const std::vector<Point3D>&           coordinates,
                                                   const std::vector<MagneticFieldData>& field_data, Real power,
                                                   size_t max_neighbors, ExtrapolationMethod extrapolation_method)
    : coordinates_(coordinates),
      field_data_(field_data),
      power_(power),
      max_neighbors_(max_neighbors),
      extrapolation_method_(extrapolation_method),
      kd_tree_(nullptr) {
    if (coordinates.size() != field_data.size()) {
        throw std::invalid_argument("Coordinates and field data size mismatch");
    }
    if (coordinates.empty()) {
        throw std::invalid_argument("Empty data set");
    }
    if (power <= 0) {
        throw std::invalid_argument("Power parameter must be positive");
    }

    // Compute bounding box
    if (!coordinates.empty()) {
        min_bound_ = coordinates[0];
        max_bound_ = coordinates[0];
        for (const auto& point : coordinates) {
            min_bound_.x = std::min(min_bound_.x, point.x);
            min_bound_.y = std::min(min_bound_.y, point.y);
            min_bound_.z = std::min(min_bound_.z, point.z);
            max_bound_.x = std::max(max_bound_.x, point.x);
            max_bound_.y = std::max(max_bound_.y, point.y);
            max_bound_.z = std::max(max_bound_.z, point.z);
        }
    }

    // Build KD-tree for efficient neighbor finding
    if (!coordinates.empty()) {
        kd_tree_ = std::make_unique<KDTree>(coordinates);
    }
}

UnstructuredInterpolator::~UnstructuredInterpolator() = default;

UnstructuredInterpolator::UnstructuredInterpolator(UnstructuredInterpolator&& other) noexcept
    : coordinates_(std::move(other.coordinates_)),
      field_data_(std::move(other.field_data_)),
      power_(other.power_),
      max_neighbors_(other.max_neighbors_),
      extrapolation_method_(other.extrapolation_method_),
      min_bound_(other.min_bound_),
      max_bound_(other.max_bound_),
      kd_tree_(std::move(other.kd_tree_)) {}

UnstructuredInterpolator& UnstructuredInterpolator::operator=(UnstructuredInterpolator&& other) noexcept {
    if (this != &other) {
        coordinates_          = std::move(other.coordinates_);
        field_data_           = std::move(other.field_data_);
        power_                = other.power_;
        max_neighbors_        = other.max_neighbors_;
        extrapolation_method_ = other.extrapolation_method_;
        min_bound_            = other.min_bound_;
        max_bound_            = other.max_bound_;
    }
    return *this;
}

InterpolationResult UnstructuredInterpolator::query(const Point3D& query_point) const {
    // Check if point is outside bounds and extrapolation is needed
    if (extrapolation_method_ != ExtrapolationMethod::None && !isPointInsideBounds(query_point)) {
        return extrapolate(query_point);
    }

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
        // Use all points - fall back to brute force for this case
        indices.reserve(coordinates_.size());
        distances.reserve(coordinates_.size());

        for (size_t i = 0; i < coordinates_.size(); ++i) {
            Real dist = distance(query_point, coordinates_[i]);
            indices.push_back(i);
            distances.push_back(dist);
        }
    } else {
        // Use KD-tree for efficient k-nearest neighbor search
        if (!kd_tree_) {
            return 0;
        }
        kd_tree_->findKNearestNeighbors(query_point, max_neighbors_, indices, distances);
    }

    return indices.size();
}

bool UnstructuredInterpolator::isPointInsideBounds(const Point3D& point) const {
    return (point.x >= min_bound_.x && point.x <= max_bound_.x) &&
           (point.y >= min_bound_.y && point.y <= max_bound_.y) && (point.z >= min_bound_.z && point.z <= max_bound_.z);
}

InterpolationResult UnstructuredInterpolator::extrapolate(const Point3D& query_point) const {
    InterpolationResult result;
    result.valid = false;

    if (extrapolation_method_ == ExtrapolationMethod::NearestNeighbor) {
        // Find the nearest neighbor
        size_t nearest_idx = 0;
        Real   min_dist    = std::numeric_limits<Real>::max();

        for (size_t i = 0; i < coordinates_.size(); ++i) {
            Real dist = distance(query_point, coordinates_[i]);
            if (dist < min_dist) {
                min_dist    = dist;
                nearest_idx = i;
            }
        }

        result.data  = field_data_[nearest_idx];
        result.valid = true;
    } else if (extrapolation_method_ == ExtrapolationMethod::LinearExtrapolation) {
        // Linear extrapolation using nearest neighbors
        const size_t num_neighbors = std::min(size_t(5), coordinates_.size());  // Use up to 5 nearest neighbors

        // Find nearest neighbors
        std::vector<std::pair<Real, size_t>> dist_idx_pairs;
        dist_idx_pairs.reserve(coordinates_.size());

        for (size_t i = 0; i < coordinates_.size(); ++i) {
            Real dist = distance(query_point, coordinates_[i]);
            dist_idx_pairs.emplace_back(dist, i);
        }

        std::sort(dist_idx_pairs.begin(), dist_idx_pairs.end());
        size_t actual_neighbors = std::min(num_neighbors, dist_idx_pairs.size());

        if (actual_neighbors >= 2) {
            // Use linear extrapolation based on nearest neighbors
            // Simple approach: extrapolate from nearest point using average gradient
            size_t            nearest_idx   = dist_idx_pairs[0].second;
            Point3D           nearest_point = coordinates_[nearest_idx];
            MagneticFieldData nearest_data  = field_data_[nearest_idx];

            // Calculate average gradient from nearest neighbors
            Real   avg_dBx_dx = 0, avg_dBx_dy = 0, avg_dBx_dz = 0;
            Real   avg_dBy_dx = 0, avg_dBy_dy = 0, avg_dBy_dz = 0;
            Real   avg_dBz_dx = 0, avg_dBz_dy = 0, avg_dBz_dz = 0;
            size_t gradient_count = 0;

            for (size_t i = 1; i < actual_neighbors; ++i) {
                size_t            idx = dist_idx_pairs[i].second;
                Point3D           p   = coordinates_[idx];
                MagneticFieldData d   = field_data_[idx];

                Real dx = p.x - nearest_point.x;
                Real dy = p.y - nearest_point.y;
                Real dz = p.z - nearest_point.z;

                if (std::abs(dx) > 1e-8 || std::abs(dy) > 1e-8 || std::abs(dz) > 1e-8) {
                    Real dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                    avg_dBx_dx += (d.Bx - nearest_data.Bx) / dist;
                    avg_dBy_dx += (d.By - nearest_data.By) / dist;
                    avg_dBz_dx += (d.Bz - nearest_data.Bz) / dist;
                    gradient_count++;
                }
            }

            if (gradient_count > 0) {
                avg_dBx_dx /= gradient_count;
                avg_dBy_dx /= gradient_count;
                avg_dBz_dx /= gradient_count;

                // Extrapolate from nearest point
                Real dx   = query_point.x - nearest_point.x;
                Real dy   = query_point.y - nearest_point.y;
                Real dz   = query_point.z - nearest_point.z;
                Real dist = std::sqrt(dx * dx + dy * dy + dz * dz);

                if (dist > 1e-8) {
                    result.data.Bx = nearest_data.Bx + avg_dBx_dx * dist;
                    result.data.By = nearest_data.By + avg_dBy_dx * dist;
                    result.data.Bz = nearest_data.Bz + avg_dBz_dx * dist;
                    result.valid   = true;
                } else {
                    result.data  = nearest_data;
                    result.valid = true;
                }
            } else {
                // Fallback to nearest neighbor if no gradient available
                result.data  = nearest_data;
                result.valid = true;
            }
        } else {
            // Fallback to nearest neighbor
            size_t nearest_idx = dist_idx_pairs[0].second;
            result.data        = field_data_[nearest_idx];
            result.valid       = true;
        }
    }
    // For None, already handled

    return result;
}

}  // namespace p3d
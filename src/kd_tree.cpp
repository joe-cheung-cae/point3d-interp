#include "point3d_interp/kd_tree.h"
#include <algorithm>
#include <cmath>
#include <limits>

P3D_NAMESPACE_BEGIN

KDTree::KDTree(const std::vector<Point3D>& points) : points_(points), root_(nullptr) {
    if (points.empty()) {
        return;
    }

    // Create initial index array
    std::vector<size_t> indices(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        indices[i] = i;
    }

    // Build the tree
    root_ = buildTree(indices, 0);
}

KDTree::~KDTree() { deleteTree(root_); }

KDTree::KDTree(KDTree&& other) noexcept : points_(std::move(other.points_)), root_(other.root_) {
    other.root_ = nullptr;
}

KDTree& KDTree::operator=(KDTree&& other) noexcept {
    if (this != &other) {
        deleteTree(root_);
        points_     = other.points_;
        root_       = other.root_;
        other.root_ = nullptr;
    }
    return *this;
}

size_t KDTree::findKNearestNeighbors(const Point3D& query_point, size_t k, std::vector<size_t>& indices,
                                     std::vector<Real>& distances) const {
    indices.clear();
    distances.clear();

    if (k == 0 || !root_) {
        return 0;
    }

    // Use priority queue to maintain k nearest neighbors (max-heap by distance squared)
    std::priority_queue<std::pair<Real, size_t>> neighbors;

    // Perform KD-tree search
    searchKNearest(root_, query_point, k, 0, neighbors);

    // Extract results from priority queue (reverse order since it's max-heap)
    size_t num_found = neighbors.size();
    indices.resize(num_found);
    distances.resize(num_found);

    for (size_t i = 0; i < num_found; ++i) {
        auto [dist_sq, idx] = neighbors.top();
        neighbors.pop();
        distances[num_found - 1 - i] = std::sqrt(dist_sq);
        indices[num_found - 1 - i]   = idx;
    }

    return indices.size();
}

size_t KDTree::findNeighborsWithinRadius(const Point3D& query_point, Real radius, std::vector<size_t>& indices,
                                         std::vector<Real>& distances) const {
    indices.clear();
    distances.clear();

    if (!root_ || radius <= 0) {
        return 0;
    }

    Real radius_squared = radius * radius;
    searchRadius(root_, query_point, radius_squared, 0, indices, distances);

    return indices.size();
}

KDNode* KDTree::buildTree(const std::vector<size_t>& indices, int depth) {
    if (indices.empty()) {
        return nullptr;
    }

    if (indices.size() == 1) {
        // Leaf node
        return new KDNode(indices[0], -1, 0.0f);
    }

    // Choose splitting dimension
    int split_dim = depth % 3;

    // Find median point along the splitting dimension
    size_t median_idx  = findMedian(indices, split_dim);
    Real   split_value = getCoordinate(median_idx, split_dim);

    KDNode* node = new KDNode(median_idx, split_dim, split_value);

    // Split remaining points into left and right subtrees
    std::vector<size_t> left_indices, right_indices;
    left_indices.reserve(indices.size() / 2);
    right_indices.reserve(indices.size() / 2);

    for (size_t idx : indices) {
        if (idx == median_idx) continue;  // Skip the median point itself

        Real coord = getCoordinate(idx, split_dim);
        if (coord <= split_value) {
            left_indices.push_back(idx);
        } else {
            right_indices.push_back(idx);
        }
    }

    // Recursively build subtrees
    node->left  = buildTree(left_indices, depth + 1);
    node->right = buildTree(right_indices, depth + 1);

    return node;
}

size_t KDTree::findMedian(const std::vector<size_t>& indices, int dim) const {
    // Find the median index without modifying the input vector
    std::vector<std::pair<Real, size_t>> values;
    values.reserve(indices.size());
    for (size_t idx : indices) {
        values.emplace_back(getCoordinate(idx, dim), idx);
    }

    // Use nth_element on the values
    std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
    return values[values.size() / 2].second;
}

void KDTree::searchKNearest(KDNode* node, const Point3D& query_point, size_t k, int depth,
                            std::priority_queue<std::pair<Real, size_t>>& neighbors) const {
    if (!node) {
        return;
    }

    // Calculate distance to current point
    Real dist_sq = squaredDistance(query_point, points_[node->point_index]);

    // Check if this point should be added to neighbors
    if (neighbors.size() < k) {
        neighbors.emplace(dist_sq, node->point_index);
    } else if (dist_sq < neighbors.top().first) {
        neighbors.pop();
        neighbors.emplace(dist_sq, node->point_index);
    }

    // Determine which subtree to search first
    int  split_dim   = node->split_dim;
    Real query_coord = (split_dim == 0) ? query_point.x : (split_dim == 1) ? query_point.y : query_point.z;
    Real split_value = node->split_value;

    KDNode* first_subtree  = (query_coord <= split_value) ? node->left : node->right;
    KDNode* second_subtree = (query_coord <= split_value) ? node->right : node->left;

    // Search first subtree
    searchKNearest(first_subtree, query_point, k, depth + 1, neighbors);

    // Check if we need to search the other subtree
    Real dist_to_plane = (query_coord - split_value) * (query_coord - split_value);
    if (neighbors.size() < k || dist_to_plane < neighbors.top().first) {
        searchKNearest(second_subtree, query_point, k, depth + 1, neighbors);
    }
}

void KDTree::searchRadius(KDNode* node, const Point3D& query_point, Real radius_squared, int depth,
                          std::vector<size_t>& indices, std::vector<Real>& distances) const {
    if (!node) {
        return;
    }

    // Calculate distance to current point
    Real dist_sq = squaredDistance(query_point, points_[node->point_index]);

    if (dist_sq <= radius_squared) {
        indices.push_back(node->point_index);
        distances.push_back(std::sqrt(dist_sq));
    }

    // Determine which subtree to search first
    int  split_dim   = node->split_dim;
    Real query_coord = (split_dim == 0) ? query_point.x : (split_dim == 1) ? query_point.y : query_point.z;
    Real split_value = node->split_value;

    KDNode* first_subtree  = (query_coord <= split_value) ? node->left : node->right;
    KDNode* second_subtree = (query_coord <= split_value) ? node->right : node->left;

    // Search first subtree
    searchRadius(first_subtree, query_point, radius_squared, depth + 1, indices, distances);

    // Check if we need to search the other subtree
    Real dist_to_plane = (query_coord - split_value) * (query_coord - split_value);
    if (dist_to_plane <= radius_squared) {
        searchRadius(second_subtree, query_point, radius_squared, depth + 1, indices, distances);
    }
}

Real KDTree::squaredDistance(const Point3D& p1, const Point3D& p2) {
    Real dx = p1.x - p2.x;
    Real dy = p1.y - p2.y;
    Real dz = p1.z - p2.z;
    return dx * dx + dy * dy + dz * dz;
}

Real KDTree::getCoordinate(size_t point_index, int dim) const {
    const Point3D& point = points_[point_index];
    switch (dim) {
        case 0:
            return point.x;
        case 1:
            return point.y;
        case 2:
            return point.z;
        default:
            return 0.0f;
    }
}

void KDTree::deleteTree(KDNode* node) {
    if (node) {
        deleteTree(node->left);
        deleteTree(node->right);
        delete node;
    }
}

P3D_NAMESPACE_END
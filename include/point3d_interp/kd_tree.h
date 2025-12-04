#pragma once

#include "types.h"
#include <vector>
#include <memory>
#include <queue>

P3D_NAMESPACE_BEGIN

/**
 * @brief KD-tree node structure
 */
struct KDNode {
    size_t  point_index;  // Index of the point in the original array
    int     split_dim;    // Splitting dimension (0=x, 1=y, 2=z)
    Real    split_value;  // Splitting value
    KDNode* left;         // Left subtree
    KDNode* right;        // Right subtree

    KDNode(size_t idx, int dim, Real val)
        : point_index(idx), split_dim(dim), split_value(val), left(nullptr), right(nullptr) {}
};

/**
 * @brief KD-tree for efficient 3D nearest neighbor searches
 *
 * Implements a KD-tree data structure for fast spatial queries on 3D point clouds.
 * Used to accelerate IDW interpolation by quickly finding nearest neighbors.
 */
class KDTree {
  public:
    /**
     * @brief Constructor
     * @param points Array of 3D points to build the tree from
     */
    explicit KDTree(const std::vector<Point3D>& points);

    /**
     * @brief Destructor
     */
    ~KDTree();

    // Disable copy, allow move
    KDTree(const KDTree&)            = delete;
    KDTree& operator=(const KDTree&) = delete;
    KDTree(KDTree&&) noexcept;
    KDTree& operator=(KDTree&&) noexcept;

    /**
     * @brief Find k nearest neighbors to a query point
     * @param query_point The query point
     * @param k Maximum number of neighbors to find
     * @param indices Output array of neighbor indices
     * @param distances Output array of distances to neighbors
     * @return Number of neighbors found
     */
    size_t findKNearestNeighbors(const Point3D& query_point, size_t k, std::vector<size_t>& indices,
                                 std::vector<Real>& distances) const;

    /**
     * @brief Find all neighbors within a given radius
     * @param query_point The query point
     * @param radius Search radius
     * @param indices Output array of neighbor indices
     * @param distances Output array of distances to neighbors
     * @return Number of neighbors found
     */
    size_t findNeighborsWithinRadius(const Point3D& query_point, Real radius, std::vector<size_t>& indices,
                                     std::vector<Real>& distances) const;

  private:
    /**
     * @brief Build the KD-tree recursively
     * @param indices Array of point indices to consider
     * @param depth Current tree depth
     * @return Root node of the subtree
     */
    KDNode* buildTree(const std::vector<size_t>& indices, int depth);

    /**
     * @brief Find the median point along a dimension
     * @param indices Array of point indices
     * @param dim Dimension to split on
     * @return Index of the median point
     */
    size_t findMedian(const std::vector<size_t>& indices, int dim) const;

    /**
     * @brief Recursive k-nearest neighbor search
     * @param node Current tree node
     * @param query_point Query point
     * @param k Number of neighbors to find
     * @param depth Current depth
     * @param neighbors Priority queue of current best neighbors
     */
    void searchKNearest(KDNode* node, const Point3D& query_point, size_t k, int depth,
                        std::priority_queue<std::pair<Real, size_t>>& neighbors) const;

    /**
     * @brief Recursive radius search
     * @param node Current tree node
     * @param query_point Query point
     * @param radius_squared Squared search radius
     * @param depth Current depth
     * @param indices Output neighbor indices
     * @param distances Output neighbor distances
     */
    void searchRadius(KDNode* node, const Point3D& query_point, Real radius_squared, int depth,
                      std::vector<size_t>& indices, std::vector<Real>& distances) const;

    /**
     * @brief Calculate squared distance between two points
     * @param p1 First point
     * @param p2 Second point
     * @return Squared distance
     */
    static Real squaredDistance(const Point3D& p1, const Point3D& p2);

    /**
     * @brief Get coordinate value of a point along a dimension
     * @param point_index Point index
     * @param dim Dimension (0=x, 1=y, 2=z)
     * @return Coordinate value
     */
    Real getCoordinate(size_t point_index, int dim) const;

    /**
     * @brief Delete tree nodes recursively
     * @param node Root node to delete
     */
    void deleteTree(KDNode* node);

    std::vector<Point3D> points_;  // Copy of the point data
    KDNode*              root_;    // Root of the KD-tree
};

P3D_NAMESPACE_END

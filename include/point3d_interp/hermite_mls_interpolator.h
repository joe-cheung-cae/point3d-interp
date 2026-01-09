#pragma once

#include "types.h"
#include "kd_tree.h"
#include <vector>
#include <memory>
#include <cmath>

P3D_NAMESPACE_BEGIN

/**
 * @brief Hermite Moving Least Squares (HMLS) Interpolator
 * 
 * Implements meshless interpolation using Moving Least Squares with Hermite constraints.
 * Incorporates both function values and derivative information for improved accuracy.
 * 
 * Key Features:
 * - Local polynomial approximation at each query point
 * - Utilizes gradient information for better accuracy
 * - Smooth interpolation with continuous derivatives
 * - Supports linear, quadratic, and cubic basis functions
 * - Distance-based weighting (Gaussian or Wendland)
 */
class HermiteMLSInterpolator {
public:
    /**
     * @brief Weight function types for distance-based weighting
     */
    enum class WeightFunction {
        Gaussian,   ///< Gaussian weight: exp(-r^2/h^2)
        Wendland    ///< Wendland C^2: (1-r/h)^4_+ * (4r/h + 1)
    };

    /**
     * @brief Polynomial basis order
     */
    enum class BasisOrder {
        Linear,     ///< Linear basis: {1, x, y, z} - 4 terms
        Quadratic,  ///< Quadratic basis: {1, x, y, z, x^2, xy, xz, y^2, yz, z^2} - 10 terms
        Cubic       ///< Cubic basis: full cubic polynomial - 20 terms
    };

    /**
     * @brief Parameters for HMLS interpolation
     */
    struct Parameters {
        BasisOrder basis_order;           ///< Polynomial basis order
        WeightFunction weight_function;    ///< Weight function type
        Real support_radius;               ///< Support radius for weight function
        Real derivative_weight;            ///< Lambda parameter: weight for derivative constraints
        size_t max_neighbors;              ///< Maximum number of neighbors to use
        Real regularization;               ///< Regularization for ill-conditioned systems
        
        Parameters()
            : basis_order(BasisOrder::Quadratic),
              weight_function(WeightFunction::Gaussian),
              support_radius(static_cast<Real>(2.0)),
              derivative_weight(static_cast<Real>(1.0)),
              max_neighbors(20),
              regularization(static_cast<Real>(1e-8)) {}
    };

    /**
     * @brief Constructor
     * @param coordinates Vector of data point coordinates
     * @param field_data Vector of magnetic field data (values and derivatives)
     * @param params Parameters for HMLS interpolation
     */
    HermiteMLSInterpolator(
        const std::vector<Point3D>& coordinates,
        const std::vector<MagneticFieldData>& field_data,
        const Parameters& params = Parameters()
    );

    /**
     * @brief Query interpolation at a single point
     * @param query_point Query point coordinates
     * @return Interpolation result with field values and derivatives
     */
    InterpolationResult query(const Point3D& query_point) const;

    /**
     * @brief Query interpolation at multiple points (batch query)
     * @param query_points Vector of query point coordinates
     * @return Vector of interpolation results
     */
    std::vector<InterpolationResult> queryBatch(const std::vector<Point3D>& query_points) const;

    /**
     * @brief Get bounds of the data
     * @param min_bound Output: minimum bound
     * @param max_bound Output: maximum bound
     */
    void getBounds(Point3D& min_bound, Point3D& max_bound) const {
        min_bound = min_bound_;
        max_bound = max_bound_;
    }

    /**
     * @brief Get parameters
     */
    const Parameters& getParameters() const { return params_; }

    /**
     * @brief Get data coordinates
     */
    const std::vector<Point3D>& getCoordinates() const { return coordinates_; }

    /**
     * @brief Get field data
     */
    const std::vector<MagneticFieldData>& getFieldData() const { return field_data_; }

private:
    /**
     * @brief Compute weight function value
     * @param distance Distance from query point
     * @param support_radius Support radius
     * @return Weight value
     */
    Real computeWeight(Real distance, Real support_radius) const;

    /**
     * @brief Get number of basis functions for current basis order
     */
    size_t getBasisSize() const;

    /**
     * @brief Evaluate polynomial basis functions at a point
     * @param point Evaluation point
     * @param center Center point (query point)
     * @param basis Output: basis function values
     */
    void evaluateBasis(const Point3D& point, const Point3D& center, std::vector<Real>& basis) const;

    /**
     * @brief Evaluate derivatives of basis functions
     * @param point Evaluation point
     * @param center Center point (query point)
     * @param dx Output: basis derivatives w.r.t. x
     * @param dy Output: basis derivatives w.r.t. y
     * @param dz Output: basis derivatives w.r.t. z
     */
    void evaluateBasisDerivatives(const Point3D& point, const Point3D& center,
                                   std::vector<Real>& dx, std::vector<Real>& dy,
                                   std::vector<Real>& dz) const;

    /**
     * @brief Solve MLS system for a query point
     * @param query_point Query point
     * @param neighbor_indices Indices of k-nearest neighbors
     * @param distances Distances to k-nearest neighbors
     * @param component Field component index (0=Bx, 1=By, 2=Bz)
     * @param coefficients Output: polynomial coefficients
     * @return True if system solved successfully
     */
    bool solveMLSSystemForComponent(const Point3D& query_point,
                                    const std::vector<size_t>& neighbor_indices,
                                    const std::vector<Real>& distances,
                                    int component,
                                    std::vector<Real>& coefficients) const;

    /**
     * @brief Solve MLS system and get interpolated field data
     * @param query_point Query point
     * @param neighbor_indices Indices of k-nearest neighbors
     * @param distances Distances to k-nearest neighbors
     * @return Interpolated magnetic field data
     */
    MagneticFieldData solveMLSSystem(const Point3D& query_point,
                                     const std::vector<size_t>& neighbor_indices,
                                     const std::vector<Real>& distances) const;

    /**
     * @brief Evaluate polynomial at query point given coefficients
     * @param coefficients Polynomial coefficients
     * @param query_point Query point
     * @param center Center point
     * @param value Output: function value
     * @param dx Output: derivative w.r.t. x
     * @param dy Output: derivative w.r.t. y
     * @param dz Output: derivative w.r.t. z
     */
    void evaluatePolynomial(const std::vector<Real>& coefficients,
                           const Point3D& query_point,
                           const Point3D& center,
                           Real& value, Real& dx, Real& dy, Real& dz) const;

    /**
     * @brief Solve least squares system: A^T * A * x = A^T * b
     * Uses QR decomposition for numerical stability
     * @param A System matrix (m x n)
     * @param b Right-hand side vector (m)
     * @param x Output: solution vector (n)
     * @param m Number of rows
     * @param n Number of columns
     * @return True if solved successfully
     */
    bool solveLeastSquares(const std::vector<Real>& A, const std::vector<Real>& b,
                          std::vector<Real>& x, size_t m, size_t n) const;

    // Member variables
    std::vector<Point3D> coordinates_;           ///< Data point coordinates
    std::vector<MagneticFieldData> field_data_;  ///< Field data at each point
    Parameters params_;                          ///< Interpolation parameters
    std::unique_ptr<KDTree> kd_tree_;           ///< KD-tree for k-NN search
    Point3D min_bound_;                          ///< Minimum bounds
    Point3D max_bound_;                          ///< Maximum bounds
};

P3D_NAMESPACE_END

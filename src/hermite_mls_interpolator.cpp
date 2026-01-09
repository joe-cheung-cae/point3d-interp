#include "point3d_interp/hermite_mls_interpolator.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

P3D_NAMESPACE_BEGIN

// Constructor
HermiteMLSInterpolator::HermiteMLSInterpolator(
    const std::vector<Point3D>& coordinates,
    const std::vector<MagneticFieldData>& field_data,
    const Parameters& params)
    : coordinates_(coordinates),
      field_data_(field_data),
      params_(params),
      min_bound_(std::numeric_limits<Real>::max(), 
                 std::numeric_limits<Real>::max(), 
                 std::numeric_limits<Real>::max()),
      max_bound_(std::numeric_limits<Real>::lowest(), 
                 std::numeric_limits<Real>::lowest(), 
                 std::numeric_limits<Real>::lowest()) {
    
    if (coordinates_.empty() || field_data_.empty()) {
        throw std::invalid_argument("Coordinates and field_data cannot be empty");
    }
    
    if (coordinates_.size() != field_data_.size()) {
        throw std::invalid_argument("Coordinates and field_data must have the same size");
    }

    // Build KD-tree for efficient k-NN search
    kd_tree_ = std::make_unique<KDTree>(coordinates_);

    // Compute bounds
    for (const auto& coord : coordinates_) {
        min_bound_.x = std::min(min_bound_.x, coord.x);
        min_bound_.y = std::min(min_bound_.y, coord.y);
        min_bound_.z = std::min(min_bound_.z, coord.z);
        max_bound_.x = std::max(max_bound_.x, coord.x);
        max_bound_.y = std::max(max_bound_.y, coord.y);
        max_bound_.z = std::max(max_bound_.z, coord.z);
    }
}

// Query single point
InterpolationResult HermiteMLSInterpolator::query(const Point3D& query_point) const {
    // Find k-nearest neighbors
    std::vector<size_t> neighbor_indices;
    std::vector<Real> distances;

    kd_tree_->findKNearestNeighbors(query_point, params_.max_neighbors, neighbor_indices, distances);

    if (neighbor_indices.empty()) {
        return InterpolationResult(MagneticFieldData(), false);
    }


    // Solve MLS system
    MagneticFieldData result_data = solveMLSSystem(query_point, neighbor_indices, distances);

    return InterpolationResult(result_data, true);
}

// Query batch
std::vector<InterpolationResult> HermiteMLSInterpolator::queryBatch(
    const std::vector<Point3D>& query_points) const {
    
    std::vector<InterpolationResult> results(query_points.size());
    
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (size_t i = 0; i < query_points.size(); ++i) {
        results[i] = query(query_points[i]);
    }
    
    return results;
}

// Compute weight function
Real HermiteMLSInterpolator::computeWeight(Real distance, Real support_radius) const {
    if (distance >= support_radius) {
        return 0.0;
    }
    
    Real r = distance / support_radius;
    
    switch (params_.weight_function) {
        case WeightFunction::Gaussian:
            // Gaussian: exp(-r^2/h^2) = exp(-r^2) where r is already normalized
            return std::exp(-r * r);
            
        case WeightFunction::Wendland:
            // Wendland C^2: (1-r)^4_+ * (4r + 1)
            {
                Real one_minus_r = 1.0 - r;
                if (one_minus_r <= 0.0) return 0.0;
                Real one_minus_r_sq = one_minus_r * one_minus_r;
                return one_minus_r_sq * one_minus_r_sq * (4.0 * r + 1.0);
            }
            
        default:
            return 1.0;
    }
}

// Get basis size
size_t HermiteMLSInterpolator::getBasisSize() const {
    switch (params_.basis_order) {
        case BasisOrder::Linear:
            return 4;  // {1, x, y, z}
        case BasisOrder::Quadratic:
            return 10; // {1, x, y, z, x^2, xy, xz, y^2, yz, z^2}
        case BasisOrder::Cubic:
            return 20; // Full cubic
        default:
            return 4;
    }
}

// Evaluate basis functions
void HermiteMLSInterpolator::evaluateBasis(const Point3D& point, const Point3D& center,
                                           std::vector<Real>& basis) const {
    Real dx = point.x - center.x;
    Real dy = point.y - center.y;
    Real dz = point.z - center.z;
    
    basis.clear();
    
    // Linear basis: {1, x, y, z}
    basis.push_back(1.0);
    basis.push_back(dx);
    basis.push_back(dy);
    basis.push_back(dz);
    
    if (params_.basis_order == BasisOrder::Linear) {
        return;
    }
    
    // Quadratic terms: {x^2, xy, xz, y^2, yz, z^2}
    basis.push_back(dx * dx);
    basis.push_back(dx * dy);
    basis.push_back(dx * dz);
    basis.push_back(dy * dy);
    basis.push_back(dy * dz);
    basis.push_back(dz * dz);
    
    if (params_.basis_order == BasisOrder::Quadratic) {
        return;
    }
    
    // Cubic terms: {x^3, x^2*y, x^2*z, x*y^2, x*y*z, x*z^2, y^3, y^2*z, y*z^2, z^3}
    basis.push_back(dx * dx * dx);
    basis.push_back(dx * dx * dy);
    basis.push_back(dx * dx * dz);
    basis.push_back(dx * dy * dy);
    basis.push_back(dx * dy * dz);
    basis.push_back(dx * dz * dz);
    basis.push_back(dy * dy * dy);
    basis.push_back(dy * dy * dz);
    basis.push_back(dy * dz * dz);
    basis.push_back(dz * dz * dz);
}

// Evaluate basis derivatives
void HermiteMLSInterpolator::evaluateBasisDerivatives(const Point3D& point, const Point3D& center,
                                                      std::vector<Real>& dx_basis,
                                                      std::vector<Real>& dy_basis,
                                                      std::vector<Real>& dz_basis) const {
    Real dx = point.x - center.x;
    Real dy = point.y - center.y;
    Real dz = point.z - center.z;
    
    dx_basis.clear();
    dy_basis.clear();
    dz_basis.clear();
    
    // Linear basis derivatives
    // d/dx {1, x, y, z} = {0, 1, 0, 0}
    dx_basis.push_back(0.0); dx_basis.push_back(1.0); dx_basis.push_back(0.0); dx_basis.push_back(0.0);
    dy_basis.push_back(0.0); dy_basis.push_back(0.0); dy_basis.push_back(1.0); dy_basis.push_back(0.0);
    dz_basis.push_back(0.0); dz_basis.push_back(0.0); dz_basis.push_back(0.0); dz_basis.push_back(1.0);
    
    if (params_.basis_order == BasisOrder::Linear) {
        return;
    }
    
    // Quadratic basis derivatives
    // d/dx {x^2, xy, xz, y^2, yz, z^2} = {2x, y, z, 0, 0, 0}
    dx_basis.push_back(2.0 * dx); dx_basis.push_back(dy); dx_basis.push_back(dz);
    dx_basis.push_back(0.0); dx_basis.push_back(0.0); dx_basis.push_back(0.0);
    
    // d/dy {x^2, xy, xz, y^2, yz, z^2} = {0, x, 0, 2y, z, 0}
    dy_basis.push_back(0.0); dy_basis.push_back(dx); dy_basis.push_back(0.0);
    dy_basis.push_back(2.0 * dy); dy_basis.push_back(dz); dy_basis.push_back(0.0);
    
    // d/dz {x^2, xy, xz, y^2, yz, z^2} = {0, 0, x, 0, y, 2z}
    dz_basis.push_back(0.0); dz_basis.push_back(0.0); dz_basis.push_back(dx);
    dz_basis.push_back(0.0); dz_basis.push_back(dy); dz_basis.push_back(2.0 * dz);
    
    if (params_.basis_order == BasisOrder::Quadratic) {
        return;
    }
    
    // Cubic basis derivatives
    // d/dx {x^3, x^2*y, x^2*z, x*y^2, x*y*z, x*z^2, y^3, y^2*z, y*z^2, z^3}
    dx_basis.push_back(3.0 * dx * dx); dx_basis.push_back(2.0 * dx * dy); dx_basis.push_back(2.0 * dx * dz);
    dx_basis.push_back(dy * dy); dx_basis.push_back(dy * dz); dx_basis.push_back(dz * dz);
    dx_basis.push_back(0.0); dx_basis.push_back(0.0); dx_basis.push_back(0.0); dx_basis.push_back(0.0);
    
    // d/dy
    dy_basis.push_back(0.0); dy_basis.push_back(dx * dx); dy_basis.push_back(0.0);
    dy_basis.push_back(2.0 * dx * dy); dy_basis.push_back(dx * dz); dy_basis.push_back(0.0);
    dy_basis.push_back(3.0 * dy * dy); dy_basis.push_back(2.0 * dy * dz); dy_basis.push_back(dz * dz); dy_basis.push_back(0.0);
    
    // d/dz
    dz_basis.push_back(0.0); dz_basis.push_back(0.0); dz_basis.push_back(dx * dx);
    dz_basis.push_back(0.0); dz_basis.push_back(dx * dy); dz_basis.push_back(2.0 * dx * dz);
    dz_basis.push_back(0.0); dz_basis.push_back(dy * dy); dz_basis.push_back(2.0 * dy * dz); dz_basis.push_back(3.0 * dz * dz);
}

// Solve least squares using normal equations with regularization
bool HermiteMLSInterpolator::solveLeastSquares(const std::vector<Real>& A,
                                               const std::vector<Real>& b,
                                               std::vector<Real>& x,
                                               size_t m, size_t n) const {
    // Solve: A^T * A * x = A^T * b
    // Where A is m x n, b is m x 1, x is n x 1
    
    x.resize(n, 0.0);
    
    if (m < n) {
        return false;  // Underdetermined system
    }
    
    // Compute A^T * A (n x n matrix)
    std::vector<Real> ATA(n * n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            Real sum = 0.0;
            for (size_t k = 0; k < m; ++k) {
                sum += A[k * n + i] * A[k * n + j];
            }
            ATA[i * n + j] = sum;
            
            // Add regularization to diagonal
            if (i == j) {
                ATA[i * n + j] += params_.regularization;
            }
        }
    }
    
    // Compute A^T * b (n x 1 vector)
    std::vector<Real> ATb(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        Real sum = 0.0;
        for (size_t k = 0; k < m; ++k) {
            sum += A[k * n + i] * b[k];
        }
        ATb[i] = sum;
    }
    
    // Solve ATA * x = ATb using Gaussian elimination
    std::vector<Real> mat = ATA;
    std::vector<Real> rhs = ATb;
    
    // Forward elimination
    for (size_t i = 0; i < n; ++i) {
        // Find pivot
        size_t max_row = i;
        Real max_val = std::abs(mat[i * n + i]);
        for (size_t k = i + 1; k < n; ++k) {
            Real val = std::abs(mat[k * n + i]);
            if (val > max_val) {
                max_val = val;
                max_row = k;
            }
        }
        
        if (max_val < 1e-12) {
            return false;  // Singular matrix
        }
        
        // Swap rows
        if (max_row != i) {
            for (size_t j = 0; j < n; ++j) {
                std::swap(mat[i * n + j], mat[max_row * n + j]);
            }
            std::swap(rhs[i], rhs[max_row]);
        }
        
        // Eliminate
        for (size_t k = i + 1; k < n; ++k) {
            Real factor = mat[k * n + i] / mat[i * n + i];
            for (size_t j = i; j < n; ++j) {
                mat[k * n + j] -= factor * mat[i * n + j];
            }
            rhs[k] -= factor * rhs[i];
        }
    }
    
    // Back substitution
    for (int i = n - 1; i >= 0; --i) {
        x[i] = rhs[i];
        for (size_t j = i + 1; j < n; ++j) {
            x[i] -= mat[i * n + j] * x[j];
        }
        x[i] /= mat[i * n + i];
    }
    
    return true;
}

// Solve MLS system for one component
bool HermiteMLSInterpolator::solveMLSSystemForComponent(
    const Point3D& query_point,
    const std::vector<size_t>& neighbor_indices,
    const std::vector<Real>& distances,
    int component,
    std::vector<Real>& coefficients) const {
    
    size_t k = neighbor_indices.size();
    size_t n = getBasisSize();
    
    // Build system: A * coefficients = b
    // System has 4k rows (k value constraints + 3k derivative constraints)
    size_t m = 4 * k;
    
    std::vector<Real> A(m * n, 0.0);
    std::vector<Real> b(m, 0.0);
    
    std::vector<Real> basis_vals, dx_vals, dy_vals, dz_vals;
    
    Real lambda_sqrt = std::sqrt(params_.derivative_weight);
    
    for (size_t i = 0; i < k; ++i) {
        size_t idx = neighbor_indices[i];
        const Point3D& pt = coordinates_[idx];
        const MagneticFieldData& fd = field_data_[idx];
        
        Real weight = computeWeight(distances[i], params_.support_radius);
        Real w_sqrt = std::sqrt(weight);
        
        // Evaluate basis functions and their derivatives
        evaluateBasis(pt, query_point, basis_vals);
        evaluateBasisDerivatives(pt, query_point, dx_vals, dy_vals, dz_vals);
        
        // Value constraint row: w^0.5 * basis
        for (size_t j = 0; j < n; ++j) {
            A[i * n + j] = w_sqrt * basis_vals[j];
        }
        
        // Get field value for this component
        Real field_value;
        Real df_dx, df_dy, df_dz;
        
        if (component == 0) {  // Bx
            field_value = fd.Bx;
            df_dx = fd.dBx_dx;
            df_dy = fd.dBx_dy;
            df_dz = fd.dBx_dz;
        } else if (component == 1) {  // By
            field_value = fd.By;
            df_dx = fd.dBy_dx;
            df_dy = fd.dBy_dy;
            df_dz = fd.dBy_dz;
        } else {  // Bz
            field_value = fd.Bz;
            df_dx = fd.dBz_dx;
            df_dy = fd.dBz_dy;
            df_dz = fd.dBz_dz;
        }
        
        b[i] = w_sqrt * field_value;
        
        // X-derivative constraint row
        size_t row_dx = k + i;
        for (size_t j = 0; j < n; ++j) {
            A[row_dx * n + j] = w_sqrt * lambda_sqrt * dx_vals[j];
        }
        b[row_dx] = w_sqrt * lambda_sqrt * df_dx;
        
        // Y-derivative constraint row
        size_t row_dy = 2 * k + i;
        for (size_t j = 0; j < n; ++j) {
            A[row_dy * n + j] = w_sqrt * lambda_sqrt * dy_vals[j];
        }
        b[row_dy] = w_sqrt * lambda_sqrt * df_dy;
        
        // Z-derivative constraint row
        size_t row_dz = 3 * k + i;
        for (size_t j = 0; j < n; ++j) {
            A[row_dz * n + j] = w_sqrt * lambda_sqrt * dz_vals[j];
        }
        b[row_dz] = w_sqrt * lambda_sqrt * df_dz;
    }
    
    // Solve least squares system
    return solveLeastSquares(A, b, coefficients, m, n);
}

// Evaluate polynomial
void HermiteMLSInterpolator::evaluatePolynomial(const std::vector<Real>& coefficients,
                                                const Point3D& query_point,
                                                const Point3D& center,
                                                Real& value, Real& dx, Real& dy, Real& dz) const {
    std::vector<Real> basis_vals, dx_vals, dy_vals, dz_vals;
    
    evaluateBasis(query_point, center, basis_vals);
    evaluateBasisDerivatives(query_point, center, dx_vals, dy_vals, dz_vals);
    
    value = 0.0;
    dx = 0.0;
    dy = 0.0;
    dz = 0.0;
    
    for (size_t i = 0; i < coefficients.size(); ++i) {
        value += coefficients[i] * basis_vals[i];
        dx += coefficients[i] * dx_vals[i];
        dy += coefficients[i] * dy_vals[i];
        dz += coefficients[i] * dz_vals[i];
    }
}

// Solve complete MLS system
MagneticFieldData HermiteMLSInterpolator::solveMLSSystem(
    const Point3D& query_point,
    const std::vector<size_t>& neighbor_indices,
    const std::vector<Real>& distances) const {

    // Check for exact matches first
    for (size_t i = 0; i < neighbor_indices.size(); ++i) {
        if (distances[i] < 1e-10) {
            // Query point coincides with data point
            return field_data_[neighbor_indices[i]];
        }
    }

    MagneticFieldData result;

    // Solve for each component (Bx, By, Bz)
    std::vector<Real> coeffs_x, coeffs_y, coeffs_z;

    bool success_x = solveMLSSystemForComponent(query_point, neighbor_indices, distances, 0, coeffs_x);
    bool success_y = solveMLSSystemForComponent(query_point, neighbor_indices, distances, 1, coeffs_y);
    bool success_z = solveMLSSystemForComponent(query_point, neighbor_indices, distances, 2, coeffs_z);

    if (!success_x || !success_y || !success_z) {
        // Fallback to simple IDW if MLS fails
        Real total_weight = 0.0;
        result = MagneticFieldData();

        for (size_t i = 0; i < neighbor_indices.size(); ++i) {
            Real weight = computeWeight(distances[i], params_.support_radius);
            weight /= (distances[i] * distances[i] + 1e-10);

            result += field_data_[neighbor_indices[i]] * weight;
            total_weight += weight;
        }

        if (total_weight > 0.0) {
            result = result * (1.0 / total_weight);
        }

        return result;
    }

    // Evaluate polynomials at query point
    evaluatePolynomial(coeffs_x, query_point, query_point,
                      result.Bx, result.dBx_dx, result.dBx_dy, result.dBx_dz);
    evaluatePolynomial(coeffs_y, query_point, query_point,
                      result.By, result.dBy_dx, result.dBy_dy, result.dBy_dz);
    evaluatePolynomial(coeffs_z, query_point, query_point,
                      result.Bz, result.dBz_dx, result.dBz_dy, result.dBz_dz);

    return result;
}

P3D_NAMESPACE_END

#include "point3d_interp/types.h"
#include "point3d_interp/hermite_mls_interpolator.h"
#include <cuda_runtime.h>

P3D_NAMESPACE_BEGIN
namespace cuda {

// HMLS Parameters structure for GPU
struct HMLSParameters {
    int basis_order;              // 0=Linear, 1=Quadratic, 2=Cubic
    int weight_function;          // 0=Gaussian, 1=Wendland  
    Real support_radius;
    Real derivative_weight;
    size_t max_neighbors;
    Real regularization;
};

/**
 * @brief Get basis size based on order
 */
__device__ int GetBasisSize(int basis_order) {
    if (basis_order == 0) return 4;   // Linear
    if (basis_order == 1) return 10;  // Quadratic  
    return 20;  // Cubic
}

/**
 * @brief Compute weight function value
 */
__device__ Real ComputeHMLSWeight(Real distance, Real support_radius, int weight_function) {
    if (distance >= support_radius) {
        return 0.0f;
    }
    
    Real r = distance / support_radius;
    
    if (weight_function == 0) {  // Gaussian
        return expf(-r * r);
    } else {  // Wendland
        Real one_minus_r = 1.0f - r;
        if (one_minus_r <= 0.0f) return 0.0f;
        Real one_minus_r_sq = one_minus_r * one_minus_r;
        return one_minus_r_sq * one_minus_r_sq * (4.0f * r + 1.0f);
    }
}

/**
 * @brief Evaluate polynomial basis functions
 */
__device__ void EvaluateBasisGPU(const Point3D& point, const Point3D& center, 
                                  int basis_order, Real* basis) {
    Real dx = point.x - center.x;
    Real dy = point.y - center.y;
    Real dz = point.z - center.z;
    
    //  Linear basis
    basis[0] = 1.0f;
    basis[1] = dx;
    basis[2] = dy;
    basis[3] = dz;
    
    if (basis_order == 0) return;
    
    // Quadratic terms
    basis[4] = dx * dx;
    basis[5] = dx * dy;
    basis[6] = dx * dz;
    basis[7] = dy * dy;
    basis[8] = dy * dz;
    basis[9] = dz * dz;
    
    if (basis_order == 1) return;
    
    // Cubic terms
    basis[10] = dx * dx * dx;
    basis[11] = dx * dx * dy;
    basis[12] = dx * dx * dz;
    basis[13] = dx * dy * dy;
    basis[14] = dx * dy * dz;
    basis[15] = dx * dz * dz;
    basis[16] = dy * dy * dy;
    basis[17] = dy * dy * dz;
    basis[18] = dy * dz * dz;
    basis[19] = dz * dz * dz;
}

/**
 * @brief Evaluate basis derivatives
 */
__device__ void EvaluateBasisDerivativesGPU(const Point3D& point, const Point3D& center,
                                             int basis_order, Real* dx_basis, 
                                             Real* dy_basis, Real* dz_basis) {
    Real dx = point.x - center.x;
    Real dy = point.y - center.y;
    Real dz = point.z - center.z;
    
    // Linear derivatives
    dx_basis[0] = 0.0f; dx_basis[1] = 1.0f; dx_basis[2] = 0.0f; dx_basis[3] = 0.0f;
    dy_basis[0] = 0.0f; dy_basis[1] = 0.0f; dy_basis[2] = 1.0f; dy_basis[3] = 0.0f;
    dz_basis[0] = 0.0f; dz_basis[1] = 0.0f; dz_basis[2] = 0.0f; dz_basis[3] = 1.0f;
    
    if (basis_order == 0) return;
    
    // Quadratic derivatives
    dx_basis[4] = 2.0f * dx; dx_basis[5] = dy; dx_basis[6] = dz;
    dx_basis[7] = 0.0f; dx_basis[8] = 0.0f; dx_basis[9] = 0.0f;
    
    dy_basis[4] = 0.0f; dy_basis[5] = dx; dy_basis[6] = 0.0f;
    dy_basis[7] = 2.0f * dy; dy_basis[8] = dz; dy_basis[9] = 0.0f;
    
    dz_basis[4] = 0.0f; dz_basis[5] = 0.0f; dz_basis[6] = dx;
    dz_basis[7] = 0.0f; dz_basis[8] = dy; dz_basis[9] = 2.0f * dz;
    
    if (basis_order == 1) return;
    
    // Cubic derivatives (simplified)
    dx_basis[10] = 3.0f * dx * dx; dx_basis[11] = 2.0f * dx * dy; dx_basis[12] = 2.0f * dx * dz;
    dx_basis[13] = dy * dy; dx_basis[14] = dy * dz; dx_basis[15] = dz * dz;
    dx_basis[16] = 0.0f; dx_basis[17] = 0.0f; dx_basis[18] = 0.0f; dx_basis[19] = 0.0f;
    
    dy_basis[10] = 0.0f; dy_basis[11] = dx * dx; dy_basis[12] = 0.0f;
    dy_basis[13] = 2.0f * dx * dy; dy_basis[14] = dx * dz; dy_basis[15] = 0.0f;
    dy_basis[16] = 3.0f * dy * dy; dy_basis[17] = 2.0f * dy * dz; dy_basis[18] = dz * dz; dy_basis[19] = 0.0f;
    
    dz_basis[10] = 0.0f; dz_basis[11] = 0.0f; dz_basis[12] = dx * dx;
    dz_basis[13] = 0.0f; dz_basis[14] = dx * dy; dz_basis[15] = 2.0f * dx * dz;
    dz_basis[16] = 0.0f; dz_basis[17] = dy * dy; dz_basis[18] = 2.0f * dy * dz; dz_basis[19] = 3.0f * dz * dz;
}

/**
 * @brief Simplified least squares solver for GPU (normal equations with Cholesky-like)
 * Solves A^T A x = A^T b
 */
__device__ bool SolveLeastSquaresGPU(const Real* A, const Real* b, Real* x,
                                      int m, int n, Real regularization) {
    // Compute A^T * A (n x n)
    Real ATA[20 * 20];  // Max size for cubic basis
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Real sum = 0.0f;
            for (int k = 0; k < m; ++k) {
                sum += A[k * n + i] * A[k * n + j];
            }
            ATA[i * n + j] = sum;
            if (i == j) ATA[i * n + j] += regularization;
        }
    }
    
    // Compute A^T * b (n x 1)
    Real ATb[20];
    for (int i = 0; i < n; ++i) {
        Real sum = 0.0f;
        for (int k = 0; k < m; ++k) {
            sum += A[k * n + i] * b[k];
        }
        ATb[i] = sum;
    }
    
    // Gauss elimination with partial pivoting
    Real mat[20 * 20];
    Real rhs[20];
    for (int i = 0; i < n * n; ++i) mat[i] = ATA[i];
    for (int i = 0; i < n; ++i) rhs[i] = ATb[i];
    
    for (int i = 0; i < n; ++i) {
        // Find pivot
        int max_row = i;
        Real max_val = fabsf(mat[i * n + i]);
        for (int k = i + 1; k < n; ++k) {
            Real val = fabsf(mat[k * n + i]);
            if (val > max_val) {
                max_val = val;
                max_row = k;
            }
        }
        
        if (max_val < 1e-10f) return false;
        
        // Swap rows
        if (max_row != i) {
            for (int j = 0; j < n; ++j) {
                Real temp = mat[i * n + j];
                mat[i * n + j] = mat[max_row * n + j];
                mat[max_row * n + j] = temp;
            }
            Real temp = rhs[i];
            rhs[i] = rhs[max_row];
            rhs[max_row] = temp;
        }
        
        // Eliminate
        for (int k = i + 1; k < n; ++k) {
            Real factor = mat[k * n + i] / mat[i * n + i];
            for (int j = i; j < n; ++j) {
                mat[k * n + j] -= factor * mat[i * n + j];
            }
            rhs[k] -= factor * rhs[i];
        }
    }
    
    // Back substitution
    for (int i = n - 1; i >= 0; --i) {
        x[i] = rhs[i];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= mat[i * n + j] * x[j];
        }
        x[i] /= mat[i * n + i];
    }
    
    return true;
}

/**
 * @brief CUDA Kernel: Hermite MLS interpolation using spatial grid
 */
__global__ void HermiteMLSKernel(
    const Point3D* __restrict__ query_points,
    const Point3D* __restrict__ data_points,
    const MagneticFieldData* __restrict__ field_data,
    const size_t data_count,
    const uint32_t* __restrict__ cell_offsets,
    const uint32_t* __restrict__ cell_points,
    const Point3D grid_origin,
    const Point3D grid_cell_size,
    uint32_t grid_dim_x,
    uint32_t grid_dim_y,
    uint32_t grid_dim_z,
    const HMLSParameters params,
    const Point3D min_bound,
    const Point3D max_bound,
    InterpolationResult* __restrict__ results,
    const size_t query_count) {
    
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= query_count) return;
    
    const Point3D query_point = query_points[tid];
    InterpolationResult result = {};
    result.valid = true;
    
    // Find cell
    int query_ix = static_cast<int>((query_point.x - grid_origin.x) / grid_cell_size.x);
    int query_iy = static_cast<int>((query_point.y - grid_origin.y) / grid_cell_size.y);
    int query_iz = static_cast<int>((query_point.z - grid_origin.z) / grid_cell_size.z);
    
    query_ix = max(0, min(query_ix, static_cast<int>(grid_dim_x) - 1));
    query_iy = max(0, min(query_iy, static_cast<int>(grid_dim_y) - 1));
    query_iz = max(0, min(query_iz, static_cast<int>(grid_dim_z) - 1));
    
    // Collect neighbors from surrounding cells
    const int max_neighbors = min(static_cast<int>(params.max_neighbors), 50);
    int neighbor_count = 0;
    uint32_t neighbor_indices[50];
    Real neighbor_distances[50];
    
    // Search 2x2x2 neighborhood
    for (int dz = -1; dz <= 1 && neighbor_count < max_neighbors; ++dz) {
        for (int dy = -1; dy <= 1 && neighbor_count < max_neighbors; ++dy) {
            for (int dx = -1; dx <= 1 && neighbor_count < max_neighbors; ++dx) {
                int cell_ix = query_ix + dx;
                int cell_iy = query_iy + dy;
                int cell_iz = query_iz + dz;
                
                if (cell_ix < 0 || cell_ix >= static_cast<int>(grid_dim_x) ||
                    cell_iy < 0 || cell_iy >= static_cast<int>(grid_dim_y) ||
                    cell_iz < 0 || cell_iz >= static_cast<int>(grid_dim_z)) {
                    continue;
                }
                
                size_t cell_idx = cell_ix + cell_iy * grid_dim_x + cell_iz * grid_dim_x * grid_dim_y;
                size_t start_idx = cell_offsets[cell_idx];
                size_t end_idx = cell_offsets[cell_idx + 1];
                
                for (size_t i = start_idx; i < end_idx && neighbor_count < max_neighbors; ++i) {
                    uint32_t point_idx = cell_points[i];
                    const Point3D& pt = data_points[point_idx];
                    
                    Real dx_val = query_point.x - pt.x;
                    Real dy_val = query_point.y - pt.y;
                    Real dz_val = query_point.z - pt.z;
                    Real dist = sqrtf(dx_val * dx_val + dy_val * dy_val + dz_val * dz_val);
                    
                    if (dist < params.support_radius) {
                        neighbor_indices[neighbor_count] = point_idx;
                        neighbor_distances[neighbor_count] = dist;
                        neighbor_count++;
                    }
                }
            }
        }
    }
    
    if (neighbor_count < 4) {
        // Fallback to IDW
        Real weight_sum = 0.0f;
        MagneticFieldData weighted_sum = {};
        
        for (int i = 0; i < neighbor_count; ++i) {
            Real dist = neighbor_distances[i];
            if (dist < 1e-6f) {
                result.data = field_data[neighbor_indices[i]];
                results[tid] = result;
                return;
            }
            Real weight = 1.0f / (dist * dist);
            weighted_sum.Bx += field_data[neighbor_indices[i]].Bx * weight;
            weighted_sum.By += field_data[neighbor_indices[i]].By * weight;
            weighted_sum.Bz += field_data[neighbor_indices[i]].Bz * weight;
            weight_sum += weight;
        }
        
        if (weight_sum > 0.0f) {
            result.data.Bx = weighted_sum.Bx / weight_sum;
            result.data.By = weighted_sum.By / weight_sum;
            result.data.Bz = weighted_sum.Bz / weight_sum;
        }
        results[tid] = result;
        return;
    }
    
    // Solve HMLS system for each component
    const int n = GetBasisSize(params.basis_order);
    const int m = 4 * neighbor_count;
    
    Real A[200 * 20];  // Max 50 neighbors * 4 constraints, 20 basis
    Real b[200];
    Real coeffs[20];
    
    Real lambda_sqrt = sqrtf(params.derivative_weight);
    
    // Build system for Bx
    for (int i = 0; i < neighbor_count; ++i) {
        uint32_t idx = neighbor_indices[i];
        const Point3D& pt = data_points[idx];
        const MagneticFieldData& fd = field_data[idx];
        
        Real weight = ComputeHMLSWeight(neighbor_distances[i], params.support_radius, params.weight_function);
        Real w_sqrt = sqrtf(weight);
        
        Real basis[20], dx_basis[20], dy_basis[20], dz_basis[20];
        EvaluateBasisGPU(pt, query_point, params.basis_order, basis);
        EvaluateBasisDerivativesGPU(pt, query_point, params.basis_order, dx_basis, dy_basis, dz_basis);
        
        // Value constraint
        for (int j = 0; j < n; ++j) {
            A[i * n + j] = w_sqrt * basis[j];
        }
        b[i] = w_sqrt * fd.Bx;
        
        // Derivative constraints
        for (int j = 0; j < n; ++j) {
            A[(neighbor_count + i) * n + j] = w_sqrt * lambda_sqrt * dx_basis[j];
            A[(2 * neighbor_count + i) * n + j] = w_sqrt * lambda_sqrt * dy_basis[j];
            A[(3 * neighbor_count + i) * n + j] = w_sqrt * lambda_sqrt * dz_basis[j];
        }
        b[neighbor_count + i] = w_sqrt * lambda_sqrt * fd.dBx_dx;
        b[2 * neighbor_count + i] = w_sqrt * lambda_sqrt * fd.dBx_dy;
        b[3 * neighbor_count + i] = w_sqrt * lambda_sqrt * fd.dBx_dz;
    }
    
    if (SolveLeastSquaresGPU(A, b, coeffs, m, n, params.regularization)) {
        Real basis[20];
        EvaluateBasisGPU(query_point, query_point, params.basis_order, basis);
        result.data.Bx = 0.0f;
        for (int i = 0; i < n; ++i) {
            result.data.Bx += coeffs[i] * basis[i];
        }
    }

    // Solve for By
    for (int i = 0; i < neighbor_count; ++i) {
        uint32_t idx = neighbor_indices[i];
        const Point3D& pt = data_points[idx];
        const MagneticFieldData& fd = field_data[idx];

        Real weight = ComputeHMLSWeight(neighbor_distances[i], params.support_radius, params.weight_function);
        Real w_sqrt = sqrtf(weight);

        Real basis[20], dx_basis[20], dy_basis[20], dz_basis[20];
        EvaluateBasisGPU(pt, query_point, params.basis_order, basis);
        EvaluateBasisDerivativesGPU(pt, query_point, params.basis_order, dx_basis, dy_basis, dz_basis);

        // Value constraint
        for (int j = 0; j < n; ++j) {
            A[i * n + j] = w_sqrt * basis[j];
        }
        b[i] = w_sqrt * fd.By;

        // Derivative constraints
        for (int j = 0; j < n; ++j) {
            A[(neighbor_count + i) * n + j] = w_sqrt * lambda_sqrt * dx_basis[j];
            A[(2 * neighbor_count + i) * n + j] = w_sqrt * lambda_sqrt * dy_basis[j];
            A[(3 * neighbor_count + i) * n + j] = w_sqrt * lambda_sqrt * dz_basis[j];
        }
        b[neighbor_count + i] = w_sqrt * lambda_sqrt * fd.dBy_dx;
        b[2 * neighbor_count + i] = w_sqrt * lambda_sqrt * fd.dBy_dy;
        b[3 * neighbor_count + i] = w_sqrt * lambda_sqrt * fd.dBy_dz;
    }

    if (SolveLeastSquaresGPU(A, b, coeffs, m, n, params.regularization)) {
        Real basis[20];
        EvaluateBasisGPU(query_point, query_point, params.basis_order, basis);
        result.data.By = 0.0f;
        for (int i = 0; i < n; ++i) {
            result.data.By += coeffs[i] * basis[i];
        }
    }

    // Solve for Bz
    for (int i = 0; i < neighbor_count; ++i) {
        uint32_t idx = neighbor_indices[i];
        const Point3D& pt = data_points[idx];
        const MagneticFieldData& fd = field_data[idx];

        Real weight = ComputeHMLSWeight(neighbor_distances[i], params.support_radius, params.weight_function);
        Real w_sqrt = sqrtf(weight);

        Real basis[20], dx_basis[20], dy_basis[20], dz_basis[20];
        EvaluateBasisGPU(pt, query_point, params.basis_order, basis);
        EvaluateBasisDerivativesGPU(pt, query_point, params.basis_order, dx_basis, dy_basis, dz_basis);

        // Value constraint
        for (int j = 0; j < n; ++j) {
            A[i * n + j] = w_sqrt * basis[j];
        }
        b[i] = w_sqrt * fd.Bz;

        // Derivative constraints
        for (int j = 0; j < n; ++j) {
            A[(neighbor_count + i) * n + j] = w_sqrt * lambda_sqrt * dx_basis[j];
            A[(2 * neighbor_count + i) * n + j] = w_sqrt * lambda_sqrt * dy_basis[j];
            A[(3 * neighbor_count + i) * n + j] = w_sqrt * lambda_sqrt * dz_basis[j];
        }
        b[neighbor_count + i] = w_sqrt * lambda_sqrt * fd.dBz_dx;
        b[2 * neighbor_count + i] = w_sqrt * lambda_sqrt * fd.dBz_dy;
        b[3 * neighbor_count + i] = w_sqrt * lambda_sqrt * fd.dBz_dz;
    }

    if (SolveLeastSquaresGPU(A, b, coeffs, m, n, params.regularization)) {
        Real basis[20];
        EvaluateBasisGPU(query_point, query_point, params.basis_order, basis);
        result.data.Bz = 0.0f;
        for (int i = 0; i < n; ++i) {
            result.data.Bz += coeffs[i] * basis[i];
        }
    }

    results[tid] = result;
}

}  // namespace cuda
P3D_NAMESPACE_END

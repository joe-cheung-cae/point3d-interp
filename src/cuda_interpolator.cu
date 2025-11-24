#include "point3d_interp/types.h"
#include "point3d_interp/constants.h"
#include <cuda_runtime.h>

namespace p3d {
namespace cuda {

/**
 * @brief Convert world coordinates to grid coordinates
 */
__device__ Point3D WorldToGrid(const Point3D& world_point, const GridParams& params) {
    return Point3D((world_point.x - params.origin.x) / params.spacing.x,
                   (world_point.y - params.origin.y) / params.spacing.y,
                   (world_point.z - params.origin.z) / params.spacing.z);
}

/**
 * @brief Get grid data index
 */
__device__ uint32_t GetGridIndex(uint32_t i, uint32_t j, uint32_t k, const GridParams& params) {
    return i + j * params.dimensions[0] + k * params.dimensions[0] * params.dimensions[1];
}

/**
 * @brief Get cell vertex indices
 */
__device__ bool GetCellVertexIndices(const Point3D& grid_coords, const GridParams& params, uint32_t indices[8]) {
    // Get cell starting indices (floor)
    int i0 = static_cast<int>(grid_coords.x);
    int j0 = static_cast<int>(grid_coords.y);
    int k0 = static_cast<int>(grid_coords.z);

    int i1 = i0 + 1;
    int j1 = j0 + 1;
    int k1 = k0 + 1;

    // Check bounds
    if (i0 < 0 || i1 >= static_cast<int>(params.dimensions[0]) || j0 < 0 ||
        j1 >= static_cast<int>(params.dimensions[1]) || k0 < 0 || k1 >= static_cast<int>(params.dimensions[2])) {
        return false;
    }

    // Calculate indices of 8 vertices
    indices[0] = GetGridIndex(i0, j0, k0, params);  // (i0, j0, k0)
    indices[1] = GetGridIndex(i1, j0, k0, params);  // (i1, j0, k0)
    indices[2] = GetGridIndex(i0, j1, k0, params);  // (i0, j1, k0)
    indices[3] = GetGridIndex(i1, j1, k0, params);  // (i1, j1, k0)
    indices[4] = GetGridIndex(i0, j0, k1, params);  // (i0, j0, k1)
    indices[5] = GetGridIndex(i1, j0, k1, params);  // (i1, j0, k1)
    indices[6] = GetGridIndex(i0, j1, k1, params);  // (i0, j1, k1)
    indices[7] = GetGridIndex(i1, j1, k1, params);  // (i1, j1, k1)

    return true;
}

/**
 * @brief 1D Hermite interpolation
 */
__device__ Real HermiteInterpolate(Real f0, Real f1, Real df0, Real df1, Real t) {
    Real t2  = t * t;
    Real t3  = t2 * t;
    Real h00 = 2 * t3 - 3 * t2 + 1;
    Real h10 = t3 - 2 * t2 + t;
    Real h01 = -2 * t3 + 3 * t2;
    Real h11 = t3 - t2;
    return f0 * h00 + df0 * h10 + f1 * h01 + df1 * h11;
}

/**
 * @brief 1D Hermite interpolation derivative
 */
__device__ Real HermiteDerivative(Real f0, Real f1, Real df0, Real df1, Real t) {
    Real t2      = t * t;
    Real dh00_dt = 6 * t2 - 6 * t;
    Real dh10_dt = 3 * t2 - 4 * t + 1;
    Real dh01_dt = -6 * t2 + 6 * t;
    Real dh11_dt = 3 * t2 - 2 * t;
    return f0 * dh00_dt + df0 * dh10_dt + f1 * dh01_dt + df1 * dh11_dt;
}

/**
 * @brief Perform tricubic Hermite interpolation
 */
__device__ MagneticFieldData TricubicHermiteInterpolate(const MagneticFieldData vertex_data[8], Real tx, Real ty,
                                                        Real tz, const GridParams& params) {
    MagneticFieldData result;

    // Vertex indices correspond to:
    // 0: (i,j,k),     1: (i+1,j,k),   2: (i,j+1,k),   3: (i+1,j+1,k)
    // 4: (i,j,k+1),   5: (i+1,j,k+1), 6: (i,j+1,k+1), 7: (i+1,j+1,k+1)

    // Interpolate along x for each of the 4 yz positions
    MagneticFieldData interp_x[4];
    for (int i = 0; i < 4; ++i) {
        int idx0 = (i % 2) * 2 + (i / 2) * 4;  // 0,2,4,6 for i=0,1,2,3
        int idx1 = idx0 + 1;                   // 1,3,5,7

        // Interpolate field values using Hermite
        interp_x[i].Bx = HermiteInterpolate(vertex_data[idx0].Bx, vertex_data[idx1].Bx, vertex_data[idx0].dBx_dx,
                                            vertex_data[idx1].dBx_dx, tx);
        interp_x[i].By = HermiteInterpolate(vertex_data[idx0].By, vertex_data[idx1].By, vertex_data[idx0].dBy_dx,
                                            vertex_data[idx1].dBy_dx, tx);
        interp_x[i].Bz = HermiteInterpolate(vertex_data[idx0].Bz, vertex_data[idx1].Bz, vertex_data[idx0].dBz_dx,
                                            vertex_data[idx1].dBz_dx, tx);

        // Compute derivatives: dBx_dx is derivative of Hermite in x direction
        interp_x[i].dBx_dx = HermiteDerivative(vertex_data[idx0].Bx, vertex_data[idx1].Bx, vertex_data[idx0].dBx_dx,
                                               vertex_data[idx1].dBx_dx, tx);
        // Other derivatives interpolated linearly
        interp_x[i].dBx_dy = (1 - tx) * vertex_data[idx0].dBx_dy + tx * vertex_data[idx1].dBx_dy;
        interp_x[i].dBx_dz = (1 - tx) * vertex_data[idx0].dBx_dz + tx * vertex_data[idx1].dBx_dz;
        interp_x[i].dBy_dx = HermiteDerivative(vertex_data[idx0].By, vertex_data[idx1].By, vertex_data[idx0].dBy_dx,
                                               vertex_data[idx1].dBy_dx, tx);
        interp_x[i].dBy_dy = (1 - tx) * vertex_data[idx0].dBy_dy + tx * vertex_data[idx1].dBy_dy;
        interp_x[i].dBy_dz = (1 - tx) * vertex_data[idx0].dBy_dz + tx * vertex_data[idx1].dBy_dz;
        interp_x[i].dBz_dx = HermiteDerivative(vertex_data[idx0].Bz, vertex_data[idx1].Bz, vertex_data[idx0].dBz_dx,
                                               vertex_data[idx1].dBz_dx, tx);
        interp_x[i].dBz_dy = (1 - tx) * vertex_data[idx0].dBz_dy + tx * vertex_data[idx1].dBz_dy;
        interp_x[i].dBz_dz = (1 - tx) * vertex_data[idx0].dBz_dz + tx * vertex_data[idx1].dBz_dz;
    }

    // Interpolate along y for the 2 z layers
    MagneticFieldData interp_y[2];
    for (int i = 0; i < 2; ++i) {
        int idx0 = i * 2;     // 0,2 for z=0,1
        int idx1 = idx0 + 1;  // 1,3

        // Interpolate field values using Hermite
        interp_y[i].Bx =
            HermiteInterpolate(interp_x[idx0].Bx, interp_x[idx1].Bx, interp_x[idx0].dBx_dy, interp_x[idx1].dBx_dy, ty);
        interp_y[i].By =
            HermiteInterpolate(interp_x[idx0].By, interp_x[idx1].By, interp_x[idx0].dBy_dy, interp_x[idx1].dBy_dy, ty);
        interp_y[i].Bz =
            HermiteInterpolate(interp_x[idx0].Bz, interp_x[idx1].Bz, interp_x[idx0].dBz_dy, interp_x[idx1].dBz_dy, ty);

        // Compute derivatives
        interp_y[i].dBx_dx = (1 - ty) * interp_x[idx0].dBx_dx + ty * interp_x[idx1].dBx_dx;
        interp_y[i].dBx_dy =
            HermiteDerivative(interp_x[idx0].Bx, interp_x[idx1].Bx, interp_x[idx0].dBx_dy, interp_x[idx1].dBx_dy, ty);
        interp_y[i].dBx_dz = (1 - ty) * interp_x[idx0].dBx_dz + ty * interp_x[idx1].dBx_dz;
        interp_y[i].dBy_dx = (1 - ty) * interp_x[idx0].dBy_dx + ty * interp_x[idx1].dBy_dx;
        interp_y[i].dBy_dy =
            HermiteDerivative(interp_x[idx0].By, interp_x[idx1].By, interp_x[idx0].dBy_dy, interp_x[idx1].dBy_dy, ty);
        interp_y[i].dBy_dz = (1 - ty) * interp_x[idx0].dBy_dz + ty * interp_x[idx1].dBy_dz;
        interp_y[i].dBz_dx = (1 - ty) * interp_x[idx0].dBz_dx + ty * interp_x[idx1].dBz_dx;
        interp_y[i].dBz_dy =
            HermiteDerivative(interp_x[idx0].Bz, interp_x[idx1].Bz, interp_x[idx0].dBz_dy, interp_x[idx1].dBz_dy, ty);
        interp_y[i].dBz_dz = (1 - ty) * interp_x[idx0].dBz_dz + ty * interp_x[idx1].dBz_dz;
    }

    // Interpolate along z
    result.Bx = HermiteInterpolate(interp_y[0].Bx, interp_y[1].Bx, interp_y[0].dBx_dz, interp_y[1].dBx_dz, tz);
    result.By = HermiteInterpolate(interp_y[0].By, interp_y[1].By, interp_y[0].dBy_dz, interp_y[1].dBy_dz, tz);
    result.Bz = HermiteInterpolate(interp_y[0].Bz, interp_y[1].Bz, interp_y[0].dBz_dz, interp_y[1].dBz_dz, tz);

    // Compute final derivatives
    result.dBx_dx = (1 - tz) * interp_y[0].dBx_dx + tz * interp_y[1].dBx_dx;
    result.dBx_dy = (1 - tz) * interp_y[0].dBx_dy + tz * interp_y[1].dBx_dy;
    result.dBx_dz = HermiteDerivative(interp_y[0].Bx, interp_y[1].Bx, interp_y[0].dBx_dz, interp_y[1].dBx_dz, tz);
    result.dBy_dx = (1 - tz) * interp_y[0].dBy_dx + tz * interp_y[1].dBy_dx;
    result.dBy_dy = (1 - tz) * interp_y[0].dBy_dy + tz * interp_y[1].dBy_dy;
    result.dBy_dz = HermiteDerivative(interp_y[0].By, interp_y[1].By, interp_y[0].dBy_dz, interp_y[1].dBy_dz, tz);
    result.dBz_dx = (1 - tz) * interp_y[0].dBz_dx + tz * interp_y[1].dBz_dx;
    result.dBz_dy = (1 - tz) * interp_y[0].dBz_dy + tz * interp_y[1].dBz_dy;
    result.dBz_dz = HermiteDerivative(interp_y[0].Bz, interp_y[1].Bz, interp_y[0].dBz_dz, interp_y[1].dBz_dz, tz);

    // Scale derivatives to world coordinates
    result.dBx_dx /= params.spacing.x;
    result.dBx_dy /= params.spacing.y;
    result.dBx_dz /= params.spacing.z;
    result.dBy_dx /= params.spacing.x;
    result.dBy_dy /= params.spacing.y;
    result.dBy_dz /= params.spacing.z;
    result.dBz_dx /= params.spacing.x;
    result.dBz_dy /= params.spacing.y;
    result.dBz_dz /= params.spacing.z;

    return result;
}

/**
 * @brief Calculate Euclidean distance between two 3D points
 */
__device__ Real Distance(const Point3D& p1, const Point3D& p2) {
    Real dx = p1.x - p2.x;
    Real dy = p1.y - p2.y;
    Real dz = p1.z - p2.z;
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

/**
 * @brief Fast power function optimized for common IDW powers
 * Replaces slow powf() with optimized implementations for common cases
 */
__device__ Real FastPow(Real base, Real exponent) {
    // Handle common integer powers efficiently
    if (exponent == 2.0f) {
        return base * base;
    } else if (exponent == 1.0f) {
        return base;
    } else if (exponent == 0.5f) {
        return sqrtf(base);
    } else if (exponent == 3.0f) {
        return base * base * base;
    } else if (exponent == 4.0f) {
        Real base_sq = base * base;
        return base_sq * base_sq;
    } else if (exponent == 0.0f) {
        return 1.0f;
    }

    // For other powers, use exp/log approximation (faster than powf for some cases)
    // Only use for reasonable ranges to avoid numerical issues
    if (base > DISTANCE_EPSILON && base < 1e8f && exponent > 0.1f && exponent < 10.0f) {
        return expf(exponent * logf(base));
    }

    // Fallback to standard powf for edge cases
    return powf(base, exponent);
}

/**
 * @brief Check if a point is inside the bounding box
 */
__device__ bool IsPointInsideBounds(const Point3D& point, const Point3D& min_bound, const Point3D& max_bound) {
    return (point.x >= min_bound.x && point.x <= max_bound.x) && (point.y >= min_bound.y && point.y <= max_bound.y) &&
           (point.z >= min_bound.z && point.z <= max_bound.z);
}

/**
 * @brief Find k nearest neighbors and compute improved linear extrapolation
 *
 * This function implements a weighted least squares gradient estimation for improved
 * accuracy in complex magnetic fields. The algorithm:
 * 1. Finds the k nearest neighbors to the query point
 * 2. Uses weighted least squares to estimate the magnetic field gradient at the nearest point
 * 3. Extrapolates linearly from the nearest point using the estimated gradient
 *
 * Limitations and accuracy expectations:
 * - Assumes locally linear field behavior (valid for smooth fields)
 * - Accuracy decreases with distance from data points
 * - Performance depends on field complexity and data distribution
 * - Best results when query point is within the convex hull of data points
 * - May produce artifacts in highly non-linear or discontinuous fields
 *
 * @param query_point The point where field values are needed
 * @param data_points Array of data point coordinates
 * @param field_data Array of magnetic field data at data points
 * @param data_count Total number of data points
 * @return Extrapolated magnetic field data at query point
 */
__device__ MagneticFieldData LinearExtrapolate(const Point3D& query_point, const Point3D* data_points,
                                               const MagneticFieldData* field_data, size_t data_count) {
    const size_t max_neighbors    = MAX_EXTRAPOLATION_NEIGHBORS;
    size_t       actual_neighbors = min(max_neighbors, data_count);

    // Find nearest neighbors (simple bubble sort for small k)
    size_t nearest_indices[MAX_EXTRAPOLATION_NEIGHBORS];
    Real   nearest_distances[MAX_EXTRAPOLATION_NEIGHBORS];

    // Initialize with first few points
    for (size_t i = 0; i < actual_neighbors; ++i) {
        nearest_indices[i]   = i;
        nearest_distances[i] = Distance(query_point, data_points[i]);
    }

    // Simple selection sort for nearest neighbors
    for (size_t i = 0; i < actual_neighbors - 1; ++i) {
        for (size_t j = i + 1; j < actual_neighbors; ++j) {
            if (nearest_distances[j] < nearest_distances[i]) {
                // Swap
                Real   temp_dist     = nearest_distances[i];
                size_t temp_idx      = nearest_indices[i];
                nearest_distances[i] = nearest_distances[j];
                nearest_indices[i]   = nearest_indices[j];
                nearest_distances[j] = temp_dist;
                nearest_indices[j]   = temp_idx;
            }
        }
    }

    // Check remaining points
    for (size_t i = actual_neighbors; i < data_count; ++i) {
        Real dist = Distance(query_point, data_points[i]);
        // Check if this point is closer than the farthest in our list
        if (dist < nearest_distances[actual_neighbors - 1]) {
            // Replace the farthest
            nearest_distances[actual_neighbors - 1] = dist;
            nearest_indices[actual_neighbors - 1]   = i;

            // Re-sort the last element
            for (size_t j = actual_neighbors - 2; j < actual_neighbors - 1; --j) {
                if (nearest_distances[j + 1] < nearest_distances[j]) {
                    Real   temp_dist         = nearest_distances[j];
                    size_t temp_idx          = nearest_indices[j];
                    nearest_distances[j]     = nearest_distances[j + 1];
                    nearest_indices[j]       = nearest_indices[j + 1];
                    nearest_distances[j + 1] = temp_dist;
                    nearest_indices[j + 1]   = temp_idx;
                } else {
                    break;
                }
            }
        }
    }

    MagneticFieldData result = {};

    if (actual_neighbors >= 3) {  // Need at least 3 points for meaningful gradient estimation
        // Use nearest point as base for extrapolation
        size_t            nearest_idx   = nearest_indices[0];
        Point3D           nearest_point = data_points[nearest_idx];
        MagneticFieldData nearest_data  = field_data[nearest_idx];

        // Weighted least squares gradient estimation
        // Solve for gradient vector g = [dBx/dx, dBy/dx, dBz/dx, dBx/dy, dBy/dy, dBz/dy, dBx/dz, dBy/dz, dBz/dz]^T
        // using the linear model: B_i = B_nearest + g · (p_i - p_nearest)

        // Normal equations: A^T W A g = A^T W b
        // where A is the design matrix, W is diagonal weight matrix, b is the residual vector

        Real A[3 * MAX_EXTRAPOLATION_NEIGHBORS];  // Design matrix (dx, dy, dz for each neighbor)
        Real W[MAX_EXTRAPOLATION_NEIGHBORS];      // Weights (inverse distance squared)
        Real b[3 * MAX_EXTRAPOLATION_NEIGHBORS];  // Residuals (B_i - B_nearest for each component)

        size_t valid_neighbors = 0;

        // Build design matrix and residuals
        for (size_t i = 1; i < actual_neighbors; ++i) {
            size_t            idx = nearest_indices[i];
            Point3D           p   = data_points[idx];
            MagneticFieldData d   = field_data[idx];

            Real dx      = p.x - nearest_point.x;
            Real dy      = p.y - nearest_point.y;
            Real dz      = p.z - nearest_point.z;
            Real dist_sq = dx * dx + dy * dy + dz * dz;

            if (dist_sq > DISTANCE_EPSILON * DISTANCE_EPSILON) {
                // Weight by inverse distance squared (emphasizes closer points)
                Real weight = 1.0f / dist_sq;

                A[3 * valid_neighbors]     = dx;  // dBx/dx coefficient
                A[3 * valid_neighbors + 1] = dy;  // dBy/dy coefficient
                A[3 * valid_neighbors + 2] = dz;  // dBz/dz coefficient

                W[valid_neighbors] = weight;

                // Residuals for each field component
                b[3 * valid_neighbors]     = (d.Bx - nearest_data.Bx) * weight;
                b[3 * valid_neighbors + 1] = (d.By - nearest_data.By) * weight;
                b[3 * valid_neighbors + 2] = (d.Bz - nearest_data.Bz) * weight;

                valid_neighbors++;
            }
        }

        if (valid_neighbors >= 3) {  // Need at least 3 valid neighbors for stable solution
            // Solve normal equations for each gradient component independently
            // For simplicity, estimate gradients separately (can be improved with full 3D gradient tensor)

            Real sum_w    = 0.0f;
            Real sum_w_dx = 0.0f, sum_w_dy = 0.0f, sum_w_dz = 0.0f;
            Real sum_w_dx_dx = 0.0f, sum_w_dy_dy = 0.0f, sum_w_dz_dz = 0.0f;
            Real sum_w_dx_bx = 0.0f, sum_w_dy_by = 0.0f, sum_w_dz_bz = 0.0f;

            for (size_t i = 0; i < valid_neighbors; ++i) {
                Real w  = W[i];
                Real dx = A[3 * i];
                Real dy = A[3 * i + 1];
                Real dz = A[3 * i + 2];

                sum_w += w;
                sum_w_dx += w * dx;
                sum_w_dy += w * dy;
                sum_w_dz += w * dz;
                sum_w_dx_dx += w * dx * dx;
                sum_w_dy_dy += w * dy * dy;
                sum_w_dz_dz += w * dz * dz;
                sum_w_dx_bx += w * dx * b[3 * i] / w;  // Note: b already weighted
                sum_w_dy_by += w * dy * b[3 * i + 1] / w;
                sum_w_dz_bz += w * dz * b[3 * i + 2] / w;
            }

            // Estimate gradients using weighted least squares
            Real dBx_dx = 0.0f, dBy_dy = 0.0f, dBz_dz = 0.0f;

            // Solve for dBx/dx
            Real denom_x = sum_w * sum_w_dx_dx - sum_w_dx * sum_w_dx;
            if (fabsf(denom_x) > DISTANCE_EPSILON) {
                dBx_dx = (sum_w * sum_w_dx_bx - sum_w_dx * (sum_w_dx_bx / sum_w)) / denom_x;
            }

            // Solve for dBy/dy
            Real denom_y = sum_w * sum_w_dy_dy - sum_w_dy * sum_w_dy;
            if (fabsf(denom_y) > DISTANCE_EPSILON) {
                dBy_dy = (sum_w * sum_w_dy_by - sum_w_dy * (sum_w_dy_by / sum_w)) / denom_y;
            }

            // Solve for dBz/dz
            Real denom_z = sum_w * sum_w_dz_dz - sum_w_dz * sum_w_dz;
            if (fabsf(denom_z) > DISTANCE_EPSILON) {
                dBz_dz = (sum_w * sum_w_dz_bz - sum_w_dz * (sum_w_dz_bz / sum_w)) / denom_z;
            }

            // Extrapolate from nearest point using estimated gradients
            Real dx   = query_point.x - nearest_point.x;
            Real dy   = query_point.y - nearest_point.y;
            Real dz   = query_point.z - nearest_point.z;
            Real dist = sqrtf(dx * dx + dy * dy + dz * dz);

            if (dist > DISTANCE_EPSILON) {
                // Linear extrapolation: B = B_nearest + ∇B · d
                result.Bx = nearest_data.Bx + dBx_dx * dx;
                result.By = nearest_data.By + dBy_dy * dy;
                result.Bz = nearest_data.Bz + dBz_dz * dz;
            } else {
                result = nearest_data;
            }
        } else {
            // Fallback to simple average gradient if insufficient valid neighbors
            Real   avg_dBx = 0.0f, avg_dBy = 0.0f, avg_dBz = 0.0f;
            size_t gradient_count = 0;

            for (size_t i = 1; i < actual_neighbors; ++i) {
                size_t            idx = nearest_indices[i];
                Point3D           p   = data_points[idx];
                MagneticFieldData d   = field_data[idx];

                Real dx   = p.x - nearest_point.x;
                Real dy   = p.y - nearest_point.y;
                Real dz   = p.z - nearest_point.z;
                Real dist = sqrtf(dx * dx + dy * dy + dz * dz);

                if (dist > DISTANCE_EPSILON) {
                    Real inv_dist = 1.0f / dist;
                    avg_dBx += (d.Bx - nearest_data.Bx) * inv_dist;
                    avg_dBy += (d.By - nearest_data.By) * inv_dist;
                    avg_dBz += (d.Bz - nearest_data.Bz) * inv_dist;
                    gradient_count++;
                }
            }

            if (gradient_count > 0) {
                avg_dBx /= gradient_count;
                avg_dBy /= gradient_count;
                avg_dBz /= gradient_count;

                Real dx   = query_point.x - nearest_point.x;
                Real dy   = query_point.y - nearest_point.y;
                Real dz   = query_point.z - nearest_point.z;
                Real dist = sqrtf(dx * dx + dy * dy + dz * dz);

                if (dist > DISTANCE_EPSILON) {
                    result.Bx = nearest_data.Bx + avg_dBx * dist;
                    result.By = nearest_data.By + avg_dBy * dist;
                    result.Bz = nearest_data.Bz + avg_dBz * dist;
                } else {
                    result = nearest_data;
                }
            } else {
                result = nearest_data;
            }
        }
    } else {
        // Fallback to nearest neighbor if insufficient neighbors
        result = field_data[nearest_indices[0]];
    }

    return result;
}
/**
 * @brief Get cell index from 3D coordinates (device function)
 */
__device__ size_t GetCellIndex(int ix, int iy, int iz, uint32_t dim_x, uint32_t dim_y, uint32_t dim_z) {
    return ix + iy * dim_x + iz * dim_x * dim_y;
}

/**
 * @brief Get cell coordinates from world point (device function)
 */
__device__ void GetCellCoords(const Point3D& point, const Point3D& origin, const Point3D& cell_size,
                              uint32_t dim_x, uint32_t dim_y, uint32_t dim_z, int& ix, int& iy, int& iz) {
    ix = static_cast<int>((point.x - origin.x) / cell_size.x);
    iy = static_cast<int>((point.y - origin.y) / cell_size.y);
    iz = static_cast<int>((point.z - origin.z) / cell_size.z);

    // Clamp to bounds
    ix = max(0, min(ix, static_cast<int>(dim_x) - 1));
    iy = max(0, min(iy, static_cast<int>(dim_y) - 1));
    iz = max(0, min(iz, static_cast<int>(dim_z) - 1));
}

/**
 * @brief CUDA kernel: Optimized IDW interpolation using spatial grid
 *
 * Each thread handles interpolation calculation for one query point
 * Uses spatial grid to find nearby points efficiently
 *
 * @param query_points Query points array (device memory)
 * @param data_points Data points array (device memory)
 * @param field_data Magnetic field data array (device memory)
 * @param cell_offsets Cell offset array (device memory)
 * @param cell_points Cell point indices array (device memory)
 * @param grid_origin Grid origin
 * @param grid_cell_size Cell size in each dimension
 * @param grid_dimensions Grid dimensions
 * @param power IDW power parameter
 * @param extrapolation_method Extrapolation method
 * @param min_bound Minimum bounds
 * @param max_bound Maximum bounds
 * @param results Output results array (device memory)
 * @param query_count Number of query points
 */
__global__ void IDWSpatialGridKernel(const Point3D* __restrict__ query_points, const Point3D* __restrict__ data_points,
                                     const MagneticFieldData* __restrict__ field_data, const size_t data_count,
                                     const uint32_t* __restrict__ cell_offsets,
                                     const uint32_t* __restrict__ cell_points, const Point3D grid_origin,
                                     const Point3D grid_cell_size, uint32_t grid_dim_x, uint32_t grid_dim_y, uint32_t grid_dim_z, const Real power,
                                     const int extrapolation_method, const Point3D min_bound, const Point3D max_bound,
                                     InterpolationResult* __restrict__ results, const size_t query_count) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= query_count) {
        return;
    }

    const Point3D       query_point = query_points[tid];
    InterpolationResult result      = {};
    result.valid                    = true;

    // Check if point is outside bounds and extrapolation is needed
    const bool inside_bounds = IsPointInsideBounds(query_point, min_bound, max_bound);
    if (!inside_bounds && extrapolation_method != 0) {  // 0 = None
        if (extrapolation_method == 1) {                // 1 = NearestNeighbor
            // Apply nearest neighbor extrapolation
            size_t nearest_idx = 0;
            Real   min_dist    = 1e10f;  // Large number

            for (size_t i = 0; i < data_count; ++i) {
                Real dist = Distance(query_point, data_points[i]);
                if (dist < min_dist) {
                    min_dist    = dist;
                    nearest_idx = i;
                }
            }

            result.data = field_data[nearest_idx];
        } else if (extrapolation_method == 2) {  // 2 = LinearExtrapolation
            // Apply linear extrapolation
            result.data = LinearExtrapolate(query_point, data_points, field_data, data_count);
        }

        results[tid] = result;
        return;
    }

    // Find the cell containing the query point
    int query_ix, query_iy, query_iz;
    GetCellCoords(query_point, grid_origin, grid_cell_size, grid_dim_x, grid_dim_y, grid_dim_z, query_ix, query_iy, query_iz);

    Real              weight_sum   = 0.0f;
    MagneticFieldData weighted_sum = {};

    // Search 3x3x3 neighborhood of cells (27 cells total) with improved memory access
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
            for (int dx = -1; dx <= 1; ++dx) {
                int cell_ix = query_ix + dx;
                int cell_iy = query_iy + dy;
                int cell_iz = query_iz + dz;

                // Check bounds
                if (cell_ix < 0 || cell_ix >= static_cast<int>(grid_dim_x) || cell_iy < 0 ||
                    cell_iy >= static_cast<int>(grid_dim_y) || cell_iz < 0 ||
                    cell_iz >= static_cast<int>(grid_dim_z)) {
                    continue;
                }

                // Get cell index and point range
                size_t cell_idx  = GetCellIndex(cell_ix, cell_iy, cell_iz, grid_dim_x, grid_dim_y, grid_dim_z);
                size_t start_idx = cell_offsets[cell_idx];
                size_t end_idx   = cell_offsets[cell_idx + 1];

                // Process points in this cell with improved memory coalescing
#pragma unroll 4
                for (size_t i = start_idx; i < end_idx; ++i) {
                    uint32_t       point_idx  = cell_points[i];
                    const Point3D& data_point = data_points[point_idx];
                    Real           dx         = query_point.x - data_point.x;
                    Real           dy         = query_point.y - data_point.y;
                    Real           dz         = query_point.z - data_point.z;
                    Real           dist       = sqrtf(dx * dx + dy * dy + dz * dz);

                    // Handle exact match (avoid division by zero)
                    if (dist < DISTANCE_EPSILON) {
                        result.data  = field_data[point_idx];
                        results[tid] = result;
                        return;
                    }

                    // Use optimized power function
                    Real inv_dist_power = 1.0f / FastPow(dist, power);
                    weight_sum += inv_dist_power;

                    // Accumulate weighted field values with improved memory access
                    const MagneticFieldData& field_val = field_data[point_idx];
                    weighted_sum.Bx += field_val.Bx * inv_dist_power;
                    weighted_sum.By += field_val.By * inv_dist_power;
                    weighted_sum.Bz += field_val.Bz * inv_dist_power;
                }
            }
        }
    }

    // Normalize by weight sum
    if (weight_sum > 0.0f) {
        Real inv_weight_sum = 1.0f / weight_sum;
        weighted_sum.Bx *= inv_weight_sum;
        weighted_sum.By *= inv_weight_sum;
        weighted_sum.Bz *= inv_weight_sum;
    }

    // Derivatives are not computed for IDW (set to 0)
    result.data  = weighted_sum;
    results[tid] = result;
}

/**
 * @brief CUDA kernel: IDW interpolation for unstructured point clouds (brute force fallback)
 *
 * Each thread handles interpolation calculation for one query point
 * Computes inverse distance weighted average from all data points
 *
 * @param query_points Query points array (device memory)
 * @param data_points Data points array (device memory)
 * @param field_data Magnetic field data array (device memory)
 * @param data_count Number of data points
 * @param power IDW power parameter
 * @param results Output results array (device memory)
 * @param query_count Number of query points
 */
__global__ void IDWInterpolationKernel(const Point3D* __restrict__ query_points,
                                       const Point3D* __restrict__ data_points,
                                       const MagneticFieldData* __restrict__ field_data, const size_t data_count,
                                       const Real power, const int extrapolation_method, const Point3D min_bound,
                                       const Point3D max_bound, InterpolationResult* __restrict__ results,
                                       const size_t  query_count) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= query_count) {
        return;
    }

    const Point3D       query_point = query_points[tid];
    InterpolationResult result      = {};
    result.valid                    = true;

    // Check if point is outside bounds and extrapolation is needed
    const bool inside_bounds = IsPointInsideBounds(query_point, min_bound, max_bound);
    if (!inside_bounds && extrapolation_method != 0) {  // 0 = None
        if (extrapolation_method == 1) {                // 1 = NearestNeighbor
            // Apply nearest neighbor extrapolation
            size_t nearest_idx = 0;
            Real   min_dist    = 1e10f;  // Large number

            for (size_t i = 0; i < data_count; ++i) {
                Real dist = Distance(query_point, data_points[i]);
                if (dist < min_dist) {
                    min_dist    = dist;
                    nearest_idx = i;
                }
            }

            result.data = field_data[nearest_idx];
        } else if (extrapolation_method == 2) {  // 2 = LinearExtrapolation
            // Apply linear extrapolation
            result.data = LinearExtrapolate(query_point, data_points, field_data, data_count);
        }

        results[tid] = result;
        return;
    }

    Real              weight_sum   = 0.0f;
    MagneticFieldData weighted_sum = {};

    // Use shared memory optimization for small datasets
    const size_t block_size = blockDim.x;
    const size_t shared_data_count =
        min(data_count, block_size * SHARED_MEMORY_LIMIT_FACTOR);  // Limit shared memory usage

    extern __shared__ char shared_mem[];
    Point3D*               shared_points = reinterpret_cast<Point3D*>(shared_mem);
    MagneticFieldData*     shared_field_data =
        reinterpret_cast<MagneticFieldData*>(shared_mem + shared_data_count * sizeof(Point3D));

    // Load data into shared memory cooperatively
    for (size_t i = threadIdx.x; i < shared_data_count; i += block_size) {
        shared_points[i]     = data_points[i];
        shared_field_data[i] = field_data[i];
    }
    __syncthreads();

    // Compute IDW from all data points with optimized power calculation
    const size_t loop_count = (data_count + block_size - 1) / block_size;

#pragma unroll 4
    for (size_t block = 0; block < loop_count; ++block) {
        const size_t base_idx = block * block_size;
        const size_t end_idx  = min(base_idx + block_size, data_count);

        for (size_t i = base_idx; i < end_idx; ++i) {
            Real                     dist;
            const MagneticFieldData* field_ptr;

            // Use shared memory for frequently accessed data
            if (i < shared_data_count) {
                const Point3D& data_point = shared_points[i];
                Real           dx         = query_point.x - data_point.x;
                Real           dy         = query_point.y - data_point.y;
                Real           dz         = query_point.z - data_point.z;
                dist                      = sqrtf(dx * dx + dy * dy + dz * dz);
                field_ptr                 = &shared_field_data[i];
            } else {
                dist      = Distance(query_point, data_points[i]);
                field_ptr = &field_data[i];
            }

            // Handle exact match (avoid division by zero)
            if (dist < DISTANCE_EPSILON) {
                result.data  = *field_ptr;
                results[tid] = result;
                return;
            }

            // Use optimized power function instead of slow powf
            Real inv_dist_power = 1.0f / FastPow(dist, power);
            weight_sum += inv_dist_power;

            // Accumulate weighted field values
            weighted_sum.Bx += field_ptr->Bx * inv_dist_power;
            weighted_sum.By += field_ptr->By * inv_dist_power;
            weighted_sum.Bz += field_ptr->Bz * inv_dist_power;
        }
    }

    // Normalize by weight sum
    if (weight_sum > 0.0f) {
        Real inv_weight_sum = 1.0f / weight_sum;
        weighted_sum.Bx *= inv_weight_sum;
        weighted_sum.By *= inv_weight_sum;
        weighted_sum.Bz *= inv_weight_sum;
    }

    // Derivatives are not computed for IDW (set to 0)
    result.data  = weighted_sum;
    results[tid] = result;
}

/**
 * @brief CUDA kernel: Optimized tricubic Hermite interpolation
 *
 * Each thread handles interpolation calculation for one query point
 * Uses Hermite interpolation for gradients with available derivatives
 *
 * @param query_points Query points array (device memory)
 * @param grid_data Grid data array (device memory)
 * @param grid_params Grid parameters
 * @param results Output results array (device memory)
 * @param count Number of query points
 */
__global__ void TricubicHermiteInterpolationKernel(const Point3D* __restrict__ query_points,
                                                   const MagneticFieldData* __restrict__ grid_data,
                                                   const GridParams grid_params,
                                                   InterpolationResult* __restrict__ results, const size_t count,
                                                   const int extrapolation_method) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= count) {
        return;
    }

    // Use shared memory for grid parameters accessed by all threads in block
    extern __shared__ char shared_mem[];
    GridParams*            shared_grid_params = reinterpret_cast<GridParams*>(shared_mem);

    // Load grid parameters to shared memory (only first thread)
    if (threadIdx.x == 0) {
        *shared_grid_params = grid_params;
    }
    __syncthreads();

    // Use __restrict__ keyword to help compiler optimization
    const Point3D query_point = query_points[tid];

    // Directly initialize struct to avoid constructor call
    InterpolationResult result = {};

    // Pre-compute grid parameters from shared memory (optimize access)
    const Point3D  origin  = shared_grid_params->origin;
    const Point3D  spacing = shared_grid_params->spacing;
    const uint32_t nx      = shared_grid_params->dimensions[0];
    const uint32_t ny      = shared_grid_params->dimensions[1];
    const uint32_t nz      = shared_grid_params->dimensions[2];

    // Fast bounds checking
    const bool inside_bounds = (query_point.x >= origin.x && query_point.x <= origin.x + (nx - 1) * spacing.x) &&
                               (query_point.y >= origin.y && query_point.y <= origin.y + (ny - 1) * spacing.y) &&
                               (query_point.z >= origin.z && query_point.z <= origin.z + (nz - 1) * spacing.z);

    if (!inside_bounds) {
        if (extrapolation_method != 0) {  // 0 = None
            // Clamp to boundary
            Point3D boundary_point = {max(grid_params.min_bound.x, min(grid_params.max_bound.x, query_point.x)),
                                      max(grid_params.min_bound.y, min(grid_params.max_bound.y, query_point.y)),
                                      max(grid_params.min_bound.z, min(grid_params.max_bound.z, query_point.z))};

            // Interpolate at boundary point (recursive call would be inefficient, so inline the logic)
            // For simplicity, implement nearest neighbor by clamping and interpolating
            // For linear extrapolation, would need to compute gradient, but that's complex in GPU
            // For now, implement nearest neighbor

            // Convert boundary point to grid coordinates
            Real grid_x_b = (boundary_point.x - grid_params.origin.x) / grid_params.spacing.x;
            Real grid_y_b = (boundary_point.y - grid_params.origin.y) / grid_params.spacing.y;
            Real grid_z_b = (boundary_point.z - grid_params.origin.z) / grid_params.spacing.z;

            int i0_b = __float2int_rd(grid_x_b);
            int j0_b = __float2int_rd(grid_y_b);
            int k0_b = __float2int_rd(grid_z_b);

            bool valid_cell_b = (i0_b >= 0) && (i0_b < static_cast<int>(grid_params.dimensions[0]) - 1) &&
                                (j0_b >= 0) && (j0_b < static_cast<int>(grid_params.dimensions[1]) - 1) &&
                                (k0_b >= 0) && (k0_b < static_cast<int>(grid_params.dimensions[2]) - 1);

            if (valid_cell_b) {
                // Interpolate at boundary
                Real tx_b = grid_x_b - __int2float_rn(i0_b);
                Real ty_b = grid_y_b - __int2float_rn(j0_b);
                Real tz_b = grid_z_b - __int2float_rn(k0_b);

                uint32_t base_idx_b = i0_b + j0_b * grid_params.dimensions[0] +
                                      k0_b * grid_params.dimensions[0] * grid_params.dimensions[1];
                uint32_t idx_100_b = base_idx_b + 1;
                uint32_t idx_010_b = base_idx_b + grid_params.dimensions[0];
                uint32_t idx_110_b = base_idx_b + grid_params.dimensions[0] + 1;
                uint32_t idx_001_b = base_idx_b + grid_params.dimensions[0] * grid_params.dimensions[1];
                uint32_t idx_101_b = base_idx_b + grid_params.dimensions[0] * grid_params.dimensions[1] + 1;
                uint32_t idx_011_b =
                    base_idx_b + grid_params.dimensions[0] * grid_params.dimensions[1] + grid_params.dimensions[0];
                uint32_t idx_111_b =
                    base_idx_b + grid_params.dimensions[0] * grid_params.dimensions[1] + grid_params.dimensions[0] + 1;

                // Load vertex data
                MagneticFieldData vertex_data_b[8];
                vertex_data_b[0] = grid_data[base_idx_b];
                vertex_data_b[1] = grid_data[idx_100_b];
                vertex_data_b[2] = grid_data[idx_010_b];
                vertex_data_b[3] = grid_data[idx_110_b];
                vertex_data_b[4] = grid_data[idx_001_b];
                vertex_data_b[5] = grid_data[idx_101_b];
                vertex_data_b[6] = grid_data[idx_011_b];
                vertex_data_b[7] = grid_data[idx_111_b];

                result.data  = TricubicHermiteInterpolate(vertex_data_b, tx_b, ty_b, tz_b, grid_params);
                result.valid = true;
            }
        }
        results[tid] = result;
        return;
    }

    // Optimized coordinate transformation (avoid function call overhead)
    const Real grid_x = (query_point.x - origin.x) / spacing.x;
    const Real grid_y = (query_point.y - origin.y) / spacing.y;
    const Real grid_z = (query_point.z - origin.z) / spacing.z;

    // Fast floor and bounds checking
    const int i0 = __float2int_rd(grid_x);
    const int j0 = __float2int_rd(grid_y);
    const int k0 = __float2int_rd(grid_z);

    // Combine bounds checking to reduce branching
    const bool valid_cell = (i0 >= 0) && (i0 < static_cast<int>(nx) - 1) && (j0 >= 0) &&
                            (j0 < static_cast<int>(ny) - 1) && (k0 >= 0) && (k0 < static_cast<int>(nz) - 1);

    if (!valid_cell) {
        results[tid] = result;
        return;
    }

    // Calculate local coordinates (using fast math)
    const Real tx = grid_x - __int2float_rn(i0);
    const Real ty = grid_y - __int2float_rn(j0);
    const Real tz = grid_z - __int2float_rn(k0);

    // Directly calculate indices to avoid function calls
    const uint32_t base_idx = i0 + j0 * nx + k0 * nx * ny;
    const uint32_t idx_100  = base_idx + 1;                 // (i1, j0, k0)
    const uint32_t idx_010  = base_idx + nx;                // (i0, j1, k0)
    const uint32_t idx_110  = base_idx + nx + 1;            // (i1, j1, k0)
    const uint32_t idx_001  = base_idx + nx * ny;           // (i0, j0, k1)
    const uint32_t idx_101  = base_idx + nx * ny + 1;       // (i1, j0, k1)
    const uint32_t idx_011  = base_idx + nx * ny + nx;      // (i0, j1, k1)
    const uint32_t idx_111  = base_idx + nx * ny + nx + 1;  // (i1, j1, k1)

    // Use shared memory for vertex data to improve memory access patterns
    MagneticFieldData* shared_vertex_data = reinterpret_cast<MagneticFieldData*>(shared_mem + sizeof(GridParams));

    // Load vertex data into shared memory cooperatively
    if (threadIdx.x < 8) {
        const uint32_t indices[8]       = {base_idx, idx_100, idx_010, idx_110, idx_001, idx_101, idx_011, idx_111};
        shared_vertex_data[threadIdx.x] = grid_data[indices[threadIdx.x]];
    }
    __syncthreads();

    // Access vertex data from shared memory
    const MagneticFieldData* vertex_data = shared_vertex_data;

    // Perform tricubic Hermite interpolation
    result.data  = TricubicHermiteInterpolate(vertex_data, tx, ty, tz, grid_params);
    result.valid = true;

    // Write back result
    results[tid] = result;
}

}  // namespace cuda
}  // namespace p3d
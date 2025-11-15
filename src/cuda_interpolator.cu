#include "point3d_interp/types.h"
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
 * @brief Check if a point is inside the bounding box
 */
__device__ bool IsPointInsideBounds(const Point3D& point, const Point3D& min_bound, const Point3D& max_bound) {
    return (point.x >= min_bound.x && point.x <= max_bound.x) && (point.y >= min_bound.y && point.y <= max_bound.y) &&
           (point.z >= min_bound.z && point.z <= max_bound.z);
}

/**
 * @brief Find k nearest neighbors and compute linear extrapolation
 */
__device__ MagneticFieldData LinearExtrapolate(const Point3D& query_point, const Point3D* data_points,
                                               const MagneticFieldData* field_data, size_t data_count) {
    const size_t max_neighbors    = 5;
    size_t       actual_neighbors = min(max_neighbors, data_count);

    // Find nearest neighbors (simple bubble sort for small k)
    size_t nearest_indices[5];
    Real   nearest_distances[5];

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

    if (actual_neighbors >= 2) {
        // Use nearest point as base
        size_t            nearest_idx   = nearest_indices[0];
        Point3D           nearest_point = data_points[nearest_idx];
        MagneticFieldData nearest_data  = field_data[nearest_idx];

        // Calculate average gradient from neighbors
        Real   avg_dBx = 0, avg_dBy = 0, avg_dBz = 0;
        size_t gradient_count = 0;

        for (size_t i = 1; i < actual_neighbors; ++i) {
            size_t            idx = nearest_indices[i];
            Point3D           p   = data_points[idx];
            MagneticFieldData d   = field_data[idx];

            Real dx = p.x - nearest_point.x;
            Real dy = p.y - nearest_point.y;
            Real dz = p.z - nearest_point.z;

            Real dist = sqrtf(dx * dx + dy * dy + dz * dz);
            if (dist > 1e-8f) {
                avg_dBx += (d.Bx - nearest_data.Bx) / dist;
                avg_dBy += (d.By - nearest_data.By) / dist;
                avg_dBz += (d.Bz - nearest_data.Bz) / dist;
                gradient_count++;
            }
        }

        if (gradient_count > 0) {
            avg_dBx /= gradient_count;
            avg_dBy /= gradient_count;
            avg_dBz /= gradient_count;

            // Extrapolate from nearest point
            Real dx   = query_point.x - nearest_point.x;
            Real dy   = query_point.y - nearest_point.y;
            Real dz   = query_point.z - nearest_point.z;
            Real dist = sqrtf(dx * dx + dy * dy + dz * dz);

            if (dist > 1e-8f) {
                result.Bx = nearest_data.Bx + avg_dBx * dist;
                result.By = nearest_data.By + avg_dBy * dist;
                result.Bz = nearest_data.Bz + avg_dBz * dist;
            } else {
                result = nearest_data;
            }
        } else {
            result = nearest_data;
        }
    } else {
        // Fallback to nearest neighbor
        result = field_data[nearest_indices[0]];
    }

    return result;
}
/**
 * @brief CUDA kernel: IDW interpolation for unstructured point clouds
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

    // Compute IDW from all data points
    for (size_t i = 0; i < data_count; ++i) {
        Real dist = Distance(query_point, data_points[i]);

        // Handle exact match (avoid division by zero)
        if (dist < 1e-8f) {
            result.data  = field_data[i];
            results[tid] = result;
            return;
        }

        Real weight = 1.0f / powf(dist, power);
        weight_sum += weight;

        // Accumulate weighted field values
        weighted_sum.Bx += field_data[i].Bx * weight;
        weighted_sum.By += field_data[i].By * weight;
        weighted_sum.Bz += field_data[i].Bz * weight;
    }

    // Normalize by weight sum
    if (weight_sum > 0.0f) {
        weighted_sum.Bx /= weight_sum;
        weighted_sum.By /= weight_sum;
        weighted_sum.Bz /= weight_sum;
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
                                                   InterpolationResult* __restrict__ results, const size_t count) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= count) {
        return;
    }

    // Use __restrict__ keyword to help compiler optimization
    const Point3D query_point = query_points[tid];

    // Directly initialize struct to avoid constructor call
    InterpolationResult result = {};

    // Pre-compute grid parameters to registers (optimize access)
    const Point3D  origin  = grid_params.origin;
    const Point3D  spacing = grid_params.spacing;
    const uint32_t nx      = grid_params.dimensions[0];
    const uint32_t ny      = grid_params.dimensions[1];
    const uint32_t nz      = grid_params.dimensions[2];

    // Fast bounds checking
    const bool inside_bounds = (query_point.x >= origin.x && query_point.x <= origin.x + (nx - 1) * spacing.x) &&
                               (query_point.y >= origin.y && query_point.y <= origin.y + (ny - 1) * spacing.y) &&
                               (query_point.z >= origin.z && query_point.z <= origin.z + (nz - 1) * spacing.z);

    if (!inside_bounds) {
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

    // Directly access data to avoid array copying
    const MagneticFieldData vertex_data[8] = {
        grid_data[base_idx],  // v000 (i0, j0, k0)
        grid_data[idx_100],   // v100 (i1, j0, k0)
        grid_data[idx_010],   // v010 (i0, j1, k0)
        grid_data[idx_110],   // v110 (i1, j1, k0)
        grid_data[idx_001],   // v001 (i0, j0, k1)
        grid_data[idx_101],   // v101 (i1, j0, k1)
        grid_data[idx_011],   // v011 (i0, j1, k1)
        grid_data[idx_111]    // v111 (i1, j1, k1)
    };

    // Perform tricubic Hermite interpolation
    result.data  = TricubicHermiteInterpolate(vertex_data, tx, ty, tz, grid_params);
    result.valid = true;

    // Write back result
    results[tid] = result;
}

}  // namespace cuda
}  // namespace p3d
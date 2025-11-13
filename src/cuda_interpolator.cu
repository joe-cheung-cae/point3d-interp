#include "point3d_interp/types.h"
#include <cuda_runtime.h>

namespace p3d {
namespace cuda {

/**
 * @brief Convert world coordinates to grid coordinates
 */
__device__ Point3D WorldToGrid(const Point3D& world_point, const GridParams& params) {
    return Point3D((world_point.x - params.origin.x) / params.spacing.x, (world_point.y - params.origin.y) / params.spacing.y,
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
    if (i0 < 0 || i1 >= static_cast<int>(params.dimensions[0]) || j0 < 0 || j1 >= static_cast<int>(params.dimensions[1]) || k0 < 0 ||
        k1 >= static_cast<int>(params.dimensions[2])) {
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
 * @brief Perform trilinear interpolation
 */
__device__ MagneticFieldData TrilinearInterpolate(const MagneticFieldData vertex_data[8], Real tx, Real ty, Real tz) {
    MagneticFieldData result;

    // Trilinear interpolation algorithm
    // Vertex indices correspond to:
    // 0: (i,j,k),     1: (i+1,j,k),   2: (i,j+1,k),   3: (i+1,j+1,k)
    // 4: (i,j,k+1),   5: (i+1,j,k+1), 6: (i,j+1,k+1), 7: (i+1,j+1,k+1)

    // X-direction interpolation (4 times)
    MagneticFieldData c00 = vertex_data[0] * (1 - tx) + vertex_data[1] * tx;  // (i,j,k) -> (i+1,j,k)
    MagneticFieldData c01 = vertex_data[4] * (1 - tx) + vertex_data[5] * tx;  // (i,j,k+1) -> (i+1,j,k+1)
    MagneticFieldData c10 = vertex_data[2] * (1 - tx) + vertex_data[3] * tx;  // (i,j+1,k) -> (i+1,j+1,k)
    MagneticFieldData c11 = vertex_data[6] * (1 - tx) + vertex_data[7] * tx;  // (i,j+1,k+1) -> (i+1,j+1,k+1)

    // Y-direction interpolation (2 times)
    MagneticFieldData c0 = c00 * (1 - ty) + c10 * ty;  // Merge k layer
    MagneticFieldData c1 = c01 * (1 - ty) + c11 * ty;  // Merge k+1 layer

    // Z-direction interpolation (1 time)
    result = c0 * (1 - tz) + c1 * tz;

    return result;
}

/**
 * @brief CUDA kernel: Optimized trilinear interpolation
 *
 * Each thread handles interpolation calculation for one query point
 * Optimization strategies:
 * 1. Reduce branch divergence
 * 2. Use texture memory or constant memory for grid parameters
 * 3. Optimize memory access patterns
 * 4. Use fast math functions
 *
 * @param query_points Query points array (device memory)
 * @param grid_data Grid data array (device memory)
 * @param grid_params Grid parameters
 * @param results Output results array (device memory)
 * @param count Number of query points
 */
__global__ void TrilinearInterpolationKernel(const Point3D* __restrict__ query_points, const MagneticFieldData* __restrict__ grid_data,
                                             const GridParams grid_params, InterpolationResult* __restrict__ results, const size_t count) {
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
    const bool valid_cell = (i0 >= 0) && (i0 < static_cast<int>(nx) - 1) && (j0 >= 0) && (j0 < static_cast<int>(ny) - 1) && (k0 >= 0) &&
                            (k0 < static_cast<int>(nz) - 1);

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
    const MagneticFieldData v000 = grid_data[base_idx];  // (i0, j0, k0)
    const MagneticFieldData v100 = grid_data[idx_100];   // (i1, j0, k0)
    const MagneticFieldData v010 = grid_data[idx_010];   // (i0, j1, k0)
    const MagneticFieldData v110 = grid_data[idx_110];   // (i1, j1, k0)
    const MagneticFieldData v001 = grid_data[idx_001];   // (i0, j0, k1)
    const MagneticFieldData v101 = grid_data[idx_101];   // (i1, j0, k1)
    const MagneticFieldData v011 = grid_data[idx_011];   // (i0, j1, k1)
    const MagneticFieldData v111 = grid_data[idx_111];   // (i1, j1, k1)

    // Pre-compute weights
    const Real tx_inv = 1.0f - tx;
    const Real ty_inv = 1.0f - ty;
    const Real tz_inv = 1.0f - tz;

    // Optimized trilinear interpolation calculation
    // X-direction interpolation (using pre-computed weights)
    const MagneticFieldData c00 = v000 * tx_inv + v100 * tx;
    const MagneticFieldData c01 = v001 * tx_inv + v101 * tx;
    const MagneticFieldData c10 = v010 * tx_inv + v110 * tx;
    const MagneticFieldData c11 = v011 * tx_inv + v111 * tx;

    // Y-direction interpolation
    const MagneticFieldData c0 = c00 * ty_inv + c10 * ty;
    const MagneticFieldData c1 = c01 * ty_inv + c11 * ty;

    // Z-direction interpolation (final result)
    result.data  = c0 * tz_inv + c1 * tz;
    result.valid = true;

    // Write back result
    results[tid] = result;
}

}  // namespace cuda
}  // namespace p3d
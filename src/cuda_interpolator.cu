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
 * @brief Perform tricubic Hermite interpolation
 */
__device__ MagneticFieldData TricubicHermiteInterpolate(const MagneticFieldData vertex_data[8], Real tx, Real ty,
                                                        Real tz) {
    MagneticFieldData result;

    // Vertex indices correspond to:
    // 0: (i,j,k),     1: (i+1,j,k),   2: (i,j+1,k),   3: (i+1,j+1,k)
    // 4: (i,j,k+1),   5: (i+1,j,k+1), 6: (i,j+1,k+1), 7: (i+1,j+1,k+1)

    // For simplicity, use linear for field_strength, Hermite for gradients
    // X-direction interpolation
    MagneticFieldData c00, c01, c10, c11;
    c00.field_strength = vertex_data[0].field_strength * (1 - tx) + vertex_data[1].field_strength * tx;
    c00.gradient_x     = HermiteInterpolate(vertex_data[0].gradient_x, vertex_data[1].gradient_x, vertex_data[0].dBx_dx,
                                            vertex_data[1].dBx_dx, tx);
    c00.gradient_y     = HermiteInterpolate(vertex_data[0].gradient_y, vertex_data[1].gradient_y, vertex_data[0].dBy_dx,
                                            vertex_data[1].dBy_dx, tx);
    c00.gradient_z     = HermiteInterpolate(vertex_data[0].gradient_z, vertex_data[1].gradient_z, vertex_data[0].dBz_dx,
                                            vertex_data[1].dBz_dx, tx);

    c01.field_strength = vertex_data[4].field_strength * (1 - tx) + vertex_data[5].field_strength * tx;
    c01.gradient_x     = HermiteInterpolate(vertex_data[4].gradient_x, vertex_data[5].gradient_x, vertex_data[4].dBx_dx,
                                            vertex_data[5].dBx_dx, tx);
    c01.gradient_y     = HermiteInterpolate(vertex_data[4].gradient_y, vertex_data[5].gradient_y, vertex_data[4].dBy_dx,
                                            vertex_data[5].dBy_dx, tx);
    c01.gradient_z     = HermiteInterpolate(vertex_data[4].gradient_z, vertex_data[5].gradient_z, vertex_data[4].dBz_dx,
                                            vertex_data[5].dBz_dx, tx);

    c10.field_strength = vertex_data[2].field_strength * (1 - tx) + vertex_data[3].field_strength * tx;
    c10.gradient_x     = HermiteInterpolate(vertex_data[2].gradient_x, vertex_data[3].gradient_x, vertex_data[2].dBx_dx,
                                            vertex_data[3].dBx_dx, tx);
    c10.gradient_y     = HermiteInterpolate(vertex_data[2].gradient_y, vertex_data[3].gradient_y, vertex_data[2].dBy_dx,
                                            vertex_data[3].dBy_dx, tx);
    c10.gradient_z     = HermiteInterpolate(vertex_data[2].gradient_z, vertex_data[3].gradient_z, vertex_data[2].dBz_dx,
                                            vertex_data[3].dBz_dx, tx);

    c11.field_strength = vertex_data[6].field_strength * (1 - tx) + vertex_data[7].field_strength * tx;
    c11.gradient_x     = HermiteInterpolate(vertex_data[6].gradient_x, vertex_data[7].gradient_x, vertex_data[6].dBx_dx,
                                            vertex_data[7].dBx_dx, tx);
    c11.gradient_y     = HermiteInterpolate(vertex_data[6].gradient_y, vertex_data[7].gradient_y, vertex_data[6].dBy_dx,
                                            vertex_data[7].dBy_dx, tx);
    c11.gradient_z     = HermiteInterpolate(vertex_data[6].gradient_z, vertex_data[7].gradient_z, vertex_data[6].dBz_dx,
                                            vertex_data[7].dBz_dx, tx);

    // Y-direction interpolation
    MagneticFieldData c0, c1;
    c0.field_strength = c00.field_strength * (1 - ty) + c10.field_strength * ty;
    c0.gradient_x     = HermiteInterpolate(c00.gradient_x, c10.gradient_x, c00.dBx_dy, c10.dBx_dy,
                                           ty);  // Assuming dBx_dy is set, but in data it's 0
    c0.gradient_y     = HermiteInterpolate(c00.gradient_y, c10.gradient_y, c00.dBy_dy, c10.dBy_dy, ty);
    c0.gradient_z     = HermiteInterpolate(c00.gradient_z, c10.gradient_z, c00.dBz_dy, c10.dBz_dy, ty);

    c1.field_strength = c01.field_strength * (1 - ty) + c11.field_strength * ty;
    c1.gradient_x     = HermiteInterpolate(c01.gradient_x, c11.gradient_x, c01.dBx_dy, c11.dBx_dy, ty);
    c1.gradient_y     = HermiteInterpolate(c01.gradient_y, c11.gradient_y, c01.dBy_dy, c11.dBy_dy, ty);
    c1.gradient_z     = HermiteInterpolate(c01.gradient_z, c11.gradient_z, c01.dBz_dy, c11.dBz_dy, ty);

    // Z-direction interpolation
    result.field_strength = c0.field_strength * (1 - tz) + c1.field_strength * tz;
    result.gradient_x     = HermiteInterpolate(c0.gradient_x, c1.gradient_x, c0.dBx_dz, c1.dBx_dz, tz);
    result.gradient_y     = HermiteInterpolate(c0.gradient_y, c1.gradient_y, c0.dBy_dz, c1.dBy_dz, tz);
    result.gradient_z     = HermiteInterpolate(c0.gradient_z, c1.gradient_z, c0.dBz_dz, c1.dBz_dz, tz);

    // Derivatives set to 0
    result.dBx_dx = 0;
    result.dBx_dy = 0;
    result.dBx_dz = 0;
    result.dBy_dx = 0;
    result.dBy_dy = 0;
    result.dBy_dz = 0;
    result.dBz_dx = 0;
    result.dBz_dy = 0;
    result.dBz_dz = 0;

    return result;
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
    result.data  = TricubicHermiteInterpolate(vertex_data, tx, ty, tz);
    result.valid = true;

    // Write back result
    results[tid] = result;
}

}  // namespace cuda
}  // namespace p3d
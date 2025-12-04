#include "point3d_interp/cpu_interpolator.h"
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
    #include <omp.h>
#endif

#ifdef ENABLE_SIMD
    #include <immintrin.h>
#endif

P3D_NAMESPACE_BEGIN

CPUInterpolator::CPUInterpolator(const RegularGrid3D& grid, ExtrapolationMethod extrapolation_method)
    : grid_ptr_(&grid), extrapolation_method_(extrapolation_method) {}

CPUInterpolator::~CPUInterpolator() = default;

CPUInterpolator::CPUInterpolator(CPUInterpolator&& other) noexcept
    : grid_ptr_(other.grid_ptr_), extrapolation_method_(other.extrapolation_method_) {}

CPUInterpolator& CPUInterpolator::operator=(CPUInterpolator&& other) noexcept {
    // const reference cannot be reassigned, nothing to do here
    // grid_ is initialized in constructor and cannot be changed
    return *this;
}

InterpolationResult CPUInterpolator::query(const Point3D& query_point) const {
    InterpolationResult result;

    auto& params = (*grid_ptr_).getParams();

    // Special case for single point grid
    if (params.dimensions[0] == 1 && params.dimensions[1] == 1 && params.dimensions[2] == 1) {
        if (query_point.x == params.origin.x && query_point.y == params.origin.y && query_point.z == params.origin.z) {
            result.data  = (*grid_ptr_).getFieldData()[0];
            result.valid = true;
            return result;
        } else {
            result.valid = false;
            return result;
        }
    }

    // Check if point is within grid bounds
    if (!params.is_point_inside(query_point)) {
        if (extrapolation_method_ != ExtrapolationMethod::None) {
            return extrapolate(query_point);
        } else {
            result.valid = false;
            return result;
        }
    }

    // Convert to grid coordinates
    Point3D grid_coords = (*grid_ptr_).worldToGrid(query_point);

    // Check if grid coordinates are valid
    if (!(*grid_ptr_).isValidGridCoords(grid_coords)) {
        result.valid = false;
        return result;
    }

    // Get cell vertex indices
    uint32_t indices[8];
    if (!(*grid_ptr_).getCellVertexIndices(grid_coords, indices)) {
        result.valid = false;
        return result;
    }

    // Get vertex data
    MagneticFieldData vertex_data[8];
    getVertexData(indices, vertex_data);

    // Calculate local coordinates (between 0 and 1)
    // Adjust for boundary cases to ensure tx, ty, tz are correct
    int  i0_x = std::min(static_cast<int>(std::floor(grid_coords.x)), static_cast<int>(params.dimensions[0]) - 2);
    int  i0_y = std::min(static_cast<int>(std::floor(grid_coords.y)), static_cast<int>(params.dimensions[1]) - 2);
    int  i0_z = std::min(static_cast<int>(std::floor(grid_coords.z)), static_cast<int>(params.dimensions[2]) - 2);
    Real tx   = grid_coords.x - i0_x;
    Real ty   = grid_coords.y - i0_y;
    Real tz   = grid_coords.z - i0_z;

    // Perform tricubic Hermite interpolation
    result.data  = tricubicHermiteInterpolate(vertex_data, tx, ty, tz);
    result.valid = true;

    return result;
}

InterpolationResult CPUInterpolator::extrapolate(const Point3D& query_point) const {
    InterpolationResult result;
    result.valid = false;

    const auto& params = (*grid_ptr_).getParams();

    if (extrapolation_method_ == ExtrapolationMethod::NearestNeighbor) {
        // Clamp to nearest boundary point
        Point3D boundary_point = {std::max(params.min_bound.x, std::min(params.max_bound.x, query_point.x)),
                                  std::max(params.min_bound.y, std::min(params.max_bound.y, query_point.y)),
                                  std::max(params.min_bound.z, std::min(params.max_bound.z, query_point.z))};

        // Interpolate at boundary point
        result = query(boundary_point);
    } else if (extrapolation_method_ == ExtrapolationMethod::LinearExtrapolation) {
        // Clamp to nearest boundary point
        Point3D boundary_point = {std::max(params.min_bound.x, std::min(params.max_bound.x, query_point.x)),
                                  std::max(params.min_bound.y, std::min(params.max_bound.y, query_point.y)),
                                  std::max(params.min_bound.z, std::min(params.max_bound.z, query_point.z))};

        // Interpolate at boundary point
        InterpolationResult boundary_result = query(boundary_point);
        if (!boundary_result.valid) {
            return result;  // Should not happen
        }

        // For linear extrapolation, we need the gradient at the boundary
        // Approximate using finite differences or use the interpolated derivatives
        // For simplicity, use the interpolated field derivatives as gradient

        Point3D delta = query_point - boundary_point;

        // Extrapolate: value = boundary_value + gradient · delta
        result.data.Bx = boundary_result.data.Bx + boundary_result.data.dBx_dx * delta.x +
                         boundary_result.data.dBx_dy * delta.y + boundary_result.data.dBx_dz * delta.z;
        result.data.By = boundary_result.data.By + boundary_result.data.dBy_dx * delta.x +
                         boundary_result.data.dBy_dy * delta.y + boundary_result.data.dBy_dz * delta.z;
        result.data.Bz = boundary_result.data.Bz + boundary_result.data.dBz_dx * delta.x +
                         boundary_result.data.dBz_dy * delta.y + boundary_result.data.dBz_dz * delta.z;

        // Derivatives remain the same as boundary (constant extrapolation)
        result.data.dBx_dx = boundary_result.data.dBx_dx;
        result.data.dBx_dy = boundary_result.data.dBx_dy;
        result.data.dBx_dz = boundary_result.data.dBx_dz;
        result.data.dBy_dx = boundary_result.data.dBy_dx;
        result.data.dBy_dy = boundary_result.data.dBy_dy;
        result.data.dBy_dz = boundary_result.data.dBy_dz;
        result.data.dBz_dx = boundary_result.data.dBz_dx;
        result.data.dBz_dy = boundary_result.data.dBz_dy;
        result.data.dBz_dz = boundary_result.data.dBz_dz;

        result.valid = true;
    }

    return result;
}

std::vector<InterpolationResult> CPUInterpolator::queryBatch(const std::vector<Point3D>& query_points) const {
    std::vector<InterpolationResult> results(query_points.size());

    // Use OpenMP for parallel processing if available
#ifdef _OPENMP
    #pragma omp parallel for if (query_points.size() > 100) schedule(dynamic, 16)
#endif
    for (size_t i = 0; i < query_points.size(); ++i) {
        results[i] = query(query_points[i]);
    }

    return results;
}

Real CPUInterpolator::hermiteInterpolate(Real f0, Real f1, Real df0, Real df1, Real t) const {
    Real t2  = t * t;
    Real t3  = t2 * t;
    Real h00 = 2 * t3 - 3 * t2 + 1;
    Real h10 = t3 - 2 * t2 + t;
    Real h01 = -2 * t3 + 3 * t2;
    Real h11 = t3 - t2;
    return f0 * h00 + df0 * h10 + f1 * h01 + df1 * h11;
}

Real CPUInterpolator::hermiteDerivative(Real f0, Real f1, Real df0, Real df1, Real t) const {
    Real t2      = t * t;
    Real dh00_dt = 6 * t2 - 6 * t;
    Real dh10_dt = 3 * t2 - 4 * t + 1;
    Real dh01_dt = -6 * t2 + 6 * t;
    Real dh11_dt = 3 * t2 - 2 * t;
    return f0 * dh00_dt + df0 * dh10_dt + f1 * dh01_dt + df1 * dh11_dt;
}

MagneticFieldData CPUInterpolator::tricubicHermiteInterpolate(const MagneticFieldData vertex_data[8], Real tx, Real ty,
                                                              Real tz) const {
    MagneticFieldData result;

    const auto& spacing = (*grid_ptr_).getParams().spacing;

    // Vertex indices correspond to:
    // 0: (i,j,k),     1: (i+1,j,k),   2: (i,j+1,k),   3: (i+1,j+1,k)
    // 4: (i,j,k+1),   5: (i+1,j,k+1), 6: (i,j+1,k+1), 7: (i+1,j+1,k+1)

    // Interpolate along x for each of the 4 yz positions
    MagneticFieldData interp_x[4];
    for (int i = 0; i < 4; ++i) {
        int idx0 = (i % 2) * 2 + (i / 2) * 4;  // 0,2,4,6 for i=0,1,2,3
        int idx1 = idx0 + 1;                   // 1,3,5,7

        // Interpolate field values using Hermite
        interp_x[i].Bx = hermiteInterpolate(vertex_data[idx0].Bx, vertex_data[idx1].Bx, vertex_data[idx0].dBx_dx,
                                            vertex_data[idx1].dBx_dx, tx);
        interp_x[i].By = hermiteInterpolate(vertex_data[idx0].By, vertex_data[idx1].By, vertex_data[idx0].dBy_dx,
                                            vertex_data[idx1].dBy_dx, tx);
        interp_x[i].Bz = hermiteInterpolate(vertex_data[idx0].Bz, vertex_data[idx1].Bz, vertex_data[idx0].dBz_dx,
                                            vertex_data[idx1].dBz_dx, tx);

        // Compute derivatives: dBx_dx is derivative of Hermite in x direction
        interp_x[i].dBx_dx = hermiteDerivative(vertex_data[idx0].Bx, vertex_data[idx1].Bx, vertex_data[idx0].dBx_dx,
                                               vertex_data[idx1].dBx_dx, tx);
        // Other derivatives interpolated linearly
        interp_x[i].dBx_dy = (1 - tx) * vertex_data[idx0].dBx_dy + tx * vertex_data[idx1].dBx_dy;
        interp_x[i].dBx_dz = (1 - tx) * vertex_data[idx0].dBx_dz + tx * vertex_data[idx1].dBx_dz;
        interp_x[i].dBy_dx = hermiteDerivative(vertex_data[idx0].By, vertex_data[idx1].By, vertex_data[idx0].dBy_dx,
                                               vertex_data[idx1].dBy_dx, tx);
        interp_x[i].dBy_dy = (1 - tx) * vertex_data[idx0].dBy_dy + tx * vertex_data[idx1].dBy_dy;
        interp_x[i].dBy_dz = (1 - tx) * vertex_data[idx0].dBy_dz + tx * vertex_data[idx1].dBy_dz;
        interp_x[i].dBz_dx = hermiteDerivative(vertex_data[idx0].Bz, vertex_data[idx1].Bz, vertex_data[idx0].dBz_dx,
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
            hermiteInterpolate(interp_x[idx0].Bx, interp_x[idx1].Bx, interp_x[idx0].dBx_dy, interp_x[idx1].dBx_dy, ty);
        interp_y[i].By =
            hermiteInterpolate(interp_x[idx0].By, interp_x[idx1].By, interp_x[idx0].dBy_dy, interp_x[idx1].dBy_dy, ty);
        interp_y[i].Bz =
            hermiteInterpolate(interp_x[idx0].Bz, interp_x[idx1].Bz, interp_x[idx0].dBz_dy, interp_x[idx1].dBz_dy, ty);

        // Compute derivatives
        interp_y[i].dBx_dx = (1 - ty) * interp_x[idx0].dBx_dx + ty * interp_x[idx1].dBx_dx;
        interp_y[i].dBx_dy =
            hermiteDerivative(interp_x[idx0].Bx, interp_x[idx1].Bx, interp_x[idx0].dBx_dy, interp_x[idx1].dBx_dy, ty);
        interp_y[i].dBx_dz = (1 - ty) * interp_x[idx0].dBx_dz + ty * interp_x[idx1].dBx_dz;
        interp_y[i].dBy_dx = (1 - ty) * interp_x[idx0].dBy_dx + ty * interp_x[idx1].dBy_dx;
        interp_y[i].dBy_dy =
            hermiteDerivative(interp_x[idx0].By, interp_x[idx1].By, interp_x[idx0].dBy_dy, interp_x[idx1].dBy_dy, ty);
        interp_y[i].dBy_dz = (1 - ty) * interp_x[idx0].dBy_dz + ty * interp_x[idx1].dBy_dz;
        interp_y[i].dBz_dx = (1 - ty) * interp_x[idx0].dBz_dx + ty * interp_x[idx1].dBz_dx;
        interp_y[i].dBz_dy =
            hermiteDerivative(interp_x[idx0].Bz, interp_x[idx1].Bz, interp_x[idx0].dBz_dy, interp_x[idx1].dBz_dy, ty);
        interp_y[i].dBz_dz = (1 - ty) * interp_x[idx0].dBz_dz + ty * interp_x[idx1].dBz_dz;
    }

    // Interpolate along z
    result.Bx = hermiteInterpolate(interp_y[0].Bx, interp_y[1].Bx, interp_y[0].dBx_dz, interp_y[1].dBx_dz, tz);
    result.By = hermiteInterpolate(interp_y[0].By, interp_y[1].By, interp_y[0].dBy_dz, interp_y[1].dBy_dz, tz);
    result.Bz = hermiteInterpolate(interp_y[0].Bz, interp_y[1].Bz, interp_y[0].dBz_dz, interp_y[1].dBz_dz, tz);

    // Compute final derivatives
    result.dBx_dx = (1 - tz) * interp_y[0].dBx_dx + tz * interp_y[1].dBx_dx;
    result.dBx_dy = (1 - tz) * interp_y[0].dBx_dy + tz * interp_y[1].dBx_dy;
    result.dBx_dz = hermiteDerivative(interp_y[0].Bx, interp_y[1].Bx, interp_y[0].dBx_dz, interp_y[1].dBx_dz, tz);
    result.dBy_dx = (1 - tz) * interp_y[0].dBy_dx + tz * interp_y[1].dBy_dx;
    result.dBy_dy = (1 - tz) * interp_y[0].dBy_dy + tz * interp_y[1].dBy_dy;
    result.dBy_dz = hermiteDerivative(interp_y[0].By, interp_y[1].By, interp_y[0].dBy_dz, interp_y[1].dBy_dz, tz);
    result.dBz_dx = (1 - tz) * interp_y[0].dBz_dx + tz * interp_y[1].dBz_dx;
    result.dBz_dy = (1 - tz) * interp_y[0].dBz_dy + tz * interp_y[1].dBz_dy;
    result.dBz_dz = hermiteDerivative(interp_y[0].Bz, interp_y[1].Bz, interp_y[0].dBz_dz, interp_y[1].dBz_dz, tz);

    // Scale derivatives to world coordinates
    result.dBx_dx /= spacing.x;
    result.dBx_dy /= spacing.y;
    result.dBx_dz /= spacing.z;
    result.dBy_dx /= spacing.x;
    result.dBy_dy /= spacing.y;
    result.dBy_dz /= spacing.z;
    result.dBz_dx /= spacing.x;
    result.dBz_dy /= spacing.y;
    result.dBz_dz /= spacing.z;

    return result;
}

void CPUInterpolator::getVertexData(const uint32_t indices[8], MagneticFieldData vertex_data[8]) const {
    const auto& field_data = (*grid_ptr_).getFieldData();

    // Cache-friendly access: load vertices in a pattern that maximizes cache hits
    // Load in order that corresponds to spatial locality
#pragma omp simd
    for (int i = 0; i < 8; ++i) {
        vertex_data[i] = field_data[indices[i]];
    }
}

#ifdef ENABLE_SIMD
/**
 * @brief SIMD-optimized Hermite interpolation using AVX2
 * Processes 4 values simultaneously for better throughput
 */
void CPUInterpolator::hermiteInterpolateSIMD(const Real f0[4], const Real f1[4], const Real df0[4], const Real df1[4],
                                             const Real t[4], Real result[4]) const {
    // Load input values into AVX registers
    __m256 f0_vec  = _mm256_loadu_ps(f0);
    __m256 f1_vec  = _mm256_loadu_ps(f1);
    __m256 df0_vec = _mm256_loadu_ps(df0);
    __m256 df1_vec = _mm256_loadu_ps(df1);
    __m256 t_vec   = _mm256_loadu_ps(t);

    // Compute powers of t
    __m256 t2 = _mm256_mul_ps(t_vec, t_vec);
    __m256 t3 = _mm256_mul_ps(t2, t_vec);

    // Hermite basis functions
    __m256 h00 =
        _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), t3), _mm256_mul_ps(_mm256_set1_ps(-3.0f), t2)),
                      _mm256_set1_ps(1.0f));
    __m256 h10 = _mm256_add_ps(_mm256_sub_ps(t3, _mm256_mul_ps(_mm256_set1_ps(2.0f), t2)), t_vec);
    __m256 h01 =
        _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(-2.0f), t3), _mm256_mul_ps(_mm256_set1_ps(3.0f), t2)),
                      _mm256_set1_ps(0.0f));
    __m256 h11 = _mm256_sub_ps(t3, t2);

    // Compute interpolation: f0*h00 + df0*h10 + f1*h01 + df1*h11
    __m256 term0 = _mm256_mul_ps(f0_vec, h00);
    __m256 term1 = _mm256_mul_ps(df0_vec, h10);
    __m256 term2 = _mm256_mul_ps(f1_vec, h01);
    __m256 term3 = _mm256_mul_ps(df1_vec, h11);

    __m256 sum = _mm256_add_ps(_mm256_add_ps(term0, term1), _mm256_add_ps(term2, term3));

    // Store result
    _mm256_storeu_ps(result, sum);
}

/**
 * @brief SIMD-optimized Hermite derivative using AVX2
 */
void CPUInterpolator::hermiteDerivativeSIMD(const Real f0[4], const Real f1[4], const Real df0[4], const Real df1[4],
                                            const Real t[4], Real result[4]) const {
    // Load input values into AVX registers
    __m256 f0_vec  = _mm256_loadu_ps(f0);
    __m256 f1_vec  = _mm256_loadu_ps(f1);
    __m256 df0_vec = _mm256_loadu_ps(df0);
    __m256 df1_vec = _mm256_loadu_ps(df1);
    __m256 t_vec   = _mm256_loadu_ps(t);

    // Compute powers of t
    __m256 t2 = _mm256_mul_ps(t_vec, t_vec);

    // Derivative of Hermite basis functions
    __m256 dh00_dt = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(6.0f), t2), _mm256_mul_ps(_mm256_set1_ps(6.0f), t_vec));
    __m256 dh10_dt = _mm256_add_ps(
        _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(3.0f), t2), _mm256_mul_ps(_mm256_set1_ps(4.0f), t_vec)),
        _mm256_set1_ps(1.0f));
    __m256 dh01_dt =
        _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(-6.0f), t2), _mm256_mul_ps(_mm256_set1_ps(-6.0f), t_vec));
    __m256 dh11_dt = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(3.0f), t2), _mm256_mul_ps(_mm256_set1_ps(2.0f), t_vec));

    // Compute derivative: f0*dh00_dt + df0*dh10_dt + f1*dh01_dt + df1*dh11_dt
    __m256 term0 = _mm256_mul_ps(f0_vec, dh00_dt);
    __m256 term1 = _mm256_mul_ps(df0_vec, dh10_dt);
    __m256 term2 = _mm256_mul_ps(f1_vec, dh01_dt);
    __m256 term3 = _mm256_mul_ps(df1_vec, dh11_dt);

    __m256 sum = _mm256_add_ps(_mm256_add_ps(term0, term1), _mm256_add_ps(term2, term3));

    // Store result
    _mm256_storeu_ps(result, sum);
}
#endif

P3D_NAMESPACE_END
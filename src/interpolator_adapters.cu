#include "point3d_interp/interpolator_adapters.h"
#include <algorithm>

// Forward declarations of CUDA kernels
namespace p3d {
namespace cuda {
__global__ void TricubicHermiteInterpolationKernel(const Point3D* __restrict__ query_points,
                                                   const MagneticFieldData* __restrict__ grid_data,
                                                   GridParams grid_params, InterpolationResult* __restrict__ results,
                                                   size_t count, int extrapolation_method);

__global__ void IDWSpatialGridKernel(const Point3D* __restrict__ query_points, const Point3D* __restrict__ data_points,
                                     const MagneticFieldData* __restrict__ field_data, const size_t data_count,
                                     const uint32_t* __restrict__ cell_offsets,
                                     const uint32_t* __restrict__ cell_points, const Point3D grid_origin,
                                     const Point3D grid_cell_size, uint32_t grid_dim_x, uint32_t grid_dim_y,
                                     uint32_t grid_dim_z, const Real power, const int extrapolation_method,
                                     const Point3D min_bound, const Point3D max_bound,
                                     InterpolationResult* __restrict__ results, const size_t query_count);

__global__ void IDWInterpolationKernel(const Point3D* __restrict__ query_points,
                                       const Point3D* __restrict__ data_points,
                                       const MagneticFieldData* __restrict__ field_data, const size_t data_count,
                                       const Real power, const int extrapolation_method, const Point3D min_bound,
                                       const Point3D max_bound, InterpolationResult* __restrict__ results,
                                       const size_t query_count);
}  // namespace cuda
}  // namespace p3d

namespace p3d {

// CPUStructuredInterpolatorAdapter implementation

CPUStructuredInterpolatorAdapter::CPUStructuredInterpolatorAdapter(
    std::unique_ptr<RegularGrid3D> grid,
    InterpolationMethod method,
    ExtrapolationMethod extrapolation)
    : grid_(std::move(grid))
    , cpu_interpolator_(std::make_unique<CPUInterpolator>(*grid_, extrapolation))
    , method_(method)
    , extrapolation_(extrapolation) {
}

InterpolationResult CPUStructuredInterpolatorAdapter::query(const Point3D& point) const {
    return cpu_interpolator_->query(point);
}

std::vector<InterpolationResult> CPUStructuredInterpolatorAdapter::queryBatch(
    const std::vector<Point3D>& points) const {
    return cpu_interpolator_->queryBatch(points);
}

size_t CPUStructuredInterpolatorAdapter::getDataCount() const {
    return grid_->getDataCount();
}

void CPUStructuredInterpolatorAdapter::getBounds(Point3D& min_bound, Point3D& max_bound) const {
    const auto& params = grid_->getParams();
    min_bound = params.min_bound;
    max_bound = params.max_bound;
}

GridParams CPUStructuredInterpolatorAdapter::getGridParams() const {
    return grid_->getParams();
}

std::vector<Point3D> CPUStructuredInterpolatorAdapter::getCoordinates() const {
    return grid_->getCoordinates();
}

std::vector<MagneticFieldData> CPUStructuredInterpolatorAdapter::getFieldData() const {
    return grid_->getFieldData();
}

// CPUUnstructuredInterpolatorAdapter implementation

CPUUnstructuredInterpolatorAdapter::CPUUnstructuredInterpolatorAdapter(
    std::unique_ptr<UnstructuredInterpolator> interpolator,
    InterpolationMethod method,
    ExtrapolationMethod extrapolation)
    : unstructured_interpolator_(std::move(interpolator))
    , method_(method)
    , extrapolation_(extrapolation) {
}

InterpolationResult CPUUnstructuredInterpolatorAdapter::query(const Point3D& point) const {
    return unstructured_interpolator_->query(point);
}

std::vector<InterpolationResult> CPUUnstructuredInterpolatorAdapter::queryBatch(
    const std::vector<Point3D>& points) const {
    return unstructured_interpolator_->queryBatch(points);
}

size_t CPUUnstructuredInterpolatorAdapter::getDataCount() const {
    return unstructured_interpolator_->getDataCount();
}

void CPUUnstructuredInterpolatorAdapter::getBounds(Point3D& min_bound, Point3D& max_bound) const {
    min_bound = unstructured_interpolator_->getMinBound();
    max_bound = unstructured_interpolator_->getMaxBound();
}

GridParams CPUUnstructuredInterpolatorAdapter::getGridParams() const {
    // Unstructured data doesn't have grid parameters, return default
    return GridParams();
}

std::vector<Point3D> CPUUnstructuredInterpolatorAdapter::getCoordinates() const {
    return unstructured_interpolator_->getCoordinates();
}

std::vector<MagneticFieldData> CPUUnstructuredInterpolatorAdapter::getFieldData() const {
    return unstructured_interpolator_->getFieldData();
}

// GPUStructuredInterpolatorAdapter implementation

GPUStructuredInterpolatorAdapter::GPUStructuredInterpolatorAdapter(
    std::unique_ptr<RegularGrid3D> grid,
    InterpolationMethod method,
    ExtrapolationMethod extrapolation)
    : grid_(std::move(grid))
    , cpu_interpolator_(std::make_unique<CPUInterpolator>(*grid_, extrapolation))
    , method_(method)
    , extrapolation_(extrapolation) {
    // Initialize GPU memory for grid data
    const auto& field_data = grid_->getFieldData();
    if (!field_data.empty()) {
        d_grid_data_.allocate(field_data.size());
        d_grid_data_.copyToDevice(field_data.data(), field_data.size());
    }
}

InterpolationResult GPUStructuredInterpolatorAdapter::query(const Point3D& point) const {
    // For single queries, use CPU fallback for simplicity
    return cpu_interpolator_->query(point);
}

std::vector<InterpolationResult> GPUStructuredInterpolatorAdapter::queryBatch(
    const std::vector<Point3D>& points) const {
    if (points.empty()) {
        return {};
    }

    // Allocate GPU memory for query points and results
    p3d::cuda::GpuMemory<Point3D> d_query_points;
    p3d::cuda::GpuMemory<InterpolationResult> d_results;

    d_query_points.allocate(points.size());
    d_results.allocate(points.size());

    // Copy query points to GPU
    d_query_points.copyToDevice(points.data(), points.size());

    // Get grid parameters
    GridParams grid_params = grid_->getParams();

    // Launch kernel
    const int extrapolation_method = static_cast<int>(extrapolation_);
    const size_t num_threads = 256;
    const size_t num_blocks = (points.size() + num_threads - 1) / num_threads;

    p3d::cuda::TricubicHermiteInterpolationKernel<<<num_blocks, num_threads>>>(
        d_query_points.getDevicePtr(),
        d_grid_data_.getDevicePtr(),
        grid_params,
        d_results.getDevicePtr(),
        points.size(),
        extrapolation_method
    );

    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        // Fallback to CPU on error
        return cpu_interpolator_->queryBatch(points);
    }

    // Wait for kernel completion
    cudaDeviceSynchronize();

    // Copy results back to host
    std::vector<InterpolationResult> results(points.size());
    d_results.copyToHost(results.data(), points.size());

    return results;
}

size_t GPUStructuredInterpolatorAdapter::getDataCount() const {
    return grid_->getDataCount();
}

void GPUStructuredInterpolatorAdapter::getBounds(Point3D& min_bound, Point3D& max_bound) const {
    const auto& params = grid_->getParams();
    min_bound = params.min_bound;
    max_bound = params.max_bound;
}

GridParams GPUStructuredInterpolatorAdapter::getGridParams() const {
    return grid_->getParams();
}

std::vector<Point3D> GPUStructuredInterpolatorAdapter::getCoordinates() const {
    return grid_->getCoordinates();
}

std::vector<MagneticFieldData> GPUStructuredInterpolatorAdapter::getFieldData() const {
    return grid_->getFieldData();
}

// GPUUnstructuredInterpolatorAdapter implementation

GPUUnstructuredInterpolatorAdapter::GPUUnstructuredInterpolatorAdapter(
    std::unique_ptr<UnstructuredInterpolator> interpolator,
    InterpolationMethod method,
    ExtrapolationMethod extrapolation)
    : unstructured_interpolator_(std::move(interpolator))
    , method_(method)
    , extrapolation_(extrapolation) {
    // Initialize GPU memory for points and field data
    const auto& coordinates = unstructured_interpolator_->getCoordinates();
    const auto& field_data = unstructured_interpolator_->getFieldData();

    if (!coordinates.empty()) {
        d_points_.allocate(coordinates.size());
        d_points_.copyToDevice(coordinates.data(), coordinates.size());
    }

    if (!field_data.empty()) {
        d_field_data_.allocate(field_data.size());
        d_field_data_.copyToDevice(field_data.data(), field_data.size());
    }

    // Build spatial grid for efficient neighbor finding
    Point3D min_bound = unstructured_interpolator_->getMinBound();
    Point3D max_bound = unstructured_interpolator_->getMaxBound();
    spatial_grid_ = buildSpatialGrid(coordinates, min_bound, max_bound);

    // Upload spatial grid data to GPU
    if (!spatial_grid_.cell_offsets.empty()) {
        d_cell_offsets_.allocate(spatial_grid_.cell_offsets.size());
        d_cell_offsets_.copyToDevice(spatial_grid_.cell_offsets.data(), spatial_grid_.cell_offsets.size());
    }

    if (!spatial_grid_.cell_points.empty()) {
        d_cell_points_.allocate(spatial_grid_.cell_points.size());
        d_cell_points_.copyToDevice(spatial_grid_.cell_points.data(), spatial_grid_.cell_points.size());
    }
}

InterpolationResult GPUUnstructuredInterpolatorAdapter::query(const Point3D& point) const {
    // For single queries, use CPU fallback for simplicity
    return unstructured_interpolator_->query(point);
}

std::vector<InterpolationResult> GPUUnstructuredInterpolatorAdapter::queryBatch(
    const std::vector<Point3D>& points) const {
    if (points.empty()) {
        return {};
    }

    // Allocate GPU memory for query points and results
    p3d::cuda::GpuMemory<Point3D> d_query_points;
    p3d::cuda::GpuMemory<InterpolationResult> d_results;

    d_query_points.allocate(points.size());
    d_results.allocate(points.size());

    // Copy query points to GPU
    d_query_points.copyToDevice(points.data(), points.size());

    // Get interpolation parameters
    Real power = unstructured_interpolator_->getPower();
    int extrapolation_method = static_cast<int>(extrapolation_);
    Point3D min_bound = unstructured_interpolator_->getMinBound();
    Point3D max_bound = unstructured_interpolator_->getMaxBound();
    size_t data_count = unstructured_interpolator_->getDataCount();

    // Launch kernel - use spatial grid kernel if spatial grid is available
    const size_t num_threads = 256;
    const size_t num_blocks = (points.size() + num_threads - 1) / num_threads;

    if (!spatial_grid_.cell_offsets.empty() && !spatial_grid_.cell_points.empty()) {
        // Use spatial grid kernel for better performance
        p3d::cuda::IDWSpatialGridKernel<<<num_blocks, num_threads>>>(
            d_query_points.getDevicePtr(),
            d_points_.getDevicePtr(),
            d_field_data_.getDevicePtr(),
            data_count,
            d_cell_offsets_.getDevicePtr(),
            d_cell_points_.getDevicePtr(),
            spatial_grid_.origin,
            spatial_grid_.cell_size,
            spatial_grid_.dimensions[0],
            spatial_grid_.dimensions[1],
            spatial_grid_.dimensions[2],
            power,
            extrapolation_method,
            min_bound,
            max_bound,
            d_results.getDevicePtr(),
            points.size()
        );
    } else {
        // Fallback to brute force kernel
        p3d::cuda::IDWInterpolationKernel<<<num_blocks, num_threads>>>(
            d_query_points.getDevicePtr(),
            d_points_.getDevicePtr(),
            d_field_data_.getDevicePtr(),
            data_count,
            power,
            extrapolation_method,
            min_bound,
            max_bound,
            d_results.getDevicePtr(),
            points.size()
        );
    }

    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        // Fallback to CPU on error
        return unstructured_interpolator_->queryBatch(points);
    }

    // Wait for kernel completion
    cudaDeviceSynchronize();

    // Copy results back to host
    std::vector<InterpolationResult> results(points.size());
    d_results.copyToHost(results.data(), points.size());

    return results;
}

size_t GPUUnstructuredInterpolatorAdapter::getDataCount() const {
    return unstructured_interpolator_->getDataCount();
}

void GPUUnstructuredInterpolatorAdapter::getBounds(Point3D& min_bound, Point3D& max_bound) const {
    min_bound = unstructured_interpolator_->getMinBound();
    max_bound = unstructured_interpolator_->getMaxBound();
}

GridParams GPUUnstructuredInterpolatorAdapter::getGridParams() const {
    // Unstructured data doesn't have grid parameters, return default
    return GridParams();
}

std::vector<Point3D> GPUUnstructuredInterpolatorAdapter::getCoordinates() const {
    return unstructured_interpolator_->getCoordinates();
}

std::vector<MagneticFieldData> GPUUnstructuredInterpolatorAdapter::getFieldData() const {
    return unstructured_interpolator_->getFieldData();
}

}  // namespace p3d
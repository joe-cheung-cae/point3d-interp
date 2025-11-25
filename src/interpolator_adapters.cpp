#include "point3d_interp/interpolator_adapters.h"
#include <algorithm>

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
    // GPU initialization would go here
}

InterpolationResult GPUStructuredInterpolatorAdapter::query(const Point3D& point) const {
    // For single queries, use CPU fallback for simplicity
    return cpu_interpolator_->query(point);
}

std::vector<InterpolationResult> GPUStructuredInterpolatorAdapter::queryBatch(
    const std::vector<Point3D>& points) const {
    // GPU batch implementation would go here
    // For now, fall back to CPU
    return cpu_interpolator_->queryBatch(points);
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
    // GPU initialization would go here
}

InterpolationResult GPUUnstructuredInterpolatorAdapter::query(const Point3D& point) const {
    // For single queries, use CPU fallback for simplicity
    return unstructured_interpolator_->query(point);
}

std::vector<InterpolationResult> GPUUnstructuredInterpolatorAdapter::queryBatch(
    const std::vector<Point3D>& points) const {
    // GPU batch implementation would go here
    // For now, fall back to CPU
    return unstructured_interpolator_->queryBatch(points);
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
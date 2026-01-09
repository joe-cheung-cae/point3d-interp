#pragma once

#include "interpolator_interface.h"
#include "cpu_interpolator.h"
#include "unstructured_interpolator.h"
#include "hermite_mls_interpolator.h"
#include "grid_structure.h"
#include "spatial_grid.h"
#include <memory>
#include "memory_manager.h"

P3D_NAMESPACE_BEGIN

/**
 * @brief Adapter for CPU structured grid interpolator
 */
class CPUStructuredInterpolatorAdapter : public IInterpolator {
  public:
    CPUStructuredInterpolatorAdapter(std::unique_ptr<RegularGrid3D> grid, InterpolationMethod method,
                                     ExtrapolationMethod extrapolation);

    InterpolationResult              query(const Point3D& point) const override;
    std::vector<InterpolationResult> queryBatch(const std::vector<Point3D>& points) const override;
    bool                             supportsGPU() const override { return false; }
    DataStructureType                getDataType() const override { return DataStructureType::RegularGrid; }
    InterpolationMethod              getMethod() const override { return method_; }
    ExtrapolationMethod              getExtrapolationMethod() const override { return extrapolation_; }
    size_t                           getDataCount() const override;
    void                             getBounds(Point3D& min_bound, Point3D& max_bound) const override;
    GridParams                       getGridParams() const override;
    std::vector<Point3D>             getCoordinates() const override;
    std::vector<MagneticFieldData>   getFieldData() const override;
    bool                             getLastKernelTime(float& kernel_time_ms) const override;

  private:
    std::unique_ptr<RegularGrid3D>   grid_;
    std::unique_ptr<CPUInterpolator> cpu_interpolator_;
    InterpolationMethod              method_;
    ExtrapolationMethod              extrapolation_;
};

/**
 * @brief Adapter for CPU unstructured data interpolator
 */
class CPUUnstructuredInterpolatorAdapter : public IInterpolator {
  public:
    CPUUnstructuredInterpolatorAdapter(std::unique_ptr<UnstructuredInterpolator> interpolator,
                                       InterpolationMethod method, ExtrapolationMethod extrapolation);

    InterpolationResult              query(const Point3D& point) const override;
    std::vector<InterpolationResult> queryBatch(const std::vector<Point3D>& points) const override;
    bool                             supportsGPU() const override { return false; }
    DataStructureType                getDataType() const override { return DataStructureType::Unstructured; }
    InterpolationMethod              getMethod() const override { return method_; }
    ExtrapolationMethod              getExtrapolationMethod() const override { return extrapolation_; }
    size_t                           getDataCount() const override;
    void                             getBounds(Point3D& min_bound, Point3D& max_bound) const override;
    GridParams                       getGridParams() const override;
    std::vector<Point3D>             getCoordinates() const override;
    std::vector<MagneticFieldData>   getFieldData() const override;
    bool                             getLastKernelTime(float& kernel_time_ms) const override;

  private:
    std::unique_ptr<UnstructuredInterpolator> unstructured_interpolator_;
    InterpolationMethod                       method_;
    ExtrapolationMethod                       extrapolation_;
};

/**
 * @brief GPU-capable structured grid interpolator adapter
 */
class GPUStructuredInterpolatorAdapter : public IInterpolator {
  public:
    GPUStructuredInterpolatorAdapter(std::unique_ptr<RegularGrid3D> grid, InterpolationMethod method,
                                     ExtrapolationMethod extrapolation);

    InterpolationResult              query(const Point3D& point) const override;
    std::vector<InterpolationResult> queryBatch(const std::vector<Point3D>& points) const override;
    bool                             supportsGPU() const override { return true; }
    DataStructureType                getDataType() const override { return DataStructureType::RegularGrid; }
    InterpolationMethod              getMethod() const override { return method_; }
    ExtrapolationMethod              getExtrapolationMethod() const override { return extrapolation_; }
    size_t                           getDataCount() const override;
    void                             getBounds(Point3D& min_bound, Point3D& max_bound) const override;
    GridParams                       getGridParams() const override;
    std::vector<Point3D>             getCoordinates() const override;
    std::vector<MagneticFieldData>   getFieldData() const override;
    bool                             getLastKernelTime(float& kernel_time_ms) const override;

  private:
    std::unique_ptr<RegularGrid3D>   grid_;
    std::unique_ptr<CPUInterpolator> cpu_interpolator_;  // Fallback for single queries
    InterpolationMethod              method_;
    ExtrapolationMethod              extrapolation_;

    // GPU resources
    cuda::GpuMemory<MagneticFieldData> d_grid_data_;  // Device memory for grid data

    // Kernel timing
    mutable float last_kernel_time_ms_;
};

/**
 * @brief GPU-capable unstructured data interpolator adapter
 */
class GPUUnstructuredInterpolatorAdapter : public IInterpolator {
  public:
    GPUUnstructuredInterpolatorAdapter(std::unique_ptr<UnstructuredInterpolator> interpolator,
                                       InterpolationMethod method, ExtrapolationMethod extrapolation);

    InterpolationResult              query(const Point3D& point) const override;
    std::vector<InterpolationResult> queryBatch(const std::vector<Point3D>& points) const override;
    bool                             supportsGPU() const override { return true; }
    DataStructureType                getDataType() const override { return DataStructureType::Unstructured; }
    InterpolationMethod              getMethod() const override { return method_; }
    ExtrapolationMethod              getExtrapolationMethod() const override { return extrapolation_; }
    size_t                           getDataCount() const override;
    void                             getBounds(Point3D& min_bound, Point3D& max_bound) const override;
    GridParams                       getGridParams() const override;
    std::vector<Point3D>             getCoordinates() const override;
    std::vector<MagneticFieldData>   getFieldData() const override;
    bool                             getLastKernelTime(float& kernel_time_ms) const override;

  private:
    std::unique_ptr<UnstructuredInterpolator> unstructured_interpolator_;
    InterpolationMethod                       method_;
    ExtrapolationMethod                       extrapolation_;

    // GPU resources
    cuda::GpuMemory<Point3D>           d_points_;        // Device memory for data points
    cuda::GpuMemory<MagneticFieldData> d_field_data_;    // Device memory for field data
    cuda::GpuMemory<uint32_t>          d_cell_offsets_;  // Device memory for spatial grid cell offsets
    cuda::GpuMemory<uint32_t>          d_cell_points_;   // Device memory for spatial grid cell points
    SpatialGrid                        spatial_grid_;    // Spatial grid for efficient neighbor finding

    // Kernel timing
    mutable float last_kernel_time_ms_;
};

/**
 * @brief Adapter for CPU Hermite MLS interpolator
 */
class CPUHermiteMLSInterpolatorAdapter : public IInterpolator {
  public:
    CPUHermiteMLSInterpolatorAdapter(std::unique_ptr<HermiteMLSInterpolator> interpolator,
                                     InterpolationMethod method, ExtrapolationMethod extrapolation);

    InterpolationResult              query(const Point3D& point) const override;
    std::vector<InterpolationResult> queryBatch(const std::vector<Point3D>& points) const override;
    bool                             supportsGPU() const override { return false; }
    DataStructureType                getDataType() const override { return DataStructureType::Unstructured; }
    InterpolationMethod              getMethod() const override { return method_; }
    ExtrapolationMethod              getExtrapolationMethod() const override { return extrapolation_; }
    size_t                           getDataCount() const override;
    void                             getBounds(Point3D& min_bound, Point3D& max_bound) const override;
    GridParams                       getGridParams() const override;
    std::vector<Point3D>             getCoordinates() const override;
    std::vector<MagneticFieldData>   getFieldData() const override;
    bool                             getLastKernelTime(float& kernel_time_ms) const override;

  private:
    std::unique_ptr<HermiteMLSInterpolator> hmls_interpolator_;
    InterpolationMethod                     method_;
    ExtrapolationMethod                     extrapolation_;
};

/**
 * @brief GPU-capable Hermite MLS interpolator adapter
 */
class GPUHermiteMLSInterpolatorAdapter : public IInterpolator {
  public:
    GPUHermiteMLSInterpolatorAdapter(std::unique_ptr<HermiteMLSInterpolator> interpolator,
                                     InterpolationMethod method, ExtrapolationMethod extrapolation);

    InterpolationResult              query(const Point3D& point) const override;
    std::vector<InterpolationResult> queryBatch(const std::vector<Point3D>& points) const override;
    bool                             supportsGPU() const override { return true; }
    DataStructureType                getDataType() const override { return DataStructureType::Unstructured; }
    InterpolationMethod              getMethod() const override { return method_; }
    ExtrapolationMethod              getExtrapolationMethod() const override { return extrapolation_; }
    size_t                           getDataCount() const override;
    void                             getBounds(Point3D& min_bound, Point3D& max_bound) const override;
    GridParams                       getGridParams() const override;
    std::vector<Point3D>             getCoordinates() const override;
    std::vector<MagneticFieldData>   getFieldData() const override;
    bool                             getLastKernelTime(float& kernel_time_ms) const override;

  private:
    std::unique_ptr<HermiteMLSInterpolator> hmls_interpolator_;
    InterpolationMethod                     method_;
    ExtrapolationMethod                     extrapolation_;

    // GPU resources
    cuda::GpuMemory<Point3D>           d_points_;
    cuda::GpuMemory<MagneticFieldData> d_field_data_;
    cuda::GpuMemory<uint32_t>          d_cell_offsets_;
    cuda::GpuMemory<uint32_t>          d_cell_points_;
    SpatialGrid                        spatial_grid_;

    // Kernel timing
    mutable float last_kernel_time_ms_;
};

P3D_NAMESPACE_END

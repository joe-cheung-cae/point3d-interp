#include "point3d_interp/api.h"
#include "point3d_interp/constants.h"
#include "point3d_interp/data_loader.h"
#include "point3d_interp/grid_structure.h"
#include "point3d_interp/cpu_interpolator.h"
#include "point3d_interp/unstructured_interpolator.h"
#include "point3d_interp/spatial_grid.h"
#include <memory>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <cmath>

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #include "point3d_interp/memory_manager.h"
#endif

namespace p3d {

/**
 * @brief Data structure type
 */
enum class DataStructureType { RegularGrid, Unstructured };

/**
 * @brief API implementation class (Pimpl pattern)
 */
class MagneticFieldInterpolator::Impl {
  private:
#ifdef __CUDACC__
    /**
     * @brief Log CUDA error with consistent formatting
     * @param operation Description of the operation that failed
     * @param cuda_err CUDA error code
     */
    static void LogCudaError(const std::string& operation, cudaError_t cuda_err) {
        std::cerr << "CUDA " << operation << " error: " << cudaGetErrorString(cuda_err) << std::endl;
    }
#endif

    /**
     * @brief Log general error with consistent formatting
     * @param operation Description of the operation that failed
     * @param message Error message
     */
    static void LogError(const std::string& operation, const std::string& message) {
        std::cerr << operation << " failed: " << message << std::endl;
    }

    /**
     * @brief Check if a floating point value is finite (not NaN or infinite)
     * @param value Value to check
     * @return true if finite
     */
    static bool IsFinite(Real value) { return std::isfinite(static_cast<double>(value)); }

    /**
     * @brief Validate coordinate values for finiteness
     * @param point Point to validate
     * @return true if all coordinates are finite
     */
    static bool ValidatePoint(const Point3D& point) {
        return IsFinite(point.x) && IsFinite(point.y) && IsFinite(point.z);
    }

    /**
     * @brief Validate magnetic field data for finiteness (allows NaN, rejects Inf)
     * @param field_data Field data to validate
     * @return true if all field values are not infinite
     */
    static bool ValidateFieldData(const MagneticFieldData& field_data) {
        return !std::isinf(field_data.Bx) && !std::isinf(field_data.By) && !std::isinf(field_data.Bz);
    }

    /**
     * @brief Check for potential integer overflow in size calculations
     * @param count Size to validate
     * @return true if safe from overflow
     */
    static bool ValidateSize(size_t count) {
        // Prevent overflow in common calculations by limiting to SIZE_MAX/2
        return count > 0 && count <= SIZE_MAX / 2;
    }

  public:
    Impl(bool use_gpu, int device_id, InterpolationMethod method,
         ExtrapolationMethod extrapolation_method = ExtrapolationMethod::None);
    ~Impl();

    // Move constructor
    Impl(Impl&& other) noexcept;
    // Move assignment
    Impl& operator=(Impl&& other) noexcept;

    ErrorCode LoadFromCSV(const std::string& filepath);
    ErrorCode LoadFromMemory(const Point3D* points, const MagneticFieldData* field_data, size_t count);
    ErrorCode Query(const Point3D& query_point, InterpolationResult& result);
    ErrorCode QueryBatch(const Point3D* query_points, InterpolationResult* results, size_t count);
    static ErrorCode ExportInputPoints(const std::vector<Point3D>& coordinates,
                                       const std::vector<MagneticFieldData>& field_data,
                                       ExportFormat format, const std::string& filename);
    static ErrorCode ExportOutputPoints(ExportFormat format, const std::vector<Point3D>& query_points,
                                        const std::vector<InterpolationResult>& results, const std::string& filename);

    ErrorCode GetLastKernelTime(float& kernel_time_ms) const {
#ifdef __CUDACC__
        if (use_gpu_ && gpu_initialized_) {
            kernel_time_ms = last_kernel_time_ms_;
            return ErrorCode::Success;
        } else {
            return ErrorCode::CudaNotAvailable;
        }
#else
        return ErrorCode::CudaNotAvailable;
#endif
    }

    const GridParams& GetGridParams() const {
        return (data_type_ == DataStructureType::RegularGrid && grid_) ? grid_->getParams() : default_params_;
    }
    bool   IsDataLoaded() const { return data_type_ != DataStructureType::RegularGrid || grid_ != nullptr; }
    size_t GetDataPointCount() const {
        if (data_type_ == DataStructureType::RegularGrid && grid_) {
            return grid_->getDataCount();
        } else if (data_type_ == DataStructureType::Unstructured && unstructured_interpolator_) {
            return unstructured_interpolator_->getDataCount();
        }
        return 0;
    }

    std::vector<Point3D> GetCoordinates() const {
        if (data_type_ == DataStructureType::RegularGrid && grid_) {
            return grid_->getCoordinates();
        } else if (data_type_ == DataStructureType::Unstructured && unstructured_interpolator_) {
            return unstructured_interpolator_->getCoordinates();
        }
        return {};
    }

    std::vector<MagneticFieldData> GetFieldData() const {
        if (data_type_ == DataStructureType::RegularGrid && grid_) {
            return grid_->getFieldData();
        } else if (data_type_ == DataStructureType::Unstructured && unstructured_interpolator_) {
            return unstructured_interpolator_->getFieldData();
        }
        return {};
    }

    const Point3D* GetDeviceGridPoints() const {
#ifdef __CUDACC__
        return gpu_grid_points_ ? gpu_grid_points_->getDevicePtr() : nullptr;
#else
        return nullptr;
#endif
    }

    const MagneticFieldData* GetDeviceFieldData() const {
#ifdef __CUDACC__
        return gpu_grid_field_data_ ? gpu_grid_field_data_->getDevicePtr() : nullptr;
#else
        return nullptr;
#endif
    }

    const GridParams* GetDeviceGridParams() const {
        // Note: GridParams is stored on host, need to upload to GPU if needed
        // For now, return nullptr as we don't maintain a GPU copy of GridParams
        return nullptr;
    }

    ErrorCode LaunchInterpolationKernel(const Point3D* d_query_points, InterpolationResult* d_results, size_t count,
                                        void* stream) {
        if (data_type_ != DataStructureType::RegularGrid || grid_ == nullptr) {
            return ErrorCode::DataNotLoaded;
        }

        if (!d_query_points || !d_results || count == 0) {
            return ErrorCode::InvalidParameter;
        }

#ifdef __CUDACC__
        if (use_gpu_ && gpu_initialized_) {
            try {
                // Get optimal kernel configuration
                KernelConfig config;
                GetOptimalKernelConfig(count, config);
                dim3 block_dim(config.block_x, config.block_y, config.block_z);
                dim3 grid_dim(config.grid_x, config.grid_y, config.grid_z);

                // Get grid parameters (host copy)
                GridParams grid_params = grid_->getParams();

                // Calculate shared memory size for kernel
                const size_t shared_mem_size = sizeof(GridParams) + 8 * sizeof(MagneticFieldData);

                // Launch kernel with specified stream
                cuda::TricubicHermiteInterpolationKernel<<<grid_dim, block_dim, shared_mem_size, (cudaStream_t)stream>>>(
                    d_query_points, gpu_grid_field_data_->getDevicePtr(), grid_params, d_results, count,
                    static_cast<int>(extrapolation_method_));

                // Check for kernel launch errors
                cudaError_t cuda_err = cudaGetLastError();
                if (cuda_err != cudaSuccess) {
                    LogCudaError("kernel launch", cuda_err);
                    return ErrorCode::CudaError;
                }

                return ErrorCode::Success;
            } catch (const std::exception& e) {
                LogError("GPU kernel launch", e.what());
                return ErrorCode::CudaError;
            }
        } else {
            return ErrorCode::CudaNotAvailable;
        }
#else
        return ErrorCode::CudaNotAvailable;
#endif
    }

    void GetOptimalKernelConfig(size_t query_count, KernelConfig& config) const {
        // Use same configuration as internal batch query
        const int BLOCK_SIZE = CUDA_DEFAULT_BLOCK_SIZE;
        const int MIN_BLOCKS = CUDA_MIN_BLOCKS;

        int num_blocks       = (query_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        num_blocks           = std::max(num_blocks, MIN_BLOCKS);
        const int MAX_BLOCKS = CUDA_MAX_BLOCKS;
        num_blocks           = std::min(num_blocks, MAX_BLOCKS);

        config.block_x = BLOCK_SIZE;
        config.block_y = 1;
        config.block_z = 1;
        config.grid_x  = num_blocks;
        config.grid_y  = 1;
        config.grid_z  = 1;
    }

  private:
    bool InitializeGPU(int device_id);
    void ReleaseGPU();
    bool UploadDataToGPU();

    // Data structure type
    DataStructureType data_type_;

    // CPU implementation
    std::unique_ptr<RegularGrid3D>            grid_;
    std::unique_ptr<CPUInterpolator>          cpu_interpolator_;
    std::unique_ptr<UnstructuredInterpolator> unstructured_interpolator_;

    // GPU implementation
    bool                use_gpu_;
    int                 device_id_;
    bool                gpu_initialized_;
    InterpolationMethod method_;
    ExtrapolationMethod extrapolation_method_;

    // Stored spatial grid parameters for GPU queries
    Point3D                 spatial_grid_origin_;
    Point3D                 spatial_grid_cell_size_;
    std::array<uint32_t, 3> spatial_grid_dimensions_;

#ifdef __CUDACC__
    // GPU memory manager for regular grids
    std::unique_ptr<cuda::GpuMemory<Point3D>>           gpu_grid_points_;
    std::unique_ptr<cuda::GpuMemory<MagneticFieldData>> gpu_grid_field_data_;

    // GPU memory manager for unstructured data
    std::unique_ptr<cuda::GpuMemory<Point3D>>           gpu_unstructured_points_;
    std::unique_ptr<cuda::GpuMemory<MagneticFieldData>> gpu_unstructured_field_data_;

    // GPU memory for spatial grid (unstructured data)
    std::unique_ptr<cuda::GpuMemory<uint32_t>> gpu_cell_offsets_;
    std::unique_ptr<cuda::GpuMemory<uint32_t>> gpu_cell_points_;
    SpatialGrid                                spatial_grid_;  // Host copy for parameters

    // Shared GPU memory for queries and results (pre-allocated for performance)
    std::unique_ptr<cuda::GpuMemory<Point3D>>             gpu_query_points_;
    std::unique_ptr<cuda::GpuMemory<InterpolationResult>> gpu_results_;
    size_t                                                gpu_memory_capacity_;  // Track allocated capacity

    // Kernel timing
    float last_kernel_time_ms_;
#endif

    // Default parameters
    GridParams default_params_;
};

ErrorCode MagneticFieldInterpolator::Impl::ExportInputPoints(const std::vector<Point3D>& coordinates,
                                                             const std::vector<MagneticFieldData>& field_data,
                                                             ExportFormat format, const std::string& filename) {
    if (coordinates.size() != field_data.size()) {
        return ErrorCode::InvalidParameter;
    }

    if (coordinates.empty()) {
        return ErrorCode::InvalidParameter;
    }

    auto exporter = CreateExporter(format);
    if (!exporter) {
        return ErrorCode::InvalidParameter;
    }

    try {
        if (!exporter->ExportInputPoints(coordinates, field_data, filename)) {
            return ErrorCode::InvalidParameter;
        }

        return ErrorCode::Success;
    } catch (const std::exception& e) {
        LogError("Export input points", e.what());
        return ErrorCode::InvalidParameter;
    }
}

ErrorCode MagneticFieldInterpolator::Impl::ExportOutputPoints(ExportFormat format, const std::vector<Point3D>& query_points,
                                                              const std::vector<InterpolationResult>& results,
                                                              const std::string& filename) {
    if (query_points.size() != results.size()) {
        return ErrorCode::InvalidParameter;
    }

    if (query_points.empty()) {
        return ErrorCode::InvalidParameter;
    }

    auto exporter = CreateExporter(format);
    if (!exporter) {
        return ErrorCode::InvalidParameter;
    }

    try {
        if (!exporter->ExportOutputPoints(query_points, results, filename)) {
            return ErrorCode::InvalidParameter;
        }

        return ErrorCode::Success;
    } catch (const std::exception& e) {
        LogError("Export output points", e.what());
        return ErrorCode::InvalidParameter;
    }
}

// Implementation

MagneticFieldInterpolator::Impl::Impl(bool use_gpu, int device_id, InterpolationMethod method,
                                      ExtrapolationMethod extrapolation_method)
    : data_type_(DataStructureType::RegularGrid),
      use_gpu_(use_gpu),
      device_id_(device_id),
      gpu_initialized_(false),
      method_(method),
      extrapolation_method_(extrapolation_method),
      spatial_grid_origin_(0, 0, 0),
      spatial_grid_cell_size_(1, 1, 1),
      spatial_grid_dimensions_{0, 0, 0}
#ifdef __CUDACC__
      ,
      gpu_memory_capacity_(0),
      last_kernel_time_ms_(0.0f)
#endif
{
#ifdef __CUDACC__
    if (use_gpu_) {
        gpu_initialized_ = InitializeGPU(device_id);
        if (!gpu_initialized_) {
            LogError("GPU initialization", "falling back to CPU");
            use_gpu_ = false;
        }
    }
#endif
}

MagneticFieldInterpolator::Impl::Impl(Impl&& other) noexcept
    : data_type_(other.data_type_),
      grid_(std::move(other.grid_)),
      cpu_interpolator_(std::move(other.cpu_interpolator_)),
      unstructured_interpolator_(std::move(other.unstructured_interpolator_)),
      use_gpu_(other.use_gpu_),
      device_id_(other.device_id_),
      gpu_initialized_(other.gpu_initialized_),
      method_(other.method_),
      extrapolation_method_(other.extrapolation_method_),
      spatial_grid_origin_(other.spatial_grid_origin_),
      spatial_grid_cell_size_(other.spatial_grid_cell_size_),
      spatial_grid_dimensions_(other.spatial_grid_dimensions_)
#ifdef __CUDACC__
      ,
      gpu_memory_capacity_(other.gpu_memory_capacity_)
#endif
{
#ifdef __CUDACC__
    gpu_grid_points_             = std::move(other.gpu_grid_points_);
    gpu_grid_field_data_         = std::move(other.gpu_grid_field_data_);
    gpu_unstructured_points_     = std::move(other.gpu_unstructured_points_);
    gpu_unstructured_field_data_ = std::move(other.gpu_unstructured_field_data_);
    gpu_cell_offsets_            = std::move(other.gpu_cell_offsets_);
    gpu_cell_points_             = std::move(other.gpu_cell_points_);
    gpu_query_points_            = std::move(other.gpu_query_points_);
    gpu_results_                 = std::move(other.gpu_results_);
#endif
    // Reset moved-from state
    other.gpu_initialized_ = false;
}

MagneticFieldInterpolator::Impl& MagneticFieldInterpolator::Impl::operator=(Impl&& other) noexcept {
    if (this != &other) {
        data_type_                 = other.data_type_;
        grid_                      = std::move(other.grid_);
        cpu_interpolator_          = std::move(other.cpu_interpolator_);
        unstructured_interpolator_ = std::move(other.unstructured_interpolator_);
        use_gpu_                   = other.use_gpu_;
        device_id_                 = other.device_id_;
        gpu_initialized_           = other.gpu_initialized_;
        method_                    = other.method_;
        extrapolation_method_      = other.extrapolation_method_;
        spatial_grid_origin_       = other.spatial_grid_origin_;
        spatial_grid_cell_size_    = other.spatial_grid_cell_size_;
        spatial_grid_dimensions_   = other.spatial_grid_dimensions_;
#ifdef __CUDACC__
        gpu_memory_capacity_ = other.gpu_memory_capacity_;
#endif
#ifdef __CUDACC__
        gpu_grid_points_             = std::move(other.gpu_grid_points_);
        gpu_grid_field_data_         = std::move(other.gpu_grid_field_data_);
        gpu_unstructured_points_     = std::move(other.gpu_unstructured_points_);
        gpu_unstructured_field_data_ = std::move(other.gpu_unstructured_field_data_);
        gpu_cell_offsets_            = std::move(other.gpu_cell_offsets_);
        gpu_cell_points_             = std::move(other.gpu_cell_points_);
        gpu_query_points_            = std::move(other.gpu_query_points_);
        gpu_results_                 = std::move(other.gpu_results_);
#endif
        // Reset moved-from state
        other.gpu_initialized_ = false;
    }
    return *this;
}

MagneticFieldInterpolator::Impl::~Impl() { ReleaseGPU(); }

ErrorCode MagneticFieldInterpolator::Impl::LoadFromCSV(const std::string& filepath) {
    DataLoader loader;

    std::vector<Point3D>           coordinates;
    std::vector<MagneticFieldData> field_data;
    GridParams                     grid_params;

    ErrorCode err = loader.LoadFromCSV(filepath, coordinates, field_data, grid_params);
    if (err != ErrorCode::Success) {
        return err;
    }

    return LoadFromMemory(coordinates.data(), field_data.data(), coordinates.size());
}

ErrorCode MagneticFieldInterpolator::Impl::LoadFromMemory(const Point3D* points, const MagneticFieldData* field_data,
                                                          size_t count) {
    // Basic null pointer and size validation
    if (!points || !field_data) {
        return ErrorCode::InvalidParameter;
    }

    if (!ValidateSize(count)) {
        return ErrorCode::InvalidParameter;
    }

    // Validate coordinate and field data values
    for (size_t i = 0; i < count; ++i) {
        if (!ValidatePoint(points[i])) {
            return ErrorCode::InvalidParameter;
        }
        if (!ValidateFieldData(field_data[i])) {
            return ErrorCode::InvalidParameter;
        }
    }

    try {
        // Try to create regular grid first
        std::vector<Point3D>           coordinates(points, points + count);
        std::vector<MagneticFieldData> field_values(field_data, field_data + count);

        try {
            grid_             = std::make_unique<RegularGrid3D>(coordinates, field_values);
            cpu_interpolator_ = std::make_unique<CPUInterpolator>(*grid_, extrapolation_method_);
            data_type_        = DataStructureType::RegularGrid;

            // Upload data to GPU
            if (use_gpu_ && !UploadDataToGPU()) {
                // Failed to upload to GPU, fall back to CPU
                use_gpu_ = false;
            }
        } catch (const std::invalid_argument&) {
            // Not a regular grid, use unstructured interpolator
            unstructured_interpolator_ = std::make_unique<UnstructuredInterpolator>(
                coordinates, field_values, DEFAULT_IDW_POWER, DEFAULT_MAX_NEIGHBORS, extrapolation_method_);
            data_type_ = DataStructureType::Unstructured;

            // Upload data to GPU for unstructured data
            if (use_gpu_ && !UploadDataToGPU()) {
                // Failed to upload to GPU, fall back to CPU
                use_gpu_ = false;
            }
        }

        return ErrorCode::Success;
    } catch (const std::exception& e) {
        LogError("Data loading", e.what());
        return ErrorCode::InvalidGridData;
    }
}

ErrorCode MagneticFieldInterpolator::Impl::Query(const Point3D& query_point, InterpolationResult& result) {
    // Validate query point coordinates
    if (!ValidatePoint(query_point)) {
        return ErrorCode::InvalidParameter;
    }

    return QueryBatch(&query_point, &result, 1);
}

ErrorCode MagneticFieldInterpolator::Impl::QueryBatch(const Point3D* query_points, InterpolationResult* results,
                                                      size_t count) {
    if (!IsDataLoaded()) {
        return ErrorCode::DataNotLoaded;
    }

    if (!query_points || !results) {
        return ErrorCode::InvalidParameter;
    }

    if (!ValidateSize(count)) {
        return ErrorCode::InvalidParameter;
    }

    // Validate all query points
    for (size_t i = 0; i < count; ++i) {
        if (!ValidatePoint(query_points[i])) {
            return ErrorCode::InvalidParameter;
        }
    }

    if (data_type_ == DataStructureType::Unstructured) {
        if (use_gpu_ && gpu_initialized_ && gpu_unstructured_points_ && gpu_unstructured_field_data_ &&
            gpu_cell_offsets_ && gpu_cell_points_) {
            // GPU implementation for unstructured data with spatial grid
            try {
                // Print information
                std::cout << "Using GPU implementation for unstructured data with spatial grid" << std::endl;

                // Ensure GPU memory is sufficient with capacity tracking and growth strategy
                size_t required_capacity = std::max(count, gpu_memory_capacity_);
                if (gpu_memory_capacity_ == 0) {
                    // Initial allocation - allocate with some headroom for future growth
                    required_capacity = std::max(count, static_cast<size_t>(1024));
                } else if (count > gpu_memory_capacity_) {
                    // Grow capacity using exponential growth strategy
                    required_capacity = std::max(count, gpu_memory_capacity_ * 2);
                }

                if (!gpu_query_points_ || gpu_query_points_->getCount() < required_capacity) {
                    gpu_query_points_ = std::make_unique<cuda::GpuMemory<Point3D>>();
                    if (!gpu_query_points_->allocate(required_capacity)) {
                        return ErrorCode::CudaError;
                    }
                    gpu_memory_capacity_ = required_capacity;
                }

                if (!gpu_results_ || gpu_results_->getCount() < required_capacity) {
                    gpu_results_ = std::make_unique<cuda::GpuMemory<InterpolationResult>>();
                    if (!gpu_results_->allocate(required_capacity)) {
                        return ErrorCode::CudaError;
                    }
                }

                // Upload query points to GPU
                if (!gpu_query_points_->copyToDevice(query_points, count)) {
                    return ErrorCode::CudaError;
                }

                // Launch optimized IDW kernel with spatial grid
                const int BLOCK_SIZE = CUDA_BLOCK_SIZE_256;
                const int num_blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

                // Get bounding box from unstructured interpolator
                Point3D min_bound = unstructured_interpolator_->getMinBound();
                Point3D max_bound = unstructured_interpolator_->getMaxBound();

                // Get stored spatial grid parameters
                Point3D  grid_origin        = spatial_grid_origin_;
                Point3D  grid_cell_size     = spatial_grid_cell_size_;
                uint32_t grid_dimensions[3] = {spatial_grid_dimensions_[0], spatial_grid_dimensions_[1],
                                               spatial_grid_dimensions_[2]};

                cuda::IDWSpatialGridKernel<<<num_blocks, BLOCK_SIZE>>>(
                    gpu_query_points_->getDevicePtr(), gpu_unstructured_points_->getDevicePtr(),
                    gpu_unstructured_field_data_->getDevicePtr(), unstructured_interpolator_->getDataCount(),
                    gpu_cell_offsets_->getDevicePtr(), gpu_cell_points_->getDevicePtr(), grid_origin, grid_cell_size,
                    grid_dimensions, unstructured_interpolator_->getPower(), static_cast<int>(extrapolation_method_),
                    min_bound, max_bound, gpu_results_->getDevicePtr(), count);

                // Check CUDA errors
                cudaError_t cuda_err = cudaGetLastError();
                if (cuda_err != cudaSuccess) {
                    LogCudaError("IDW spatial grid kernel", cuda_err);
                    return ErrorCode::CudaError;
                }

                // Download results
                if (!gpu_results_->copyToHost(results, count)) {
                    return ErrorCode::CudaError;
                }

                return ErrorCode::Success;
            } catch (const std::exception& e) {
                LogError("GPU IDW spatial grid query", std::string(e.what()) + ", falling back to CPU");
                use_gpu_ = false;
            }
        } else if (use_gpu_ && gpu_initialized_ && gpu_unstructured_points_ && gpu_unstructured_field_data_) {
            // Fallback to brute force GPU implementation
            try {
                // Print information
                std::cout << "Using GPU implementation for unstructured data" << std::endl;

                // Ensure GPU memory is sufficient with capacity tracking and growth strategy
                size_t required_capacity = std::max(count, gpu_memory_capacity_);
                if (gpu_memory_capacity_ == 0) {
                    // Initial allocation - allocate with some headroom for future growth
                    required_capacity = std::max(count, static_cast<size_t>(1024));
                } else if (count > gpu_memory_capacity_) {
                    // Grow capacity using exponential growth strategy
                    required_capacity = std::max(count, gpu_memory_capacity_ * 2);
                }

                if (!gpu_query_points_ || gpu_query_points_->getCount() < required_capacity) {
                    gpu_query_points_ = std::make_unique<cuda::GpuMemory<Point3D>>();
                    if (!gpu_query_points_->allocate(required_capacity)) {
                        return ErrorCode::CudaError;
                    }
                    gpu_memory_capacity_ = required_capacity;
                }

                if (!gpu_results_ || gpu_results_->getCount() < required_capacity) {
                    gpu_results_ = std::make_unique<cuda::GpuMemory<InterpolationResult>>();
                    if (!gpu_results_->allocate(required_capacity)) {
                        return ErrorCode::CudaError;
                    }
                }

                // Upload query points to GPU
                if (!gpu_query_points_->copyToDevice(query_points, count)) {
                    return ErrorCode::CudaError;
                }

                // Launch IDW kernel
                const int BLOCK_SIZE = CUDA_BLOCK_SIZE_256;
                const int num_blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

                // Get bounding box from unstructured interpolator
                Point3D min_bound = unstructured_interpolator_->getMinBound();
                Point3D max_bound = unstructured_interpolator_->getMaxBound();

                // Calculate shared memory size for optimization
                const size_t data_count = unstructured_interpolator_->getDataCount();
                const size_t shared_data_count =
                    std::min(data_count, static_cast<size_t>(BLOCK_SIZE * SHARED_MEMORY_LIMIT_FACTOR));
                const size_t shared_mem_size = shared_data_count * (sizeof(Point3D) + sizeof(MagneticFieldData));

                cuda::IDWInterpolationKernel<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(
                    gpu_query_points_->getDevicePtr(), gpu_unstructured_points_->getDevicePtr(),
                    gpu_unstructured_field_data_->getDevicePtr(), data_count, unstructured_interpolator_->getPower(),
                    static_cast<int>(extrapolation_method_), min_bound, max_bound, gpu_results_->getDevicePtr(), count);

                // Check CUDA errors
                cudaError_t cuda_err = cudaGetLastError();
                if (cuda_err != cudaSuccess) {
                    LogCudaError("IDW kernel", cuda_err);
                    return ErrorCode::CudaError;
                }

                // Download results
                if (!gpu_results_->copyToHost(results, count)) {
                    return ErrorCode::CudaError;
                }

                return ErrorCode::Success;
            } catch (const std::exception& e) {
                LogError("GPU IDW query", std::string(e.what()) + ", falling back to CPU");
                use_gpu_ = false;
            }
        }

        // Print information
        std::cout << "Using CPU implementation for unstructured data" << std::endl;

        // CPU fallback for unstructured data
        std::vector<Point3D> query_vec(query_points, query_points + count);
        auto                 cpu_results = unstructured_interpolator_->queryBatch(query_vec);
        std::copy(cpu_results.begin(), cpu_results.end(), results);
        return ErrorCode::Success;
    }

    if (use_gpu_ && gpu_initialized_ && data_type_ == DataStructureType::RegularGrid) {
        // GPU implementation
        try {
            // Print information
            std::cout << "Using GPU implementation for regular grid data" << std::endl;
          
            // Ensure GPU memory is sufficient with capacity tracking and growth strategy
            size_t required_capacity = std::max(count, gpu_memory_capacity_);
            if (gpu_memory_capacity_ == 0) {
                // Initial allocation - allocate with some headroom for future growth
                required_capacity = std::max(count, static_cast<size_t>(1024));
            } else if (count > gpu_memory_capacity_) {
                // Grow capacity using exponential growth strategy
                required_capacity = std::max(count, gpu_memory_capacity_ * 2);
            }

            if (!gpu_query_points_ || gpu_query_points_->getCount() < required_capacity) {
                gpu_query_points_ = std::make_unique<cuda::GpuMemory<Point3D>>();
                if (!gpu_query_points_->allocate(required_capacity)) {
                    return ErrorCode::CudaError;
                }
                gpu_memory_capacity_ = required_capacity;
            }

            if (!gpu_results_ || gpu_results_->getCount() < required_capacity) {
                gpu_results_ = std::make_unique<cuda::GpuMemory<InterpolationResult>>();
                if (!gpu_results_->allocate(required_capacity)) {
                    return ErrorCode::CudaError;
                }
            }

            // Upload query points to GPU
            if (!gpu_query_points_->copyToDevice(query_points, count)) {
                return ErrorCode::CudaError;
            }

            // Launch CUDA kernel - optimized configuration
            // Use larger thread blocks for better occupancy
            const int BLOCK_SIZE = CUDA_DEFAULT_BLOCK_SIZE;  // Increased to 512 for better performance
            const int MIN_BLOCKS = CUDA_MIN_BLOCKS;          // Minimum 4 blocks to hide latency

            // Calculate required number of blocks, ensure at least minimum blocks
            int num_blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
            num_blocks     = std::max(num_blocks, MIN_BLOCKS);

            // Limit maximum blocks to avoid excessive resource usage
            const int MAX_BLOCKS = CUDA_MAX_BLOCKS;
            num_blocks           = std::min(num_blocks, MAX_BLOCKS);

            // Configure kernel
            dim3 block_dim(BLOCK_SIZE);
            dim3 grid_dim(num_blocks);

            // Calculate shared memory size for kernel
            const size_t shared_mem_size = sizeof(GridParams) + 8 * sizeof(MagneticFieldData);

            // Create CUDA events for kernel timing
            cudaEvent_t start_event, stop_event;
            cudaEventCreate(&start_event);
            cudaEventCreate(&stop_event);

            // Record start event
            cudaEventRecord(start_event);

            cuda::TricubicHermiteInterpolationKernel<<<grid_dim, block_dim, shared_mem_size>>>(
                gpu_query_points_->getDevicePtr(), gpu_grid_field_data_->getDevicePtr(), grid_->getParams(),
                gpu_results_->getDevicePtr(), count, static_cast<int>(extrapolation_method_));

            // Record stop event
            cudaEventRecord(stop_event);
            cudaEventSynchronize(stop_event);

            // Calculate kernel execution time
            cudaEventElapsedTime(&last_kernel_time_ms_, start_event, stop_event);

            // Cleanup events
            cudaEventDestroy(start_event);
            cudaEventDestroy(stop_event);

            // Check CUDA errors
            cudaError_t cuda_err = cudaGetLastError();
            if (cuda_err != cudaSuccess) {
                LogCudaError("kernel", cuda_err);
                return ErrorCode::CudaError;
            }

            // Download results
            if (!gpu_results_->copyToHost(results, count)) {
                return ErrorCode::CudaError;
            }

            return ErrorCode::Success;
        } catch (const std::exception& e) {
            LogError("GPU query", std::string(e.what()) + ", falling back to CPU");
            use_gpu_ = false;
        }
    }

    // Print information
    std::cout << "Using CPU implementation for regular grid data(Fallback method)" << std::endl;

    // CPU implementation (fallback for regular grid)
    std::vector<Point3D> query_vec(query_points, query_points + count);
    auto                 cpu_results = cpu_interpolator_->queryBatch(query_vec);

    std::copy(cpu_results.begin(), cpu_results.end(), results);

    return ErrorCode::Success;
}

bool MagneticFieldInterpolator::Impl::InitializeGPU(int device_id) {
#ifdef __CUDACC__
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        LogCudaError("set device " + std::to_string(device_id), err);
        return false;
    }

    // Check device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        LogCudaError("get device properties", err);
        return false;
    }

    std::cout << "Using GPU: " << prop.name << std::endl;
    return true;
#else
    return false;
#endif
}

void MagneticFieldInterpolator::Impl::ReleaseGPU() {
#ifdef __CUDACC__
    gpu_grid_points_.reset();
    gpu_grid_field_data_.reset();
    gpu_unstructured_points_.reset();
    gpu_unstructured_field_data_.reset();
    gpu_cell_offsets_.reset();
    gpu_cell_points_.reset();
    gpu_query_points_.reset();
    gpu_results_.reset();

    gpu_initialized_ = false;
#endif
}

bool MagneticFieldInterpolator::Impl::UploadDataToGPU() {
#ifdef __CUDACC__
    if (!gpu_initialized_) {
        return false;
    }

    try {
        if (data_type_ == DataStructureType::RegularGrid && grid_) {
            size_t data_count = grid_->getDataCount();

            // Allocate GPU memory for regular grid
            gpu_grid_points_     = std::make_unique<cuda::GpuMemory<Point3D>>();
            gpu_grid_field_data_ = std::make_unique<cuda::GpuMemory<MagneticFieldData>>();

            if (!gpu_grid_points_->allocate(data_count) || !gpu_grid_field_data_->allocate(data_count)) {
                return false;
            }

            // Upload data
            const auto& coordinates = grid_->getCoordinates();
            const auto& field_data  = grid_->getFieldData();

            if (!gpu_grid_points_->copyToDevice(coordinates.data(), data_count) ||
                !gpu_grid_field_data_->copyToDevice(field_data.data(), data_count)) {
                return false;
            }
        } else if (data_type_ == DataStructureType::Unstructured && unstructured_interpolator_) {
            size_t data_count = unstructured_interpolator_->getDataCount();

            // Allocate GPU memory for unstructured data
            gpu_unstructured_points_     = std::make_unique<cuda::GpuMemory<Point3D>>();
            gpu_unstructured_field_data_ = std::make_unique<cuda::GpuMemory<MagneticFieldData>>();

            if (!gpu_unstructured_points_->allocate(data_count) ||
                !gpu_unstructured_field_data_->allocate(data_count)) {
                return false;
            }

            // Upload data
            const auto& coordinates = unstructured_interpolator_->getCoordinates();
            const auto& field_data  = unstructured_interpolator_->getFieldData();

            if (!gpu_unstructured_points_->copyToDevice(coordinates.data(), data_count) ||
                !gpu_unstructured_field_data_->copyToDevice(field_data.data(), data_count)) {
                return false;
            }

            // Build and upload spatial grid for efficient neighbor finding
            Point3D     min_bound    = unstructured_interpolator_->getMinBound();
            Point3D     max_bound    = unstructured_interpolator_->getMaxBound();
            SpatialGrid spatial_grid = buildSpatialGrid(coordinates, min_bound, max_bound);

            // Store spatial grid parameters for GPU queries
            spatial_grid_origin_     = spatial_grid.origin;
            spatial_grid_cell_size_  = spatial_grid.cell_size;
            spatial_grid_dimensions_ = spatial_grid.dimensions;

            // Allocate GPU memory for spatial grid
            gpu_cell_offsets_ = std::make_unique<cuda::GpuMemory<uint32_t>>();
            gpu_cell_points_  = std::make_unique<cuda::GpuMemory<uint32_t>>();

            size_t num_cells_plus_one = spatial_grid.cell_offsets.size();
            size_t num_cell_points    = spatial_grid.cell_points.size();

            if (!gpu_cell_offsets_->allocate(num_cells_plus_one) || !gpu_cell_points_->allocate(num_cell_points)) {
                return false;
            }

            // Upload spatial grid data
            if (!gpu_cell_offsets_->copyToDevice(spatial_grid.cell_offsets.data(), num_cells_plus_one) ||
                !gpu_cell_points_->copyToDevice(spatial_grid.cell_points.data(), num_cell_points)) {
                return false;
            }
        }

        return true;
    } catch (const std::exception& e) {
        LogError("GPU data upload", e.what());
        return false;
    }
#else
    return false;
#endif
}

// MagneticFieldInterpolator implementation

MagneticFieldInterpolator::MagneticFieldInterpolator(bool use_gpu, int device_id, InterpolationMethod method,
                                                     ExtrapolationMethod extrapolation_method)
    : impl_(std::make_unique<Impl>(use_gpu, device_id, method, extrapolation_method)) {}

MagneticFieldInterpolator::~MagneticFieldInterpolator() = default;

MagneticFieldInterpolator::MagneticFieldInterpolator(MagneticFieldInterpolator&& other) noexcept
    : impl_(std::move(other.impl_)) {}

MagneticFieldInterpolator& MagneticFieldInterpolator::operator=(MagneticFieldInterpolator&& other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
    }
    return *this;
}

ErrorCode MagneticFieldInterpolator::LoadFromCSV(const std::string& filepath) {
    if (!impl_) {
        impl_ = std::make_unique<Impl>(false, 0, InterpolationMethod::TricubicHermite, ExtrapolationMethod::None);
    }
    return impl_->LoadFromCSV(filepath);
}

ErrorCode MagneticFieldInterpolator::LoadFromMemory(const Point3D* points, const MagneticFieldData* field_data,
                                                    size_t count) {
    if (!impl_) {
        impl_ = std::make_unique<Impl>(false, 0, InterpolationMethod::TricubicHermite, ExtrapolationMethod::None);
    }
    return impl_->LoadFromMemory(points, field_data, count);
}

ErrorCode MagneticFieldInterpolator::Query(const Point3D& query_point, InterpolationResult& result) {
    return impl_->Query(query_point, result);
}

ErrorCode MagneticFieldInterpolator::QueryBatch(const Point3D* query_points, InterpolationResult* results,
                                                size_t count) {
    return impl_->QueryBatch(query_points, results, count);
}

ErrorCode MagneticFieldInterpolator::QueryBatch(const std::vector<Point3D>&       query_points,
                                                std::vector<InterpolationResult>& results) {
    results.resize(query_points.size());
    return impl_->QueryBatch(query_points.data(), results.data(), query_points.size());
}

InterpolationResult MagneticFieldInterpolator::QueryEx(const Point3D& query_point) {
    InterpolationResult result;
    ErrorCode           err = Query(query_point, result);
    if (err != ErrorCode::Success) {
        throw std::runtime_error("Interpolation query failed: " + std::string(ErrorCodeToString(err)));
    }
    return result;
}

std::vector<InterpolationResult> MagneticFieldInterpolator::QueryBatchEx(const std::vector<Point3D>& query_points) {
    std::vector<InterpolationResult> results;
    ErrorCode                        err = QueryBatch(query_points, results);
    if (err != ErrorCode::Success) {
        throw std::runtime_error("Batch interpolation query failed: " + std::string(ErrorCodeToString(err)));
    }
    return results;
}

const GridParams& MagneticFieldInterpolator::GetGridParams() const {
    static GridParams default_params;
    return impl_ ? impl_->GetGridParams() : default_params;
}

bool MagneticFieldInterpolator::IsDataLoaded() const { return impl_ && impl_->IsDataLoaded(); }

size_t MagneticFieldInterpolator::GetDataPointCount() const { return impl_ ? impl_->GetDataPointCount() : 0; }

std::vector<Point3D> MagneticFieldInterpolator::GetCoordinates() const {
    return impl_ ? impl_->GetCoordinates() : std::vector<Point3D>{};
}

std::vector<MagneticFieldData> MagneticFieldInterpolator::GetFieldData() const {
    return impl_ ? impl_->GetFieldData() : std::vector<MagneticFieldData>{};
}

const Point3D* MagneticFieldInterpolator::GetDeviceGridPoints() const {
    return impl_ ? impl_->GetDeviceGridPoints() : nullptr;
}

const MagneticFieldData* MagneticFieldInterpolator::GetDeviceFieldData() const {
    return impl_ ? impl_->GetDeviceFieldData() : nullptr;
}

const GridParams* MagneticFieldInterpolator::GetDeviceGridParams() const {
    return impl_ ? impl_->GetDeviceGridParams() : nullptr;
}

ErrorCode MagneticFieldInterpolator::LaunchInterpolationKernel(const Point3D*       d_query_points,
                                                               InterpolationResult* d_results, size_t count,
                                                               void* stream) {
    if (!impl_) {
        return ErrorCode::DataNotLoaded;
    }
    return impl_->LaunchInterpolationKernel(d_query_points, d_results, count, stream);
}

void MagneticFieldInterpolator::GetOptimalKernelConfig(size_t query_count, KernelConfig& config) const {
    if (impl_) {
        impl_->GetOptimalKernelConfig(query_count, config);
    } else {
        // Default configuration
        const int BLOCK_SIZE = CUDA_DEFAULT_BLOCK_SIZE;
        int       num_blocks = (query_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        config.block_x       = BLOCK_SIZE;
        config.block_y       = 1;
        config.block_z       = 1;
        config.grid_x        = num_blocks;
        config.grid_y        = 1;
        config.grid_z        = 1;
    }
}

ErrorCode MagneticFieldInterpolator::ExportInputPoints(const std::vector<Point3D>& coordinates,
                                                       const std::vector<MagneticFieldData>& field_data,
                                                       ExportFormat format, const std::string& filename) {
    return Impl::ExportInputPoints(coordinates, field_data, format, filename);
}

ErrorCode MagneticFieldInterpolator::ExportOutputPoints(ExportFormat format, const std::vector<Point3D>& query_points,
                                                        const std::vector<InterpolationResult>& results,
                                                        const std::string& filename) {
    return Impl::ExportOutputPoints(format, query_points, results, filename);
}

ErrorCode MagneticFieldInterpolator::GetLastKernelTime(float& kernel_time_ms) const {
    if (!impl_) {
        return ErrorCode::DataNotLoaded;
    }
    return impl_->GetLastKernelTime(kernel_time_ms);
}

}  // namespace p3d
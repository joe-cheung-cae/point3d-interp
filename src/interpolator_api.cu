#include "point3d_interp/interpolator_api.h"
#include "point3d_interp/constants.h"
#include "point3d_interp/data_loader.h"
#include "point3d_interp/grid_structure.h"
#include "point3d_interp/interpolator_interface.h"
#include "point3d_interp/interpolator_factory.h"
#include "point3d_interp/spatial_grid.h"
#include <memory>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <cmath>
#include <cuda_runtime.h>
#include "point3d_interp/memory_manager.h"

namespace p3d {

/**
 * @brief API implementation class (Pimpl pattern)
 */
class MagneticFieldInterpolator::Impl {
  private:
    /**
     * @brief Log CUDA error with consistent formatting
     * @param operation Description of the operation that failed
     * @param cuda_err CUDA error code
     */
    static void LogCudaError(const std::string& operation, cudaError_t cuda_err) {
        std::cerr << "CUDA " << operation << " error: " << cudaGetErrorString(cuda_err) << std::endl;
    }

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

    ErrorCode        LoadFromCSV(const std::string& filepath);
    ErrorCode        LoadFromMemory(const Point3D* points, const MagneticFieldData* field_data, size_t count);
    ErrorCode        Query(const Point3D& query_point, InterpolationResult& result);
    ErrorCode        QueryBatch(const Point3D* query_points, InterpolationResult* results, size_t count);
    static ErrorCode ExportInputPoints(const std::vector<Point3D>&           coordinates,
                                       const std::vector<MagneticFieldData>& field_data, ExportFormat format,
                                       const std::string& filename);
    static ErrorCode ExportOutputPoints(ExportFormat format, const std::vector<Point3D>& query_points,
                                        const std::vector<InterpolationResult>& results, const std::string& filename);

    ErrorCode GetLastKernelTime(float& kernel_time_ms) const {
        if (use_gpu_ && gpu_initialized_ && interpolator_) {
            if (interpolator_->getLastKernelTime(kernel_time_ms)) {
                return ErrorCode::Success;
            }
        }
        return ErrorCode::CudaNotAvailable;
    }

    const GridParams& GetGridParams() const {
        // For structured data, get from interpolator
        if (interpolator_ && interpolator_->getDataType() == DataStructureType::RegularGrid) {
            cached_grid_params_ = interpolator_->getGridParams();
            return cached_grid_params_;
        }
        return default_params_;
    }

    bool   IsDataLoaded() const { return interpolator_ != nullptr; }
    size_t GetDataPointCount() const { return interpolator_ ? interpolator_->getDataCount() : 0; }

    std::vector<Point3D> GetCoordinates() const {
        return interpolator_ ? interpolator_->getCoordinates() : std::vector<Point3D>{};
    }

    std::vector<MagneticFieldData> GetFieldData() const {
        return interpolator_ ? interpolator_->getFieldData() : std::vector<MagneticFieldData>{};
    }

    const Point3D* GetDeviceGridPoints() const { return gpu_grid_points_ ? gpu_grid_points_->getDevicePtr() : nullptr; }

    const MagneticFieldData* GetDeviceFieldData() const {
        return gpu_grid_field_data_ ? gpu_grid_field_data_->getDevicePtr() : nullptr;
    }

    static const GridParams* GetDeviceGridParams() {
        // Note: GridParams is stored on host, need to upload to GPU if needed
        // For now, return nullptr as we don't maintain a GPU copy of GridParams
        return nullptr;
    }

    ErrorCode LaunchInterpolationKernel(const Point3D* d_query_points, InterpolationResult* d_results, size_t count,
                                        void* stream) {
        if (!interpolator_ || interpolator_->getDataType() != DataStructureType::RegularGrid) {
            return ErrorCode::DataNotLoaded;
        }

        if (!d_query_points || !d_results || count == 0) {
            return ErrorCode::InvalidParameter;
        }

        if (use_gpu_ && gpu_initialized_) {
            try {
                // Get optimal kernel configuration
                KernelConfig config;
                GetOptimalKernelConfig(count, config);
                dim3 block_dim(config.block_x, config.block_y, config.block_z);
                dim3 grid_dim(config.grid_x, config.grid_y, config.grid_z);

                // Get grid parameters (host copy) - this needs to be updated for new architecture
                GridParams grid_params = default_params_;  // Placeholder

                // Calculate shared memory size for kernel
                const size_t shared_mem_size = sizeof(GridParams);

                // Launch kernel with specified stream
                cuda::TricubicHermiteInterpolationKernel<<<grid_dim, block_dim, shared_mem_size,
                                                           static_cast<cudaStream_t>(stream)>>>(
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
    }

    static void GetOptimalKernelConfig(size_t query_count, KernelConfig& config) {
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
    static bool InitializeGPU(int device_id);
    void        ReleaseGPU();
    bool        UploadDataToGPU();

    // New architecture: unified interpolator interface
    std::unique_ptr<IInterpolator>       interpolator_;
    std::unique_ptr<InterpolatorFactory> factory_;

    // GPU implementation
    bool                use_gpu_;
    int                 device_id_;
    bool                gpu_initialized_;
    InterpolationMethod method_;
    ExtrapolationMethod extrapolation_method_;

    // GPU memory manager for regular grids (legacy support)
    std::unique_ptr<cuda::GpuMemory<Point3D>>           gpu_grid_points_;
    std::unique_ptr<cuda::GpuMemory<MagneticFieldData>> gpu_grid_field_data_;

    // Shared GPU memory for queries and results (pre-allocated for performance)
    std::unique_ptr<cuda::GpuMemory<Point3D>>             gpu_query_points_;
    std::unique_ptr<cuda::GpuMemory<InterpolationResult>> gpu_results_;
    size_t                                                gpu_memory_capacity_;  // Track allocated capacity

    // Kernel timing
    float last_kernel_time_ms_;

    // Default parameters
    GridParams default_params_;
    // Cached grid parameters for structured data
    mutable GridParams cached_grid_params_;
};

ErrorCode MagneticFieldInterpolator::Impl::ExportInputPoints(const std::vector<Point3D>&           coordinates,
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

ErrorCode MagneticFieldInterpolator::Impl::ExportOutputPoints(ExportFormat                            format,
                                                              const std::vector<Point3D>&             query_points,
                                                              const std::vector<InterpolationResult>& results,
                                                              const std::string&                      filename) {
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
    : use_gpu_(use_gpu),
      device_id_(device_id),
      gpu_initialized_(false),
      method_(method),
      extrapolation_method_(extrapolation_method),
      gpu_memory_capacity_(0),
      last_kernel_time_ms_(0.0f) {
    // Initialize factory
    factory_ = std::make_unique<InterpolatorFactory>();

    if (use_gpu_) {
        gpu_initialized_ = InitializeGPU(device_id);
        if (!gpu_initialized_) {
            LogError("GPU initialization", "falling back to CPU");
            use_gpu_ = false;
        }
    }
}

MagneticFieldInterpolator::Impl::Impl(Impl&& other) noexcept
    : interpolator_(std::move(other.interpolator_)),
      factory_(std::move(other.factory_)),
      use_gpu_(other.use_gpu_),
      device_id_(other.device_id_),
      gpu_initialized_(other.gpu_initialized_),
      method_(other.method_),
      extrapolation_method_(other.extrapolation_method_),
      gpu_memory_capacity_(other.gpu_memory_capacity_) {
    gpu_grid_points_     = std::move(other.gpu_grid_points_);
    gpu_grid_field_data_ = std::move(other.gpu_grid_field_data_);
    gpu_query_points_    = std::move(other.gpu_query_points_);
    gpu_results_         = std::move(other.gpu_results_);
    // Reset moved-from state
    other.gpu_initialized_ = false;
}

MagneticFieldInterpolator::Impl& MagneticFieldInterpolator::Impl::operator=(Impl&& other) noexcept {
    if (this != &other) {
        interpolator_         = std::move(other.interpolator_);
        factory_              = std::move(other.factory_);
        use_gpu_              = other.use_gpu_;
        device_id_            = other.device_id_;
        gpu_initialized_      = other.gpu_initialized_;
        method_               = other.method_;
        extrapolation_method_ = other.extrapolation_method_;
        gpu_memory_capacity_  = other.gpu_memory_capacity_;
        gpu_grid_points_      = std::move(other.gpu_grid_points_);
        gpu_grid_field_data_  = std::move(other.gpu_grid_field_data_);
        gpu_query_points_     = std::move(other.gpu_query_points_);
        gpu_results_          = std::move(other.gpu_results_);
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
        // Convert to vectors
        std::vector<Point3D>           coordinates(points, points + count);
        std::vector<MagneticFieldData> field_values(field_data, field_data + count);

        // Try to determine data structure type and create appropriate interpolator
        DataStructureType dataType = DataStructureType::Unstructured;

        // Check if it's a regular grid
        try {
            RegularGrid3D test_grid(coordinates, field_values);
            dataType = DataStructureType::RegularGrid;
        } catch (const std::invalid_argument&) {
            // Not a regular grid, use unstructured
            dataType = DataStructureType::Unstructured;
        }

        // For unstructured data, we need to use IDW method
        InterpolationMethod method = (dataType == DataStructureType::RegularGrid) ? InterpolationMethod::TricubicHermite
                                                                                  : InterpolationMethod::IDW;

        // Create interpolator using factory
        interpolator_ =
            factory_->createInterpolator(dataType, method, coordinates, field_values, extrapolation_method_, use_gpu_);

        if (!interpolator_) {
            return ErrorCode::InvalidGridData;
        }

        // Upload data to GPU if needed (legacy support for direct kernel access)
        if (use_gpu_ && !UploadDataToGPU()) {
            // Failed to upload to GPU, fall back to CPU
            use_gpu_ = false;
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

    try {
        // Use the new unified interpolator interface
        std::vector<Point3D> query_vec(query_points, query_points + count);
        auto                 cpu_results = interpolator_->queryBatch(query_vec);

        // Copy results
        std::copy(cpu_results.begin(), cpu_results.end(), results);

        return ErrorCode::Success;
    } catch (const std::exception& e) {
        LogError("Query batch", e.what());
        return ErrorCode::InvalidParameter;
    }
}

bool MagneticFieldInterpolator::Impl::InitializeGPU(int device_id) {
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
}

void MagneticFieldInterpolator::Impl::ReleaseGPU() {
    gpu_grid_points_.reset();
    gpu_grid_field_data_.reset();
    gpu_query_points_.reset();
    gpu_results_.reset();

    gpu_initialized_ = false;
}

bool MagneticFieldInterpolator::Impl::UploadDataToGPU() {
    if (!gpu_initialized_ || !interpolator_) {
        return false;
    }

    try {
        // For now, simplified GPU upload - the new architecture handles GPU internally
        // This is kept for legacy direct kernel access support
        if (interpolator_->getDataType() == DataStructureType::RegularGrid) {
            // Allocate minimal GPU memory for legacy support
            gpu_grid_points_     = std::make_unique<cuda::GpuMemory<Point3D>>();
            gpu_grid_field_data_ = std::make_unique<cuda::GpuMemory<MagneticFieldData>>();

            // Allocate with placeholder size - actual upload would be more complex
            if (!gpu_grid_points_->allocate(1) || !gpu_grid_field_data_->allocate(1)) {
                return false;
            }
        }

        return true;
    } catch (const std::exception& e) {
        LogError("GPU data upload", e.what());
        return false;
    }
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

ErrorCode MagneticFieldInterpolator::ExportInputPoints(const std::vector<Point3D>&           coordinates,
                                                       const std::vector<MagneticFieldData>& field_data,
                                                       ExportFormat format, const std::string& filename) {
    return Impl::ExportInputPoints(coordinates, field_data, format, filename);
}

ErrorCode MagneticFieldInterpolator::ExportOutputPoints(ExportFormat format, const std::vector<Point3D>& query_points,
                                                        const std::vector<InterpolationResult>& results,
                                                        const std::string&                      filename) {
    return Impl::ExportOutputPoints(format, query_points, results, filename);
}

ErrorCode MagneticFieldInterpolator::GetLastKernelTime(float& kernel_time_ms) const {
    if (!impl_) {
        return ErrorCode::DataNotLoaded;
    }
    return impl_->GetLastKernelTime(kernel_time_ms);
}

}  // namespace p3d
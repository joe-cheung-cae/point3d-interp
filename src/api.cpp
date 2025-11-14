#include "point3d_interp/api.h"
#include "point3d_interp/data_loader.h"
#include "point3d_interp/grid_structure.h"
#include "point3d_interp/cpu_interpolator.h"
#include "point3d_interp/unstructured_interpolator.h"
#include <memory>
#include <iostream>
#include <stdexcept>
#include <algorithm>

#ifdef __CUDACC__
    #include <cuda_runtime.h>
#endif

namespace p3d {

// Forward declarations for CUDA classes
namespace cuda {
class GpuMemoryManager;
template <typename T>
class GpuMemory;
}  // namespace cuda

/**
 * @brief Data structure type
 */
enum class DataStructureType { RegularGrid, Unstructured };

/**
 * @brief API implementation class (Pimpl pattern)
 */
class MagneticFieldInterpolator::Impl {
  public:
    Impl(bool use_gpu, int device_id, InterpolationMethod method);
    ~Impl();

    // Move constructor
    Impl(Impl&& other) noexcept;
    // Move assignment
    Impl& operator=(Impl&& other) noexcept;

    ErrorCode LoadFromCSV(const std::string& filepath);
    ErrorCode LoadFromMemory(const Point3D* points, const MagneticFieldData* field_data, size_t count);
    ErrorCode Query(const Point3D& query_point, InterpolationResult& result);
    ErrorCode QueryBatch(const Point3D* query_points, InterpolationResult* results, size_t count);

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
        if (!IsDataLoaded()) {
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

                // Launch kernel with specified stream
                cuda::TricubicHermiteInterpolationKernel<<<grid_dim, block_dim, 0, (cudaStream_t)stream>>>(
                    d_query_points, gpu_grid_field_data_->getDevicePtr(), grid_params, d_results, count);

                // Check for kernel launch errors
                cudaError_t cuda_err = cudaGetLastError();
                if (cuda_err != cudaSuccess) {
                    std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(cuda_err) << std::endl;
                    return ErrorCode::CudaError;
                }

                return ErrorCode::Success;
            } catch (const std::exception& e) {
                std::cerr << "GPU kernel launch failed: " << e.what() << std::endl;
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
        const int BLOCK_SIZE = 512;
        const int MIN_BLOCKS = 4;

        int num_blocks       = (query_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        num_blocks           = std::max(num_blocks, MIN_BLOCKS);
        const int MAX_BLOCKS = 1024;
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

#ifdef __CUDACC__
    // GPU memory manager for regular grids
    std::unique_ptr<cuda::GpuMemory<Point3D>>           gpu_grid_points_;
    std::unique_ptr<cuda::GpuMemory<MagneticFieldData>> gpu_grid_field_data_;

    // GPU memory manager for unstructured data
    std::unique_ptr<cuda::GpuMemory<Point3D>>           gpu_unstructured_points_;
    std::unique_ptr<cuda::GpuMemory<MagneticFieldData>> gpu_unstructured_field_data_;

    // Shared GPU memory for queries and results
    std::unique_ptr<cuda::GpuMemory<Point3D>>             gpu_query_points_;
    std::unique_ptr<cuda::GpuMemory<InterpolationResult>> gpu_results_;
#endif

    // Default parameters
    GridParams default_params_;
};

// Implementation

MagneticFieldInterpolator::Impl::Impl(bool use_gpu, int device_id, InterpolationMethod method)
    : data_type_(DataStructureType::RegularGrid),
      use_gpu_(use_gpu),
      device_id_(device_id),
      gpu_initialized_(false),
      method_(method) {
#ifdef __CUDACC__
    if (use_gpu_) {
        gpu_initialized_ = InitializeGPU(device_id);
        if (!gpu_initialized_) {
            std::cerr << "Warning: GPU initialization failed, falling back to CPU" << std::endl;
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
      method_(other.method_) {
#ifdef __CUDACC__
    gpu_grid_points_             = std::move(other.gpu_grid_points_);
    gpu_grid_field_data_         = std::move(other.gpu_grid_field_data_);
    gpu_unstructured_points_     = std::move(other.gpu_unstructured_points_);
    gpu_unstructured_field_data_ = std::move(other.gpu_unstructured_field_data_);
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
#ifdef __CUDACC__
        gpu_grid_points_             = std::move(other.gpu_grid_points_);
        gpu_grid_field_data_         = std::move(other.gpu_grid_field_data_);
        gpu_unstructured_points_     = std::move(other.gpu_unstructured_points_);
        gpu_unstructured_field_data_ = std::move(other.gpu_unstructured_field_data_);
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
    if (!points || !field_data || count == 0) {
        return ErrorCode::InvalidParameter;
    }

    try {
        // Try to create regular grid first
        std::vector<Point3D>           coordinates(points, points + count);
        std::vector<MagneticFieldData> field_values(field_data, field_data + count);

        try {
            grid_             = std::make_unique<RegularGrid3D>(coordinates, field_values);
            cpu_interpolator_ = std::make_unique<CPUInterpolator>(*grid_);
            data_type_        = DataStructureType::RegularGrid;

            // Upload data to GPU
            if (use_gpu_ && !UploadDataToGPU()) {
                // Failed to upload to GPU, fall back to CPU
                use_gpu_ = false;
            }
        } catch (const std::invalid_argument&) {
            // Not a regular grid, use unstructured interpolator
            unstructured_interpolator_ = std::make_unique<UnstructuredInterpolator>(coordinates, field_values);
            data_type_                 = DataStructureType::Unstructured;

            // Upload data to GPU for unstructured data
            if (use_gpu_ && !UploadDataToGPU()) {
                // Failed to upload to GPU, fall back to CPU
                use_gpu_ = false;
            }
        }

        return ErrorCode::Success;
    } catch (const std::exception& e) {
        std::cerr << "Error loading data: " << e.what() << std::endl;
        return ErrorCode::InvalidGridData;
    }
}

ErrorCode MagneticFieldInterpolator::Impl::Query(const Point3D& query_point, InterpolationResult& result) {
    if (!IsDataLoaded()) {
        return ErrorCode::DataNotLoaded;
    }

    if (data_type_ == DataStructureType::RegularGrid) {
        // Use CPU interpolator for regular grid
        result = cpu_interpolator_->query(query_point);
    } else if (data_type_ == DataStructureType::Unstructured) {
        // Use unstructured interpolator
        result = unstructured_interpolator_->query(query_point);
    }

    return ErrorCode::Success;
}

ErrorCode MagneticFieldInterpolator::Impl::QueryBatch(const Point3D* query_points, InterpolationResult* results,
                                                      size_t count) {
    if (!IsDataLoaded()) {
        return ErrorCode::DataNotLoaded;
    }

    if (!query_points || !results || count == 0) {
        return ErrorCode::InvalidParameter;
    }

    if (data_type_ == DataStructureType::Unstructured) {
#ifdef __CUDACC__
        if (use_gpu_ && gpu_initialized_ && gpu_unstructured_points_ && gpu_unstructured_field_data_) {
            // GPU implementation for unstructured data
            try {
                // Ensure GPU memory is sufficient
                if (!gpu_query_points_ || gpu_query_points_->getCount() < count) {
                    gpu_query_points_ = std::make_unique<cuda::GpuMemory<Point3D>>();
                    if (!gpu_query_points_->allocate(count)) {
                        return ErrorCode::CudaError;
                    }
                }

                if (!gpu_results_ || gpu_results_->getCount() < count) {
                    gpu_results_ = std::make_unique<cuda::GpuMemory<InterpolationResult>>();
                    if (!gpu_results_->allocate(count)) {
                        return ErrorCode::CudaError;
                    }
                }

                // Upload query points to GPU
                if (!gpu_query_points_->copyToDevice(query_points, count)) {
                    return ErrorCode::CudaError;
                }

                // Launch IDW kernel
                const int BLOCK_SIZE = 256;
                const int num_blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

                cuda::IDWInterpolationKernel<<<num_blocks, BLOCK_SIZE>>>(
                    gpu_query_points_->getDevicePtr(), gpu_unstructured_points_->getDevicePtr(),
                    gpu_unstructured_field_data_->getDevicePtr(), unstructured_interpolator_->getDataCount(),
                    unstructured_interpolator_->getPower(), gpu_results_->getDevicePtr(), count);

                // Check CUDA errors
                cudaError_t cuda_err = cudaGetLastError();
                if (cuda_err != cudaSuccess) {
                    std::cerr << "CUDA IDW kernel error: " << cudaGetErrorString(cuda_err) << std::endl;
                    return ErrorCode::CudaError;
                }

                // Download results
                if (!gpu_results_->copyToHost(results, count)) {
                    return ErrorCode::CudaError;
                }

                return ErrorCode::Success;
            } catch (const std::exception& e) {
                std::cerr << "GPU IDW query failed: " << e.what() << ", falling back to CPU" << std::endl;
                use_gpu_ = false;
            }
        }
#endif

        // CPU fallback for unstructured data
        std::vector<Point3D> query_vec(query_points, query_points + count);
        auto                 cpu_results = unstructured_interpolator_->queryBatch(query_vec);
        std::copy(cpu_results.begin(), cpu_results.end(), results);
        return ErrorCode::Success;
    }

#ifdef __CUDACC__
    if (use_gpu_ && gpu_initialized_ && data_type_ == DataStructureType::RegularGrid) {
        // GPU implementation
        try {
            // Ensure GPU memory is sufficient
            if (!gpu_query_points_ || gpu_query_points_->getCount() < count) {
                gpu_query_points_ = std::make_unique<cuda::GpuMemory<Point3D>>();
                if (!gpu_query_points_->allocate(count)) {
                    return ErrorCode::CudaError;
                }
            }

            if (!gpu_results_ || gpu_results_->getCount() < count) {
                gpu_results_ = std::make_unique<cuda::GpuMemory<InterpolationResult>>();
                if (!gpu_results_->allocate(count)) {
                    return ErrorCode::CudaError;
                }
            }

            // Upload query points to GPU
            if (!gpu_query_points_->copyToDevice(query_points, count)) {
                return ErrorCode::CudaError;
            }

            // Launch CUDA kernel - optimized configuration
            // Use larger thread blocks for better occupancy
            const int BLOCK_SIZE = 512;  // Increased to 512 for better performance
            const int MIN_BLOCKS = 4;    // Minimum 4 blocks to hide latency

            // Calculate required number of blocks, ensure at least minimum blocks
            int num_blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
            num_blocks     = std::max(num_blocks, MIN_BLOCKS);

            // Limit maximum blocks to avoid excessive resource usage
            const int MAX_BLOCKS = 1024;
            num_blocks           = std::min(num_blocks, MAX_BLOCKS);

            // Configure kernel
            dim3 block_dim(BLOCK_SIZE);
            dim3 grid_dim(num_blocks);

            cuda::TricubicHermiteInterpolationKernel<<<grid_dim, block_dim>>>(
                gpu_query_points_->getDevicePtr(), gpu_grid_field_data_->getDevicePtr(), grid_->getParams(),
                gpu_results_->getDevicePtr(), count);

            // Check CUDA errors
            cudaError_t cuda_err = cudaGetLastError();
            if (cuda_err != cudaSuccess) {
                std::cerr << "CUDA kernel error: " << cudaGetErrorString(cuda_err) << std::endl;
                return ErrorCode::CudaError;
            }

            // Download results
            if (!gpu_results_->copyToHost(results, count)) {
                return ErrorCode::CudaError;
            }

            return ErrorCode::Success;
        } catch (const std::exception& e) {
            std::cerr << "GPU query failed: " << e.what() << ", falling back to CPU" << std::endl;
            use_gpu_ = false;
        }
    }
#endif

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
        std::cerr << "Failed to set CUDA device " << device_id << ": " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Check device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
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
    gpu_query_points_.reset();
    gpu_results_.reset();

    if (gpu_initialized_) {
        cudaDeviceReset();
        gpu_initialized_ = false;
    }
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
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to upload data to GPU: " << e.what() << std::endl;
        return false;
    }
#else
    return false;
#endif
}

// MagneticFieldInterpolator implementation

MagneticFieldInterpolator::MagneticFieldInterpolator(bool use_gpu, int device_id, InterpolationMethod method)
    : impl_(std::make_unique<Impl>(use_gpu, device_id, method)) {}

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
        impl_ = std::make_unique<Impl>(false, 0, InterpolationMethod::TricubicHermite);
    }
    return impl_->LoadFromCSV(filepath);
}

ErrorCode MagneticFieldInterpolator::LoadFromMemory(const Point3D* points, const MagneticFieldData* field_data,
                                                    size_t count) {
    if (!impl_) {
        impl_ = std::make_unique<Impl>(false, 0, InterpolationMethod::TricubicHermite);
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
        const int BLOCK_SIZE = 512;
        int       num_blocks = (query_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        config.block_x       = BLOCK_SIZE;
        config.block_y       = 1;
        config.block_z       = 1;
        config.grid_x        = num_blocks;
        config.grid_y        = 1;
        config.grid_z        = 1;
    }
}

}  // namespace p3d
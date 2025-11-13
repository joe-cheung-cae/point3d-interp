#include "point3d_interp/api.h"
#include "point3d_interp/data_loader.h"
#include "point3d_interp/grid_structure.h"
#include "point3d_interp/cpu_interpolator.h"
#include <memory>
#include <iostream>

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
 * @brief API implementation class (Pimpl pattern)
 */
class MagneticFieldInterpolator::Impl {
  public:
    Impl(bool use_gpu, int device_id);
    ~Impl();

    ErrorCode LoadFromCSV(const std::string& filepath);
    ErrorCode LoadFromMemory(const Point3D* points, const MagneticFieldData* field_data, size_t count);
    ErrorCode Query(const Point3D& query_point, InterpolationResult& result);
    ErrorCode QueryBatch(const Point3D* query_points, InterpolationResult* results, size_t count);

    const GridParams& GetGridParams() const { return grid_ ? grid_->getParams() : default_params_; }
    bool              IsDataLoaded() const { return grid_ != nullptr; }
    size_t            GetDataPointCount() const { return grid_ ? grid_->getDataCount() : 0; }

  private:
    bool InitializeGPU(int device_id);
    void ReleaseGPU();
    bool UploadDataToGPU();

    // CPU implementation
    std::unique_ptr<RegularGrid3D>   grid_;
    std::unique_ptr<CPUInterpolator> cpu_interpolator_;

    // GPU implementation
    bool use_gpu_;
    int  device_id_;
    bool gpu_initialized_;

#ifdef __CUDACC__
    // GPU memory manager
    std::unique_ptr<cuda::GpuMemory<Point3D>>             gpu_points_;
    std::unique_ptr<cuda::GpuMemory<MagneticFieldData>>   gpu_field_data_;
    std::unique_ptr<cuda::GpuMemory<Point3D>>             gpu_query_points_;
    std::unique_ptr<cuda::GpuMemory<InterpolationResult>> gpu_results_;
#endif

    // Default parameters
    GridParams default_params_;
};

// Implementation

MagneticFieldInterpolator::Impl::Impl(bool use_gpu, int device_id)
    : use_gpu_(use_gpu), device_id_(device_id), gpu_initialized_(false) {
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
        // Create grid
        std::vector<Point3D>           coordinates(points, points + count);
        std::vector<MagneticFieldData> field_values(field_data, field_data + count);

        grid_ = std::make_unique<RegularGrid3D>(coordinates, field_values);

        // Create CPU interpolator
        cpu_interpolator_ = std::make_unique<CPUInterpolator>(*grid_);

        // Upload data to GPU
        if (use_gpu_ && !UploadDataToGPU()) {
            std::cerr << "Warning: Failed to upload data to GPU, using CPU only" << std::endl;
            use_gpu_ = false;
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

    // Use CPU interpolator
    result = cpu_interpolator_->query(query_point);

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

#ifdef __CUDACC__
    if (use_gpu_ && gpu_initialized_) {
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
            num_blocks     = max(num_blocks, MIN_BLOCKS);

            // Limit maximum blocks to avoid excessive resource usage
            const int MAX_BLOCKS = 1024;
            num_blocks           = min(num_blocks, MAX_BLOCKS);

            // Configure kernel
            dim3 block_dim(BLOCK_SIZE);
            dim3 grid_dim(num_blocks);

            cuda::TrilinearInterpolationKernel<<<grid_dim, block_dim>>>(
                gpu_query_points_->getDevicePtr(), gpu_field_data_->getDevicePtr(), grid_->getParams(),
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

    // CPU implementation (fallback)
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
    gpu_points_.reset();
    gpu_field_data_.reset();
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
    if (!grid_ || !gpu_initialized_) {
        return false;
    }

    try {
        size_t data_count = grid_->getDataCount();

        // Allocate GPU memory
        gpu_points_     = std::make_unique<cuda::GpuMemory<Point3D>>();
        gpu_field_data_ = std::make_unique<cuda::GpuMemory<MagneticFieldData>>();

        if (!gpu_points_->allocate(data_count) || !gpu_field_data_->allocate(data_count)) {
            return false;
        }

        // Upload data
        const auto& coordinates = grid_->getCoordinates();
        const auto& field_data  = grid_->getFieldData();

        if (!gpu_points_->copyToDevice(coordinates.data(), data_count) ||
            !gpu_field_data_->copyToDevice(field_data.data(), data_count)) {
            return false;
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

MagneticFieldInterpolator::MagneticFieldInterpolator(bool use_gpu, int device_id)
    : impl_(std::make_unique<Impl>(use_gpu, device_id)) {}

MagneticFieldInterpolator::~MagneticFieldInterpolator() = default;

MagneticFieldInterpolator::MagneticFieldInterpolator(MagneticFieldInterpolator&& other) noexcept
    : impl_(std::move(other.impl_)) {}

MagneticFieldInterpolator& MagneticFieldInterpolator::operator=(MagneticFieldInterpolator&& other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
    }
    return *this;
}

ErrorCode MagneticFieldInterpolator::LoadFromCSV(const std::string& filepath) { return impl_->LoadFromCSV(filepath); }

ErrorCode MagneticFieldInterpolator::LoadFromMemory(const Point3D* points, const MagneticFieldData* field_data,
                                                    size_t count) {
    return impl_->LoadFromMemory(points, field_data, count);
}

ErrorCode MagneticFieldInterpolator::Query(const Point3D& query_point, InterpolationResult& result) {
    return impl_->Query(query_point, result);
}

ErrorCode MagneticFieldInterpolator::QueryBatch(const Point3D* query_points, InterpolationResult* results,
                                                size_t count) {
    return impl_->QueryBatch(query_points, results, count);
}

const GridParams& MagneticFieldInterpolator::GetGridParams() const { return impl_->GetGridParams(); }

bool MagneticFieldInterpolator::IsDataLoaded() const { return impl_->IsDataLoaded(); }

size_t MagneticFieldInterpolator::GetDataPointCount() const { return impl_->GetDataPointCount(); }

}  // namespace p3d
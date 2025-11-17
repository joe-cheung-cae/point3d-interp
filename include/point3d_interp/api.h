#ifndef POINTER3D_INTERP_API_H
#define POINTER3D_INTERP_API_H

#include "types.h"
#include "error_codes.h"
#include <string>
#include <vector>
#include <memory>

// CUDA-compatible types for public API
struct KernelConfig {
    unsigned int block_x, block_y, block_z;
    unsigned int grid_x, grid_y, grid_z;

    KernelConfig() : block_x(0), block_y(1), block_z(1), grid_x(0), grid_y(1), grid_z(1) {}
    KernelConfig(unsigned int bx, unsigned int by, unsigned int bz, unsigned int gx, unsigned int gy, unsigned int gz)
        : block_x(bx), block_y(by), block_z(bz), grid_x(gx), grid_y(gy), grid_z(gz) {}
};

#ifdef __CUDACC__
// Forward declaration of CUDA kernel for external use
namespace p3d {
namespace cuda {
__global__ void TricubicHermiteInterpolationKernel(const Point3D* __restrict__ query_points,
                                                   const MagneticFieldData* __restrict__ grid_data,
                                                   GridParams grid_params, InterpolationResult* __restrict__ results,
                                                   size_t     count);

__global__ void IDWSpatialGridKernel(const Point3D* __restrict__ query_points, const Point3D* __restrict__ data_points,
                                     const MagneticFieldData* __restrict__ field_data, const size_t data_count,
                                     const uint32_t* __restrict__ cell_offsets,
                                     const uint32_t* __restrict__ cell_points, const Point3D grid_origin,
                                     const Point3D grid_cell_size, const uint32_t grid_dimensions[3], const Real power,
                                     const int extrapolation_method, const Point3D min_bound, const Point3D max_bound,
                                     InterpolationResult* __restrict__ results, const size_t query_count);

__global__ void IDWInterpolationKernel(const Point3D* __restrict__ query_points,
                                       const Point3D* __restrict__ data_points,
                                       const MagneticFieldData* __restrict__ field_data, const size_t data_count,
                                       const Real power, const int extrapolation_method, const Point3D min_bound,
                                       const Point3D max_bound, InterpolationResult* __restrict__ results,
                                       const size_t  query_count);
}  // namespace cuda
}  // namespace p3d
#endif

namespace p3d {

/**
 * @brief Main magnetic field data interpolator class
 *
 * This is the main interface class of the library, responsible for data loading, GPU resource management, and
 * interpolation calculations
 *
 * @note Thread Safety: This class is not thread-safe. Do not call methods on the same instance concurrently from
 * multiple threads.
 */
class MagneticFieldInterpolator {
  public:
    /**
     * @brief Constructor
     * @param use_gpu Whether to use GPU acceleration (default true)
     * @param device_id GPU device ID (default 0)
     * @param method Interpolation method (default TricubicHermite)
     * @param extrapolation_method Extrapolation method for unstructured data (default None)
     */
    explicit MagneticFieldInterpolator(bool use_gpu = true, int device_id = 0,
                                       InterpolationMethod method               = InterpolationMethod::TricubicHermite,
                                       ExtrapolationMethod extrapolation_method = ExtrapolationMethod::None);

    ~MagneticFieldInterpolator();

    // Disable copy, allow move
    MagneticFieldInterpolator(const MagneticFieldInterpolator&)            = delete;
    MagneticFieldInterpolator& operator=(const MagneticFieldInterpolator&) = delete;
    MagneticFieldInterpolator(MagneticFieldInterpolator&&) noexcept;
    MagneticFieldInterpolator& operator=(MagneticFieldInterpolator&&) noexcept;

    /**
     * @brief Load magnetic field data from CSV file
     * @param filepath CSV file path
     * @return Error code
     */
    ErrorCode LoadFromCSV(const std::string& filepath);

    /**
     * @brief Load data from memory
     * @param points Coordinate array (must not be null, coordinates must be finite, not NaN)
     * @param field_data Magnetic field data array (must not be null, field values must be finite, not NaN)
     * @param count Number of data points (must be > 0 and <= SIZE_MAX/2 to prevent overflow)
     * @return Error code
     *
     * @note Input validation:
     * - points and field_data must not be null
     * - count must be > 0 and <= SIZE_MAX/2
     * - All coordinate values (x, y, z) must be finite (not NaN or infinite)
     * - All magnetic field values (Bx, By, Bz) must be finite (not NaN or infinite)
     */
    ErrorCode LoadFromMemory(const Point3D* points, const MagneticFieldData* field_data, size_t count);

    /**
     * @brief Single point interpolation query
     * @param query_point Query point coordinates (must be finite, not NaN)
     * @param result Output result
     * @return Error code
     *
     * @note Input validation:
     * - Data must be loaded first (call LoadFromCSV or LoadFromMemory)
     * - query_point coordinates must be finite (not NaN or infinite)
     */
    ErrorCode Query(const Point3D& query_point, InterpolationResult& result);

    /**
     * @brief Batch interpolation query
     * @param query_points Query point array (must not be null, coordinates must be finite, not NaN)
     * @param results Output result array (must not be null)
     * @param count Number of query points (must be > 0 and <= SIZE_MAX/2 to prevent overflow)
     * @return Error code
     *
     * @note Input validation:
     * - Data must be loaded first (call LoadFromCSV or LoadFromMemory)
     * - query_points and results must not be null
     * - count must be > 0 and <= SIZE_MAX/2
     * - All query point coordinates must be finite (not NaN or infinite)
     */
    ErrorCode QueryBatch(const Point3D* query_points, InterpolationResult* results, size_t count);

    /**
     * @brief Batch interpolation query with vectors
     * @param query_points Query point vector
     * @param results Output result vector (will be resized)
     * @return Error code
     */
    ErrorCode QueryBatch(const std::vector<Point3D>& query_points, std::vector<InterpolationResult>& results);

    /**
     * @brief Single point interpolation query (throws on error)
     * @param query_point Query point coordinates
     * @return Interpolation result
     * @throws std::runtime_error on error
     */
    InterpolationResult QueryEx(const Point3D& query_point);

    /**
     * @brief Batch interpolation query with vectors (throws on error)
     * @param query_points Query point vector
     * @return Interpolation result vector
     * @throws std::runtime_error on error
     */
    std::vector<InterpolationResult> QueryBatchEx(const std::vector<Point3D>& query_points);

    /**
     * @brief Get grid parameters
     * @return Grid parameters
     */
    const GridParams& GetGridParams() const;

    /**
     * @brief Check if data is loaded
     * @return true if loaded
     */
    bool IsDataLoaded() const;

    /**
     * @brief Get number of data points
     * @return Number of data points
     */
    size_t GetDataPointCount() const;

    /**
     * @brief Get GPU device pointer to grid coordinates (for direct CUDA kernel access)
     * @return Device pointer to grid points, nullptr if not available
     */
    const Point3D* GetDeviceGridPoints() const;

    /**
     * @brief Get GPU device pointer to field data (for direct CUDA kernel access)
     * @return Device pointer to field data, nullptr if not available
     */
    const MagneticFieldData* GetDeviceFieldData() const;

    /**
     * @brief Get GPU device pointer to grid parameters (for direct CUDA kernel access)
     * @return Device pointer to grid parameters, nullptr if not available
     */
    const GridParams* GetDeviceGridParams() const;

    /**
     * @brief Launch interpolation kernel directly with custom device pointers
     * @param d_query_points Device pointer to query points array (must not be null)
     * @param d_results Device pointer to results array (must not be null)
     * @param count Number of query points (must be > 0 and <= SIZE_MAX/2 to prevent overflow)
     * @param stream CUDA stream for asynchronous execution (default nullptr)
     * @return Error code
     *
     * @note Input validation:
     * - Data must be loaded first (call LoadFromCSV or LoadFromMemory)
     * - d_query_points and d_results must not be null
     * - count must be > 0 and <= SIZE_MAX/2
     * - Only works with regular grid data (not unstructured)
     */
    ErrorCode LaunchInterpolationKernel(const Point3D* d_query_points, InterpolationResult* d_results, size_t count,
                                        void* stream = nullptr);

    /**
     * @brief Get optimal kernel launch configuration for given query count
     * @param query_count Number of query points
     * @param config Output kernel configuration
     */
    void GetOptimalKernelConfig(size_t query_count, KernelConfig& config) const;

  private:
    /**
     * @brief Initialize GPU resources
     * @return Whether successful
     */
    bool InitializeGPU(int device_id);

    /**
     * @brief Release GPU resources
     */
    void ReleaseGPU();

    /**
     * @brief Upload data to GPU
     * @return Whether successful
     */
    bool UploadDataToGPU();

    class Impl;  // Pimpl pattern implementation
    std::unique_ptr<Impl> impl_;
};

}  // namespace p3d

#endif  // POINTER3D_INTERP_API_H
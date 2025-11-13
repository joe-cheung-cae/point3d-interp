#ifndef POINTER3D_INTERP_API_H
#define POINTER3D_INTERP_API_H

#include "types.h"
#include "error_codes.h"
#include <string>
#include <vector>
#include <memory>

namespace p3d {

/**
 * @brief Main magnetic field data interpolator class
 *
 * This is the main interface class of the library, responsible for data loading, GPU resource management, and
 * interpolation calculations
 */
class MagneticFieldInterpolator {
  public:
    /**
     * @brief Constructor
     * @param use_gpu Whether to use GPU acceleration (default true)
     * @param device_id GPU device ID (default 0)
     */
    explicit MagneticFieldInterpolator(bool use_gpu = true, int device_id = 0);

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
     * @param points Coordinate array
     * @param field_data Magnetic field data array
     * @param count Number of data points
     * @return Error code
     */
    ErrorCode LoadFromMemory(const Point3D* points, const MagneticFieldData* field_data, size_t count);

    /**
     * @brief Single point interpolation query
     * @param query_point Query point coordinates
     * @param result Output result
     * @return Error code
     */
    ErrorCode Query(const Point3D& query_point, InterpolationResult& result);

    /**
     * @brief Batch interpolation query
     * @param query_points Query point array
     * @param results Output result array
     * @param count Number of query points
     * @return Error code
     */
    ErrorCode QueryBatch(const Point3D* query_points, InterpolationResult* results, size_t count);

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

  private:
    class Impl;  // Pimpl pattern implementation
    std::unique_ptr<Impl> impl_;
};

}  // namespace p3d

#endif  // POINTER3D_INTERP_API_H
#pragma once

#include "interpolator_interface.h"
#include "grid_structure.h"
#include <memory>
#include <vector>

P3D_NAMESPACE_BEGIN

/**
 * @brief Concrete interpolator factory implementation
 */
class InterpolatorFactory : public IInterpolatorFactory {
  public:
    std::unique_ptr<IInterpolator> createInterpolator(DataStructureType dataType, InterpolationMethod method,
                                                      const std::vector<Point3D>&           coordinates,
                                                      const std::vector<MagneticFieldData>& fieldData,
                                                      ExtrapolationMethod extrapolation, bool useGPU) override;

    bool supports(DataStructureType dataType, InterpolationMethod method, bool useGPU) const override;

  private:
    /**
     * @brief Attempt to create a structured grid interpolator
     */
    std::unique_ptr<IInterpolator> createStructuredInterpolator(InterpolationMethod                   method,
                                                                const std::vector<Point3D>&           coordinates,
                                                                const std::vector<MagneticFieldData>& fieldData,
                                                                ExtrapolationMethod extrapolation, bool useGPU);

    /**
     * @brief Create an unstructured data interpolator
     */
    std::unique_ptr<IInterpolator> createUnstructuredInterpolator(InterpolationMethod                   method,
                                                                  const std::vector<Point3D>&           coordinates,
                                                                  const std::vector<MagneticFieldData>& fieldData,
                                                                  ExtrapolationMethod extrapolation, bool useGPU);

    /**
     * @brief Check if coordinates form a regular grid
     */
    static bool isRegularGrid(const std::vector<Point3D>& coordinates, const std::vector<MagneticFieldData>& fieldData,
                              std::unique_ptr<RegularGrid3D>& grid);
};

/**
 * @brief Plugin-based interpolator factory for extensibility
 */
class PluginInterpolatorFactory : public IInterpolatorFactory {
  public:
    /**
     * @brief Register a plugin factory
     */
    void registerPlugin(std::unique_ptr<IInterpolatorFactory> plugin);

    std::unique_ptr<IInterpolator> createInterpolator(DataStructureType dataType, InterpolationMethod method,
                                                      const std::vector<Point3D>&           coordinates,
                                                      const std::vector<MagneticFieldData>& fieldData,
                                                      ExtrapolationMethod extrapolation, bool useGPU) override;

    bool supports(DataStructureType dataType, InterpolationMethod method, bool useGPU) const override;

  private:
    std::vector<std::unique_ptr<IInterpolatorFactory>> plugins_;
};

/**
 * @brief Global interpolator factory instance
 */
class GlobalInterpolatorFactory {
  public:
    static GlobalInterpolatorFactory& instance();

    void                           registerFactory(std::unique_ptr<IInterpolatorFactory> factory);
    std::unique_ptr<IInterpolator> createInterpolator(DataStructureType dataType, InterpolationMethod method,
                                                      const std::vector<Point3D>&           coordinates,
                                                      const std::vector<MagneticFieldData>& fieldData,
                                                      ExtrapolationMethod extrapolation = ExtrapolationMethod::None,
                                                      bool                useGPU        = false);

  private:
    GlobalInterpolatorFactory() = default;
    std::vector<std::unique_ptr<IInterpolatorFactory>> factories_;
};

P3D_NAMESPACE_END

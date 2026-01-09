#include "point3d_interp/interpolator_factory.h"
#include <stdexcept>
#include <algorithm>
#include "point3d_interp/interpolator_adapters.h"

P3D_NAMESPACE_BEGIN

// InterpolatorFactory implementation

std::unique_ptr<IInterpolator> InterpolatorFactory::createInterpolator(DataStructureType           dataType,
                                                                       InterpolationMethod         method,
                                                                       const std::vector<Point3D>& coordinates,
                                                                       const std::vector<MagneticFieldData>& fieldData,
                                                                       ExtrapolationMethod extrapolation, bool useGPU) {
    if (!supports(dataType, method, useGPU)) {
        throw std::invalid_argument("Unsupported interpolator configuration");
    }

    if (dataType == DataStructureType::RegularGrid) {
        return createStructuredInterpolator(method, coordinates, fieldData, extrapolation, useGPU);
    } else {
        return createUnstructuredInterpolator(method, coordinates, fieldData, extrapolation, useGPU);
    }
}

bool InterpolatorFactory::supports(DataStructureType dataType, InterpolationMethod method, bool useGPU) const {
    // Support current methods
    if (method != InterpolationMethod::TricubicHermite &&
        method != InterpolationMethod::IDW &&
        method != InterpolationMethod::HermiteMLS) {
        return false;
    }

    // Check data type compatibility
    if (dataType == DataStructureType::RegularGrid && method != InterpolationMethod::TricubicHermite) {
        return false;
    }
    if (dataType == DataStructureType::Unstructured &&
        method != InterpolationMethod::IDW &&
        method != InterpolationMethod::HermiteMLS) {
        return false;
    }

    // For now, GPU support is limited
    if (useGPU) {
        // Only support GPU for basic configurations
        return (dataType == DataStructureType::RegularGrid && method == InterpolationMethod::TricubicHermite) ||
               (dataType == DataStructureType::Unstructured && method == InterpolationMethod::IDW) ||
               (dataType == DataStructureType::Unstructured && method == InterpolationMethod::HermiteMLS);
    }

    return true;
}

std::unique_ptr<IInterpolator> InterpolatorFactory::createStructuredInterpolator(
    InterpolationMethod method, const std::vector<Point3D>& coordinates,
    const std::vector<MagneticFieldData>& fieldData, ExtrapolationMethod extrapolation, bool useGPU) {
    // Try to create regular grid
    std::unique_ptr<RegularGrid3D> grid;
    if (!isRegularGrid(coordinates, fieldData, grid)) {
        throw std::invalid_argument("Coordinates do not form a regular grid");
    }

    if (useGPU) {
        return std::make_unique<GPUStructuredInterpolatorAdapter>(std::move(grid), method, extrapolation);
    } else {
        return std::make_unique<CPUStructuredInterpolatorAdapter>(std::move(grid), method, extrapolation);
    }
}

std::unique_ptr<IInterpolator> InterpolatorFactory::createUnstructuredInterpolator(
    InterpolationMethod method, const std::vector<Point3D>& coordinates,
    const std::vector<MagneticFieldData>& fieldData, ExtrapolationMethod extrapolation, bool useGPU) {
    
    // Handle HermiteMLS method
    if (method == InterpolationMethod::HermiteMLS) {
        HermiteMLSInterpolator::Parameters params;  // Use default parameters
        auto hmls_interpolator = std::make_unique<HermiteMLSInterpolator>(coordinates, fieldData, params);
        
        if (useGPU) {
            return std::make_unique<GPUHermiteMLSInterpolatorAdapter>(std::move(hmls_interpolator), method, extrapolation);
        } else {
            return std::make_unique<CPUHermiteMLSInterpolatorAdapter>(std::move(hmls_interpolator), method, extrapolation);
        }
    }
    
    // Handle IDW method (original code)
    Real   power         = 2.0f;  // Default for IDW
    size_t max_neighbors = 0;     // Use all neighbors by default

    auto interpolator =
        std::make_unique<UnstructuredInterpolator>(coordinates, fieldData, power, max_neighbors, extrapolation);

    if (useGPU) {
        return std::make_unique<GPUUnstructuredInterpolatorAdapter>(std::move(interpolator), method, extrapolation);
    } else {
        return std::make_unique<CPUUnstructuredInterpolatorAdapter>(std::move(interpolator), method, extrapolation);
    }
}

bool InterpolatorFactory::isRegularGrid(const std::vector<Point3D>&           coordinates,
                                        const std::vector<MagneticFieldData>& fieldData,
                                        std::unique_ptr<RegularGrid3D>&       grid) {
    try {
        grid = std::make_unique<RegularGrid3D>(coordinates, fieldData);
        return true;
    } catch (const std::invalid_argument&) {
        return false;
    }
}

// PluginInterpolatorFactory implementation

void PluginInterpolatorFactory::registerPlugin(std::unique_ptr<IInterpolatorFactory> plugin) {
    plugins_.push_back(std::move(plugin));
}

std::unique_ptr<IInterpolator> PluginInterpolatorFactory::createInterpolator(
    DataStructureType dataType, InterpolationMethod method, const std::vector<Point3D>& coordinates,
    const std::vector<MagneticFieldData>& fieldData, ExtrapolationMethod extrapolation, bool useGPU) {
    for (auto& plugin : plugins_) {
        if (plugin->supports(dataType, method, useGPU)) {
            return plugin->createInterpolator(dataType, method, coordinates, fieldData, extrapolation, useGPU);
        }
    }

    throw std::invalid_argument("No plugin supports the requested interpolator configuration");
}

bool PluginInterpolatorFactory::supports(DataStructureType dataType, InterpolationMethod method, bool useGPU) const {
    return std::any_of(plugins_.begin(), plugins_.end(),
                       [&](const auto& plugin) { return plugin->supports(dataType, method, useGPU); });
}

// GlobalInterpolatorFactory implementation

GlobalInterpolatorFactory& GlobalInterpolatorFactory::instance() {
    static GlobalInterpolatorFactory instance;
    return instance;
}

void GlobalInterpolatorFactory::registerFactory(std::unique_ptr<IInterpolatorFactory> factory) {
    factories_.push_back(std::move(factory));
}

std::unique_ptr<IInterpolator> GlobalInterpolatorFactory::createInterpolator(
    DataStructureType dataType, InterpolationMethod method, const std::vector<Point3D>& coordinates,
    const std::vector<MagneticFieldData>& fieldData, ExtrapolationMethod extrapolation, bool useGPU) {
    for (auto& factory : factories_) {
        if (factory->supports(dataType, method, useGPU)) {
            return factory->createInterpolator(dataType, method, coordinates, fieldData, extrapolation, useGPU);
        }
    }

    throw std::invalid_argument("No factory supports the requested interpolator configuration");
}

P3D_NAMESPACE_END
#pragma once

#include "types.h"
#include <string>
#include <vector>
#include <memory>

namespace p3d {

/**
 * @brief Export format enumeration
 */
enum class ExportFormat {
    ParaviewVTK,  // VTK legacy format for Paraview
    BinaryData,   // Fast binary format for internal use
    Tecplot       // Reserved for future implementation
};

/**
 * @brief Base exporter class for different visualization formats
 *
 * This abstract base class defines the interface for exporting interpolation data
 * to various visualization formats. Concrete implementations handle specific formats.
 */
class Exporter {
  public:
    /**
     * @brief Constructor
     * @param format The export format
     */
    explicit Exporter(ExportFormat format) : format_(format) {}

    /**
     * @brief Virtual destructor
     */
    virtual ~Exporter() = default;

    /**
     * @brief Export input sampling points with magnetic field data
     * @param points Input sampling point coordinates
     * @param field_data Magnetic field data at each point
     * @param filename Output filename
     * @return true if export successful
     */
    virtual bool ExportInputPoints(const std::vector<Point3D>& points, const std::vector<MagneticFieldData>& field_data,
                                   const std::string& filename) = 0;

    /**
     * @brief Export output interpolation points with results
     * @param points Query point coordinates
     * @param results Interpolation results at each point
     * @param filename Output filename
     * @return true if export successful
     */
    virtual bool ExportOutputPoints(const std::vector<Point3D>& points, const std::vector<InterpolationResult>& results,
                                    const std::string& filename) = 0;

    /**
     * @brief Get the export format
     * @return Export format
     */
    ExportFormat GetFormat() const { return format_; }

  protected:
    ExportFormat format_;
};

/**
 * @brief Factory function to create exporters
 * @param format The desired export format
 * @return Unique pointer to exporter instance
 */
std::unique_ptr<Exporter> CreateExporter(ExportFormat format);

}  // namespace p3d

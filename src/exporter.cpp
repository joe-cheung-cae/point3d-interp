#include "point3d_interp/exporter.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace p3d {

/**
 * @brief Paraview VTK exporter implementation
 *
 * Exports data in VTK legacy unstructured grid format compatible with Paraview.
 */
class ParaviewExporter : public Exporter {
  public:
    ParaviewExporter() : Exporter(ExportFormat::ParaviewVTK) {}

    bool ExportInputPoints(const std::vector<Point3D>& points, const std::vector<MagneticFieldData>& field_data,
                           const std::string& filename) override;

    bool ExportOutputPoints(const std::vector<Point3D>& points, const std::vector<InterpolationResult>& results,
                            const std::string& filename) override;

  private:
    /**
     * @brief Write VTK header
     * @param file Output file stream
     * @param title Dataset title
     * @param num_points Number of points
     */
    static void WriteVTKHeader(std::ofstream& file, const std::string& title, size_t num_points);

    /**
     * @brief Write points section
     * @param file Output file stream
     * @param points Point coordinates
     */
    static void WritePoints(std::ofstream& file, const std::vector<Point3D>& points);

    /**
     * @brief Write cells section for unstructured grid (vertices)
     * @param file Output file stream
     * @param num_points Number of points
     */
    static void WriteCells(std::ofstream& file, size_t num_points);

    /**
     * @brief Write cell types section
     * @param file Output file stream
     * @param num_points Number of points
     */
    static void WriteCellTypes(std::ofstream& file, size_t num_points);

    /**
     * @brief Write point data section for input points
     * @param file Output file stream
     * @param field_data Magnetic field data
     */
    static void WriteInputPointData(std::ofstream& file, const std::vector<MagneticFieldData>& field_data);

    /**
     * @brief Write point data section for output points
     * @param file Output file stream
     * @param results Interpolation results
     */
    static void WriteOutputPointData(std::ofstream& file, const std::vector<InterpolationResult>& results);
};

bool ParaviewExporter::ExportInputPoints(const std::vector<Point3D>&           points,
                                         const std::vector<MagneticFieldData>& field_data,
                                         const std::string&                    filename) {
    if (points.size() != field_data.size()) {
        std::cerr << "Error: Number of points (" << points.size() << ") does not match number of field data entries ("
                  << field_data.size() << ")" << std::endl;
        return false;
    }

    if (points.empty()) {
        std::cerr << "Error: No data to export" << std::endl;
        return false;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return false;
    }

    try {
        WriteVTKHeader(file, "Input Sampling Points", points.size());
        WritePoints(file, points);
        WriteCells(file, points.size());
        WriteCellTypes(file, points.size());
        WriteInputPointData(file, field_data);

        file.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error writing VTK file: " << e.what() << std::endl;
        file.close();
        return false;
    }
}

bool ParaviewExporter::ExportOutputPoints(const std::vector<Point3D>&             points,
                                          const std::vector<InterpolationResult>& results,
                                          const std::string&                      filename) {
    if (points.size() != results.size()) {
        std::cerr << "Error: Number of points (" << points.size() << ") does not match number of results ("
                  << results.size() << ")" << std::endl;
        return false;
    }

    if (points.empty()) {
        std::cerr << "Error: No data to export" << std::endl;
        return false;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return false;
    }

    try {
        WriteVTKHeader(file, "Output Interpolation Points", points.size());
        WritePoints(file, points);
        WriteCells(file, points.size());
        WriteCellTypes(file, points.size());
        WriteOutputPointData(file, results);

        file.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error writing VTK file: " << e.what() << std::endl;
        file.close();
        return false;
    }
}

void ParaviewExporter::WriteVTKHeader(std::ofstream& file, const std::string& title, size_t num_points) {
    file << "# vtk DataFile Version 3.0" << std::endl;
    file << title << std::endl;
    file << "ASCII" << std::endl;
    file << "DATASET UNSTRUCTURED_GRID" << std::endl;
    file << "POINTS " << num_points << " float" << std::endl;
}

void ParaviewExporter::WritePoints(std::ofstream& file, const std::vector<Point3D>& points) {
    file << std::fixed << std::setprecision(6);
    for (const auto& point : points) {
        file << point.x << " " << point.y << " " << point.z << std::endl;
    }
    file << std::endl;
}

void ParaviewExporter::WriteCells(std::ofstream& file, size_t num_points) {
    file << "CELLS " << num_points << " " << (2 * num_points) << std::endl;
    for (size_t i = 0; i < num_points; ++i) {
        file << "1 " << i << std::endl;
    }
    file << std::endl;
}

void ParaviewExporter::WriteCellTypes(std::ofstream& file, size_t num_points) {
    file << "CELL_TYPES " << num_points << std::endl;
    for (size_t i = 0; i < num_points; ++i) {
        file << "1" << std::endl;  // VTK_VERTEX
    }
    file << std::endl;
}

void ParaviewExporter::WriteInputPointData(std::ofstream& file, const std::vector<MagneticFieldData>& field_data) {
    size_t num_points = field_data.size();
    file << "POINT_DATA " << num_points << std::endl;

    // Magnetic field vector
    file << "VECTORS B float" << std::endl;
    file << std::fixed << std::setprecision(6);
    for (const auto& data : field_data) {
        file << data.Bx << " " << data.By << " " << data.Bz << std::endl;
    }
    file << std::endl;

    // Derivative scalars
    file << "SCALARS dBx_dx float" << std::endl;
    file << "LOOKUP_TABLE default" << std::endl;
    for (const auto& data : field_data) {
        file << std::fixed << std::setprecision(6) << data.dBx_dx << std::endl;
    }
    file << std::endl;

    file << "SCALARS dBx_dy float" << std::endl;
    file << "LOOKUP_TABLE default" << std::endl;
    for (const auto& data : field_data) {
        file << std::fixed << std::setprecision(6) << data.dBx_dy << std::endl;
    }
    file << std::endl;

    file << "SCALARS dBx_dz float" << std::endl;
    file << "LOOKUP_TABLE default" << std::endl;
    for (const auto& data : field_data) {
        file << std::fixed << std::setprecision(6) << data.dBx_dz << std::endl;
    }
    file << std::endl;

    file << "SCALARS dBy_dx float" << std::endl;
    file << "LOOKUP_TABLE default" << std::endl;
    for (const auto& data : field_data) {
        file << std::fixed << std::setprecision(6) << data.dBy_dx << std::endl;
    }
    file << std::endl;

    file << "SCALARS dBy_dy float" << std::endl;
    file << "LOOKUP_TABLE default" << std::endl;
    for (const auto& data : field_data) {
        file << std::fixed << std::setprecision(6) << data.dBy_dy << std::endl;
    }
    file << std::endl;

    file << "SCALARS dBy_dz float" << std::endl;
    file << "LOOKUP_TABLE default" << std::endl;
    for (const auto& data : field_data) {
        file << std::fixed << std::setprecision(6) << data.dBy_dz << std::endl;
    }
    file << std::endl;

    file << "SCALARS dBz_dx float" << std::endl;
    file << "LOOKUP_TABLE default" << std::endl;
    for (const auto& data : field_data) {
        file << std::fixed << std::setprecision(6) << data.dBz_dx << std::endl;
    }
    file << std::endl;

    file << "SCALARS dBz_dy float" << std::endl;
    file << "LOOKUP_TABLE default" << std::endl;
    for (const auto& data : field_data) {
        file << std::fixed << std::setprecision(6) << data.dBz_dy << std::endl;
    }
    file << std::endl;

    file << "SCALARS dBz_dz float" << std::endl;
    file << "LOOKUP_TABLE default" << std::endl;
    for (const auto& data : field_data) {
        file << std::fixed << std::setprecision(6) << data.dBz_dz << std::endl;
    }
    file << std::endl;
}

void ParaviewExporter::WriteOutputPointData(std::ofstream& file, const std::vector<InterpolationResult>& results) {
    size_t num_points = results.size();
    file << "POINT_DATA " << num_points << std::endl;

    // Magnetic field vector
    file << "VECTORS B float" << std::endl;
    file << std::fixed << std::setprecision(6);
    for (const auto& result : results) {
        file << result.data.Bx << " " << result.data.By << " " << result.data.Bz << std::endl;
    }
    file << std::endl;

    // Derivative scalars
    const char* derivative_names[9] = {"dBx_dx", "dBx_dy", "dBx_dz", "dBy_dx", "dBy_dy",
                                       "dBy_dz", "dBz_dx", "dBz_dy", "dBz_dz"};

    for (int i = 0; i < 9; ++i) {
        file << "SCALARS " << derivative_names[i] << " float" << std::endl;
        file << "LOOKUP_TABLE default" << std::endl;
        for (const auto& result : results) {
            Real value = 0.0f;
            switch (i) {
                case 0:
                    value = result.data.dBx_dx;
                    break;
                case 1:
                    value = result.data.dBx_dy;
                    break;
                case 2:
                    value = result.data.dBx_dz;
                    break;
                case 3:
                    value = result.data.dBy_dx;
                    break;
                case 4:
                    value = result.data.dBy_dy;
                    break;
                case 5:
                    value = result.data.dBy_dz;
                    break;
                case 6:
                    value = result.data.dBz_dx;
                    break;
                case 7:
                    value = result.data.dBz_dy;
                    break;
                case 8:
                    value = result.data.dBz_dz;
                    break;
            }
            file << std::fixed << std::setprecision(6) << value << std::endl;
        }
        file << std::endl;
    }

    // Validity scalar
    file << "SCALARS validity int" << std::endl;
    file << "LOOKUP_TABLE default" << std::endl;
    for (const auto& result : results) {
        file << (result.valid ? 1 : 0) << std::endl;
    }
    file << std::endl;
}

// Factory function implementation
std::unique_ptr<Exporter> CreateExporter(ExportFormat format) {
    switch (format) {
        case ExportFormat::ParaviewVTK:
            return std::make_unique<ParaviewExporter>();
        case ExportFormat::Tecplot:
            // Reserved for future implementation
            std::cerr << "Tecplot export not yet implemented" << std::endl;
            return nullptr;
        default:
            std::cerr << "Unknown export format" << std::endl;
            return nullptr;
    }
}

}  // namespace p3d
#include "point3d_interp/interpolator_exporter.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

P3D_NAMESPACE_BEGIN

/**
 * @brief Binary data exporter implementation
 *
 * Exports data in fast binary format for internal use. Much faster than text formats.
 */
class BinaryExporter : public Exporter {
  public:
    BinaryExporter() : Exporter(ExportFormat::BinaryData) {}

    bool ExportInputPoints(const std::vector<Point3D>& points, const std::vector<MagneticFieldData>& field_data,
                           const std::string& filename) override;

    bool ExportOutputPoints(const std::vector<Point3D>& points, const std::vector<InterpolationResult>& results,
                            const std::string& filename) override;

  private:
    /**
     * @brief Write binary header with format version and metadata
     */
    static void WriteBinaryHeader(std::ofstream& file, uint32_t num_points, bool is_input_data);

    /**
     * @brief Write grid parameters in binary format
     */
    static void WriteGridParams(std::ofstream& file, const GridParams& grid_params);
};

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

    // Estimate buffer size for better performance
    // Each point: ~50 chars for vectors + ~10 chars per scalar * 9 scalars + newlines
    size_t      estimated_size = num_points * 150;
    std::string buffer;
    buffer.reserve(estimated_size);

    // Use stringstream for simplicity and correctness, but reserve capacity
    std::stringstream ss;
    ss << "POINT_DATA " << num_points << "\n";

    // Magnetic field vector
    ss << "VECTORS B float\n";
    ss << std::fixed << std::setprecision(6);
    for (const auto& data : field_data) {
        ss << data.Bx << " " << data.By << " " << data.Bz << "\n";
    }
    ss << "\n";

    // Derivative scalars
    const char* scalar_names[9] = {"dBx_dx", "dBx_dy", "dBx_dz", "dBy_dx", "dBy_dy",
                                   "dBy_dz", "dBz_dx", "dBz_dy", "dBz_dz"};

    for (int s = 0; s < 9; ++s) {
        ss << "SCALARS " << scalar_names[s] << " float\n";
        ss << "LOOKUP_TABLE default\n";

        for (const auto& data : field_data) {
            float value = 0.0f;
            switch (s) {
                case 0:
                    value = data.dBx_dx;
                    break;
                case 1:
                    value = data.dBx_dy;
                    break;
                case 2:
                    value = data.dBx_dz;
                    break;
                case 3:
                    value = data.dBy_dx;
                    break;
                case 4:
                    value = data.dBy_dy;
                    break;
                case 5:
                    value = data.dBy_dz;
                    break;
                case 6:
                    value = data.dBz_dx;
                    break;
                case 7:
                    value = data.dBz_dy;
                    break;
                case 8:
                    value = data.dBz_dz;
                    break;
            }
            ss << std::fixed << std::setprecision(6) << value << "\n";
        }
        ss << "\n";
    }

    // Write the entire buffer at once
    file << ss.str();

    // Write the entire buffer at once
    file << buffer;
}

void ParaviewExporter::WriteOutputPointData(std::ofstream& file, const std::vector<InterpolationResult>& results) {
    size_t num_points = results.size();

    // Use stringstream for buffered writing
    std::stringstream ss;
    ss << "POINT_DATA " << num_points << "\n";

    // Magnetic field vector
    ss << "VECTORS B float\n";
    ss << std::fixed << std::setprecision(6);
    for (const auto& result : results) {
        ss << result.data.Bx << " " << result.data.By << " " << result.data.Bz << "\n";
    }
    ss << "\n";

    // Derivative scalars
    const char* derivative_names[9] = {"dBx_dx", "dBx_dy", "dBx_dz", "dBy_dx", "dBy_dy",
                                       "dBy_dz", "dBz_dx", "dBz_dy", "dBz_dz"};

    for (int i = 0; i < 9; ++i) {
        ss << "SCALARS " << derivative_names[i] << " float\n";
        ss << "LOOKUP_TABLE default\n";

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
            ss << std::fixed << std::setprecision(6) << value << "\n";
        }
        ss << "\n";
    }

    // Validity scalar
    ss << "SCALARS validity int\n";
    ss << "LOOKUP_TABLE default\n";
    for (const auto& result : results) {
        ss << (result.valid ? 1 : 0) << "\n";
    }
    ss << "\n";

    // Write the entire buffer at once
    file << ss.str();
}

// Binary exporter implementation
bool BinaryExporter::ExportInputPoints(const std::vector<Point3D>&           points,
                                       const std::vector<MagneticFieldData>& field_data, const std::string& filename) {
    if (points.size() != field_data.size()) {
        std::cerr << "Error: Number of points (" << points.size() << ") does not match number of field data entries ("
                  << field_data.size() << ")" << std::endl;
        return false;
    }

    if (points.empty()) {
        std::cerr << "Error: No data to export" << std::endl;
        return false;
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return false;
    }

    try {
        // Write header
        WriteBinaryHeader(file, points.size(), true);

        // Write points data
        file.write(reinterpret_cast<const char*>(points.data()), points.size() * sizeof(Point3D));

        // Write field data
        file.write(reinterpret_cast<const char*>(field_data.data()), field_data.size() * sizeof(MagneticFieldData));

        file.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error writing binary file: " << e.what() << std::endl;
        file.close();
        return false;
    }
}

bool BinaryExporter::ExportOutputPoints(const std::vector<Point3D>&             points,
                                        const std::vector<InterpolationResult>& results, const std::string& filename) {
    if (points.size() != results.size()) {
        std::cerr << "Error: Number of points (" << points.size() << ") does not match number of results ("
                  << results.size() << ")" << std::endl;
        return false;
    }

    if (points.empty()) {
        std::cerr << "Error: No data to export" << std::endl;
        return false;
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return false;
    }

    try {
        // Write header
        WriteBinaryHeader(file, points.size(), false);

        // Write points data
        file.write(reinterpret_cast<const char*>(points.data()), points.size() * sizeof(Point3D));

        // Write results data
        file.write(reinterpret_cast<const char*>(results.data()), results.size() * sizeof(InterpolationResult));

        file.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error writing binary file: " << e.what() << std::endl;
        file.close();
        return false;
    }
}

void BinaryExporter::WriteBinaryHeader(std::ofstream& file, uint32_t num_points, bool is_input_data) {
    // Magic number for format identification
    const uint32_t MAGIC_NUMBER   = 0x50494441;  // "PIDA" (Point3D Interpolation Data)
    const uint32_t FORMAT_VERSION = 1;

    file.write(reinterpret_cast<const char*>(&MAGIC_NUMBER), sizeof(MAGIC_NUMBER));
    file.write(reinterpret_cast<const char*>(&FORMAT_VERSION), sizeof(FORMAT_VERSION));
    file.write(reinterpret_cast<const char*>(&num_points), sizeof(num_points));
    file.write(reinterpret_cast<const char*>(&is_input_data), sizeof(is_input_data));
}

// Factory function implementation
std::unique_ptr<Exporter> CreateExporter(ExportFormat format) {
    switch (format) {
        case ExportFormat::ParaviewVTK:
            return std::make_unique<ParaviewExporter>();
        case ExportFormat::BinaryData:
            return std::make_unique<BinaryExporter>();
        case ExportFormat::Tecplot:
            // Reserved for future implementation
            std::cerr << "Tecplot export not yet implemented" << std::endl;
            return nullptr;
        default:
            std::cerr << "Unknown export format" << std::endl;
            return nullptr;
    }
}

P3D_NAMESPACE_END
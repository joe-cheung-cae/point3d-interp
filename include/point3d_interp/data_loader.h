#pragma once

#include "types.h"
#include "error_codes.h"
#include <string>
#include <vector>
#include <memory>

namespace p3d {

/**
 * @brief CSV data loader
 *
 * Responsible for loading magnetic field data from CSV files, supports multiple delimiters and formats
 */
class DataLoader {
  public:
    DataLoader();
    ~DataLoader();

    /**
     * @brief Load data from CSV file
     * @param filepath CSV file path
     * @param coordinates Output coordinate array
     * @param field_data Output magnetic field data array
     * @param grid_params Output grid parameters
     * @return Error code
     */
    ErrorCode LoadFromCSV(const std::string& filepath, std::vector<Point3D>& coordinates,
                          std::vector<MagneticFieldData>& field_data, GridParams& grid_params);

    /**
     * @brief Set delimiter
     * @param delimiter Delimiter (default comma)
     */
    void SetDelimiter(char delimiter) { delimiter_ = delimiter; }

    /**
     * @brief Set whether to skip header row
     * @param skip_header Whether to skip header row (default true)
     */
    void SetSkipHeader(bool skip_header) { skip_header_ = skip_header; }

    /**
     * @brief Set column indices
     * @param coord_cols Coordinate column indices [x, y, z]
     * @param field_cols Magnetic field data column indices [Bx, By, Bz, dBx_dx, dBx_dy, dBx_dz, dBy_dx, dBy_dy,
     * dBy_dz, dBz_dx, dBz_dy, dBz_dz]
     */
    void SetColumnIndices(const std::array<size_t, 3>& coord_cols, const std::array<size_t, 12>& field_cols);

    /**
     * @brief Set tolerance for floating point comparisons
     * @param tolerance Tolerance value (default 1e-6)
     */
    void SetTolerance(Real tolerance) { tolerance_ = tolerance; }

  private:
    /**
     * @brief Parse single line data
     * @param line Input line
     * @param point Output coordinate point
     * @param field Output magnetic field data
     * @return Whether parsing succeeded
     */
    bool ParseLine(const std::string& line, Point3D& point, MagneticFieldData& field);

    /**
     * @brief Detect grid parameters from data
     * @param coordinates Coordinate array
     * @param grid_params Output grid parameters
     * @return Whether detection succeeded
     */
    bool DetectGridParams(const std::vector<Point3D>& coordinates, GridParams& grid_params);

    /**
     * @brief Validate grid regularity
     * @param coordinates Coordinate array
     * @param grid_params Grid parameters
     * @return Whether it is a regular grid
     */
    bool ValidateGridRegularity(const std::vector<Point3D>& coordinates, const GridParams& grid_params);

    /**
     * @brief Split string
     * @param line Input line
     * @param delimiter Delimiter
     * @return Array of split strings
     */
    std::vector<std::string> SplitString(const std::string& line, char delimiter);

    /**
     * @brief Convert string to numeric value
     * @param str Input string
     * @param value Output numeric value
     * @return Whether conversion succeeded
     */
    template <typename T>
    bool StringToValue(const std::string& str, T& value);

  private:
    char                   delimiter_;    // Delimiter
    bool                   skip_header_;  // Whether to skip header row
    Real                   tolerance_;    // Tolerance for floating point comparisons
    std::array<size_t, 3>  coord_cols_;   // Coordinate column indices
    std::array<size_t, 12> field_cols_;   // Magnetic field data column indices
};

}  // namespace p3d

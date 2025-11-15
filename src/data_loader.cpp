#include "point3d_interp/data_loader.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_set>

namespace p3d {

DataLoader::DataLoader()
    : delimiter_(','),
      skip_header_(true),
      coord_cols_{0, 1, 2},  // x, y, z
      field_cols_{3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
// Bx, By, Bz, dBx_dx, dBx_dy, dBx_dz, dBy_dx, dBy_dy, dBy_dz, dBz_dx, dBz_dy, dBz_dz
{}

DataLoader::~DataLoader() = default;

ErrorCode DataLoader::LoadFromCSV(const std::string& filepath, std::vector<Point3D>& coordinates,
                                  std::vector<MagneticFieldData>& field_data, GridParams& grid_params) {
    // Open file
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return ErrorCode::FileNotFound;
    }

    coordinates.clear();
    field_data.clear();

    std::string line;
    size_t      line_number = 0;

    // Skip header line
    if (skip_header_) {
        if (!std::getline(file, line)) {
            return ErrorCode::FileReadError;
        }
        line_number++;
    }

    // Read data lines
    while (std::getline(file, line)) {
        line_number++;

        // Skip empty lines
        if (line.empty()) {
            continue;
        }

        Point3D           point;
        MagneticFieldData field;

        if (!ParseLine(line, point, field)) {
            return ErrorCode::InvalidFileFormat;
        }

        coordinates.push_back(point);
        field_data.push_back(field);
    }

    if (coordinates.empty()) {
        return ErrorCode::InvalidFileFormat;
    }

    // Detect grid parameters
    if (!DetectGridParams(coordinates, grid_params)) {
        return ErrorCode::InvalidGridData;
    }

    // Validate grid regularity
    if (!ValidateGridRegularity(coordinates, grid_params)) {
        return ErrorCode::InvalidGridData;
    }

    return ErrorCode::Success;
}

void DataLoader::SetColumnIndices(const std::array<size_t, 3>& coord_cols, const std::array<size_t, 12>& field_cols) {
    coord_cols_ = coord_cols;
    field_cols_ = field_cols;
}

bool DataLoader::ParseLine(const std::string& line, Point3D& point, MagneticFieldData& field) {
    auto tokens = SplitString(line, delimiter_);

    // Check if there are enough columns
    size_t max_col = std::max({coord_cols_[0], coord_cols_[1], coord_cols_[2]});
    for (size_t col : field_cols_) {
        max_col = std::max(max_col, col);
    }

    if (tokens.size() <= max_col) {
        return false;
    }

    // Parse coordinates
    if (!StringToValue(tokens[coord_cols_[0]], point.x) || !StringToValue(tokens[coord_cols_[1]], point.y) ||
        !StringToValue(tokens[coord_cols_[2]], point.z)) {
        return false;
    }

    // Parse magnetic field data
    if (!StringToValue(tokens[field_cols_[0]], field.Bx) || !StringToValue(tokens[field_cols_[1]], field.By) ||
        !StringToValue(tokens[field_cols_[2]], field.Bz) || !StringToValue(tokens[field_cols_[3]], field.dBx_dx) ||
        !StringToValue(tokens[field_cols_[4]], field.dBx_dy) || !StringToValue(tokens[field_cols_[5]], field.dBx_dz) ||
        !StringToValue(tokens[field_cols_[6]], field.dBy_dx) || !StringToValue(tokens[field_cols_[7]], field.dBy_dy) ||
        !StringToValue(tokens[field_cols_[8]], field.dBy_dz) || !StringToValue(tokens[field_cols_[9]], field.dBz_dx) ||
        !StringToValue(tokens[field_cols_[10]], field.dBz_dy) ||
        !StringToValue(tokens[field_cols_[11]], field.dBz_dz)) {
        return false;
    }

    return true;
}

bool DataLoader::DetectGridParams(const std::vector<Point3D>& coordinates, GridParams& grid_params) {
    if (coordinates.empty()) {
        return false;
    }

    // Collect all x, y, z coordinates
    std::unordered_set<Real> x_coords, y_coords, z_coords;

    for (const auto& coord : coordinates) {
        x_coords.insert(coord.x);
        y_coords.insert(coord.y);
        z_coords.insert(coord.z);
    }

    // Convert to ordered vectors
    std::vector<Real> x_unique(x_coords.begin(), x_coords.end());
    std::vector<Real> y_unique(y_coords.begin(), y_coords.end());
    std::vector<Real> z_unique(z_coords.begin(), z_coords.end());

    std::sort(x_unique.begin(), x_unique.end());
    std::sort(y_unique.begin(), y_unique.end());
    std::sort(z_unique.begin(), z_unique.end());

    // Check if it's a regular grid
    if (x_unique.size() < 2 || y_unique.size() < 2 || z_unique.size() < 2) {
        return false;
    }

    // Calculate spacing (check if uniform)
    auto calculate_spacing = [](const std::vector<Real>& coords) -> Real {
        if (coords.size() < 2) return 0;
        Real spacing = coords[1] - coords[0];
        for (size_t i = 2; i < coords.size(); ++i) {
            Real current_spacing = coords[i] - coords[i - 1];
            if (std::abs(current_spacing - spacing) > 1e-6) {
                return 0;  // Non-uniform
            }
        }
        return spacing;
    };

    Real dx = calculate_spacing(x_unique);
    Real dy = calculate_spacing(y_unique);
    Real dz = calculate_spacing(z_unique);

    if (dx <= 0 || dy <= 0 || dz <= 0) {
        return false;
    }

    // Set grid parameters
    grid_params.origin     = Point3D(x_unique[0], y_unique[0], z_unique[0]);
    grid_params.spacing    = Point3D(dx, dy, dz);
    grid_params.dimensions = {static_cast<uint32_t>(x_unique.size()), static_cast<uint32_t>(y_unique.size()),
                              static_cast<uint32_t>(z_unique.size())};
    grid_params.update_bounds();

    return true;
}

bool DataLoader::ValidateGridRegularity(const std::vector<Point3D>& coordinates, const GridParams& grid_params) {
    // Calculate expected number of data points
    size_t expected_count =
        static_cast<size_t>(grid_params.dimensions[0] * grid_params.dimensions[1] * grid_params.dimensions[2]);

    if (coordinates.size() != expected_count) {
        return false;
    }

    // Validate each point is in the correct position
    for (const auto& coord : coordinates) {
        // Calculate grid indices
        int ix = static_cast<int>(std::round((coord.x - grid_params.origin.x) / grid_params.spacing.x));
        int iy = static_cast<int>(std::round((coord.y - grid_params.origin.y) / grid_params.spacing.y));
        int iz = static_cast<int>(std::round((coord.z - grid_params.origin.z) / grid_params.spacing.z));

        // Check if indices are valid
        if (ix < 0 || ix >= static_cast<int>(grid_params.dimensions[0]) || iy < 0 ||
            iy >= static_cast<int>(grid_params.dimensions[1]) || iz < 0 ||
            iz >= static_cast<int>(grid_params.dimensions[2])) {
            return false;
        }

        // Check if coordinates match exactly
        Real expected_x = grid_params.origin.x + ix * grid_params.spacing.x;
        Real expected_y = grid_params.origin.y + iy * grid_params.spacing.y;
        Real expected_z = grid_params.origin.z + iz * grid_params.spacing.z;

        if (std::abs(coord.x - expected_x) > 1e-6 || std::abs(coord.y - expected_y) > 1e-6 ||
            std::abs(coord.z - expected_z) > 1e-6) {
            return false;
        }
    }

    return true;
}

std::vector<std::string> DataLoader::SplitString(const std::string& line, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream        ss(line);
    std::string              token;

    while (std::getline(ss, token, delimiter)) {
        // Trim leading and trailing spaces
        token.erase(token.begin(),
                    std::find_if(token.begin(), token.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        token.erase(
            std::find_if(token.rbegin(), token.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(),
            token.end());

        tokens.push_back(token);
    }

    return tokens;
}

template <typename T>
bool DataLoader::StringToValue(const std::string& str, T& value) {
    std::istringstream iss(str);
    iss >> value;
    return !iss.fail() && iss.eof();
}

// Explicit template instantiations
template bool DataLoader::StringToValue<float>(const std::string& str, float& value);
template bool DataLoader::StringToValue<double>(const std::string& str, double& value);

}  // namespace p3d
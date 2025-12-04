#include "point3d_interp/data_loader.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_set>
#include <stdexcept>
#include <thread>
#include <future>
#include <cstring>
#include <charconv>

P3D_NAMESPACE_BEGIN

DataLoader::DataLoader()
    : delimiter_(','),
      skip_header_(true),
      tolerance_(1e-6),
      coord_cols_{0, 1, 2},  // x, y, z
      field_cols_{3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
// Bx, By, Bz, dBx_dx, dBx_dy, dBx_dz, dBy_dx, dBy_dy, dBy_dz, dBz_dx, dBz_dy, dBz_dz
{}

/**
 * @brief Fast string splitting using manual parsing
 * @param line Input line
 * @param delimiter Delimiter character
 * @return Vector of string tokens
 */
std::vector<std::string> FastSplitString(const std::string& line, char delimiter) {
    std::vector<std::string> tokens;
    size_t start = 0;
    size_t end = 0;

    while ((end = line.find(delimiter, start)) != std::string::npos) {
        std::string token = line.substr(start, end - start);
        // Trim whitespace
        token.erase(token.begin(), std::find_if(token.begin(), token.end(),
                    [](unsigned char ch) { return !std::isspace(ch); }));
        token.erase(std::find_if(token.rbegin(), token.rend(),
                    [](unsigned char ch) { return !std::isspace(ch); }).base(), token.end());
        tokens.push_back(token);
        start = end + 1;
    }

    // Last token
    std::string token = line.substr(start);
    token.erase(token.begin(), std::find_if(token.begin(), token.end(),
                [](unsigned char ch) { return !std::isspace(ch); }));
    token.erase(std::find_if(token.rbegin(), token.rend(),
                [](unsigned char ch) { return !std::isspace(ch); }).base(), token.end());
    tokens.push_back(token);

    return tokens;
}

/**
 * @brief Fast string to float conversion using strtof
 * @param str Input string
 * @param value Output value
 * @return True if conversion successful
 */
template <typename T>
bool FastStringToValue(const std::string& str, T& value) {
    if (str.empty()) return false;

    const char* start = str.c_str();
    char* end = nullptr;

    if constexpr (std::is_same_v<T, float>) {
        value = strtof(start, &end);
    } else if constexpr (std::is_same_v<T, double>) {
        value = strtod(start, &end);
    } else {
        // Fallback for other types
        std::istringstream iss(str);
        iss >> value;
        return !iss.fail() && iss.eof();
    }

    return end != start && *end == '\0';
}

DataLoader::~DataLoader() = default;

void DataLoader::LoadFromCSV(const std::string& filepath, std::vector<Point3D>& coordinates,
                             std::vector<MagneticFieldData>& field_data, GridParams& grid_params) {
    // Open file
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("File not found: " + filepath);
    }

    coordinates.clear();
    field_data.clear();

    // Estimate file size for memory pre-allocation
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Estimate number of lines (rough estimate: 100 bytes per line)
    size_t estimated_lines = file_size / 100;
    if (estimated_lines > 0) {
        coordinates.reserve(estimated_lines);
        field_data.reserve(estimated_lines);
    }

    // For large files, use parallel parsing
    static const size_t PARALLEL_THRESHOLD = 10000;  // Use parallel parsing for files with > 10K lines

    if (estimated_lines > PARALLEL_THRESHOLD) {
        LoadFromCSVParallel(file, coordinates, field_data, grid_params);
    } else {
        LoadFromCSVSequential(file, coordinates, field_data, grid_params);
    }
}

/**
 * @brief Sequential CSV loading for small files
 */
void DataLoader::LoadFromCSVSequential(std::ifstream& file, std::vector<Point3D>& coordinates,
                                       std::vector<MagneticFieldData>& field_data, GridParams& grid_params) {
    std::string line;
    size_t      line_number = 0;

    // Skip header line
    if (skip_header_) {
        if (!std::getline(file, line)) {
            throw std::runtime_error("File read error: unable to read header");
        }
        line_number++;
    }

    // Read data lines with optimized parsing
    while (std::getline(file, line)) {
        line_number++;

        // Skip empty lines
        if (line.empty()) {
            continue;
        }

        Point3D           point;
        MagneticFieldData field;

        if (!ParseLineFast(line, point, field)) {
            throw std::runtime_error("Invalid file format at line " + std::to_string(line_number));
        }

        coordinates.push_back(point);
        field_data.push_back(field);
    }

    if (coordinates.empty()) {
        throw std::runtime_error("Invalid file format: no data found");
    }

    // Shrink to fit to save memory
    coordinates.shrink_to_fit();
    field_data.shrink_to_fit();

    // Detect grid parameters
    if (!DetectGridParams(coordinates, grid_params)) {
        throw std::runtime_error("Invalid grid data: unable to detect grid parameters");
    }

    // Validate grid regularity
    if (!ValidateGridRegularity(coordinates, grid_params)) {
        throw std::runtime_error("Invalid grid data: grid is not regular");
    }
}

/**
 * @brief Parallel CSV loading for large files
 */
void DataLoader::LoadFromCSVParallel(std::ifstream& file, std::vector<Point3D>& coordinates,
                                     std::vector<MagneticFieldData>& field_data, GridParams& grid_params) {
    // Read all lines into memory first
    std::vector<std::string> lines;
    std::string line;
    size_t line_number = 0;

    // Skip header line
    if (skip_header_) {
        if (!std::getline(file, line)) {
            throw std::runtime_error("File read error: unable to read header");
        }
        line_number++;
    }

    // Read all data lines
    while (std::getline(file, line)) {
        if (!line.empty()) {
            lines.push_back(std::move(line));
        }
    }

    if (lines.empty()) {
        throw std::runtime_error("Invalid file format: no data found");
    }

    // Determine number of threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;  // Fallback
    if (num_threads > 8) num_threads = 8;   // Cap at 8 threads

    static const size_t MIN_CHUNK_SIZE = 1000;  // Minimum lines per thread
    size_t lines_per_thread = std::max(MIN_CHUNK_SIZE, lines.size() / num_threads);
    if (lines_per_thread * num_threads < lines.size()) {
        lines_per_thread = lines.size() / num_threads + 1;
    }

    // Launch parallel parsing
    std::vector<std::future<std::pair<std::vector<Point3D>, std::vector<MagneticFieldData>>>> futures;

    for (size_t i = 0; i < num_threads; ++i) {
        size_t start_idx = i * lines_per_thread;
        size_t end_idx = std::min(start_idx + lines_per_thread, lines.size());

        if (start_idx >= lines.size()) break;

        futures.push_back(std::async(std::launch::async, [this, start_idx, end_idx, &lines]() {
            std::vector<Point3D> local_coords;
            std::vector<MagneticFieldData> local_fields;
            local_coords.reserve(end_idx - start_idx);
            local_fields.reserve(end_idx - start_idx);

            for (size_t j = start_idx; j < end_idx; ++j) {
                Point3D point;
                MagneticFieldData field;
                if (ParseLineFast(lines[j], point, field)) {
                    local_coords.push_back(point);
                    local_fields.push_back(field);
                }
            }

            return std::make_pair(std::move(local_coords), std::move(local_fields));
        }));
    }

    // Collect results
    for (auto& future : futures) {
        auto [local_coords, local_fields] = future.get();
        coordinates.insert(coordinates.end(), local_coords.begin(), local_coords.end());
        field_data.insert(field_data.end(), local_fields.begin(), local_fields.end());
    }

    // Shrink to fit to save memory
    coordinates.shrink_to_fit();
    field_data.shrink_to_fit();

    // Detect grid parameters
    if (!DetectGridParams(coordinates, grid_params)) {
        throw std::runtime_error("Invalid grid data: unable to detect grid parameters");
    }

    // Validate grid regularity
    if (!ValidateGridRegularity(coordinates, grid_params)) {
        throw std::runtime_error("Invalid grid data: grid is not regular");
    }
}

void DataLoader::LoadFromBinary(const std::string& filepath, std::vector<Point3D>& coordinates,
                                std::vector<MagneticFieldData>& field_data, GridParams& grid_params) {
    // Open file in binary mode
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("File not found: " + filepath);
    }

    coordinates.clear();
    field_data.clear();

    // Read and validate header
    const uint32_t EXPECTED_MAGIC = 0x50494441;  // "PIDA"
    const uint32_t EXPECTED_VERSION = 1;

    uint32_t magic_number, format_version, num_points;
    bool is_input_data;

    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    if (magic_number != EXPECTED_MAGIC) {
        throw std::runtime_error("Invalid binary file format: wrong magic number");
    }

    file.read(reinterpret_cast<char*>(&format_version), sizeof(format_version));
    if (format_version != EXPECTED_VERSION) {
        throw std::runtime_error("Unsupported binary format version: " + std::to_string(format_version));
    }

    file.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));
    file.read(reinterpret_cast<char*>(&is_input_data), sizeof(is_input_data));

    if (!is_input_data) {
        throw std::runtime_error("Binary file contains output data, not input data");
    }

    if (num_points == 0) {
        throw std::runtime_error("Binary file contains no data points");
    }

    // Pre-allocate memory
    coordinates.resize(num_points);
    field_data.resize(num_points);

    // Read data in one go for maximum performance
    file.read(reinterpret_cast<char*>(coordinates.data()), num_points * sizeof(Point3D));
    if (file.gcount() != static_cast<std::streamsize>(num_points * sizeof(Point3D))) {
        throw std::runtime_error("Failed to read coordinate data from binary file");
    }

    file.read(reinterpret_cast<char*>(field_data.data()), num_points * sizeof(MagneticFieldData));
    if (file.gcount() != static_cast<std::streamsize>(num_points * sizeof(MagneticFieldData))) {
        throw std::runtime_error("Failed to read field data from binary file");
    }

    // Detect grid parameters
    if (!DetectGridParams(coordinates, grid_params)) {
        throw std::runtime_error("Invalid grid data: unable to detect grid parameters");
    }

    // Validate grid regularity
    if (!ValidateGridRegularity(coordinates, grid_params)) {
        throw std::runtime_error("Invalid grid data: grid is not regular");
    }
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

/**
 * @brief Fast line parsing using optimized string operations
 * @param line Input line
 * @param point Output coordinate point
 * @param field Output magnetic field data
 * @return Whether parsing succeeded
 */
bool DataLoader::ParseLineFast(const std::string& line, Point3D& point, MagneticFieldData& field) {
    auto tokens = FastSplitString(line, delimiter_);

    // Check if there are enough columns
    size_t max_col = std::max({coord_cols_[0], coord_cols_[1], coord_cols_[2]});
    for (size_t col : field_cols_) {
        max_col = std::max(max_col, col);
    }

    if (tokens.size() <= max_col) {
        return false;
    }

    // Parse coordinates using fast conversion
    if (!FastStringToValue(tokens[coord_cols_[0]], point.x) || !FastStringToValue(tokens[coord_cols_[1]], point.y) ||
        !FastStringToValue(tokens[coord_cols_[2]], point.z)) {
        return false;
    }

    // Parse magnetic field data using fast conversion
    if (!FastStringToValue(tokens[field_cols_[0]], field.Bx) || !FastStringToValue(tokens[field_cols_[1]], field.By) ||
        !FastStringToValue(tokens[field_cols_[2]], field.Bz) || !FastStringToValue(tokens[field_cols_[3]], field.dBx_dx) ||
        !FastStringToValue(tokens[field_cols_[4]], field.dBx_dy) || !FastStringToValue(tokens[field_cols_[5]], field.dBx_dz) ||
        !FastStringToValue(tokens[field_cols_[6]], field.dBy_dx) || !FastStringToValue(tokens[field_cols_[7]], field.dBy_dy) ||
        !FastStringToValue(tokens[field_cols_[8]], field.dBy_dz) || !FastStringToValue(tokens[field_cols_[9]], field.dBz_dx) ||
        !FastStringToValue(tokens[field_cols_[10]], field.dBz_dy) ||
        !FastStringToValue(tokens[field_cols_[11]], field.dBz_dz)) {
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
    auto calculate_spacing = [this](const std::vector<Real>& coords) -> Real {
        if (coords.size() < 2) return 0;
        Real spacing = coords[1] - coords[0];
        for (size_t i = 2; i < coords.size(); ++i) {
            Real current_spacing = coords[i] - coords[i - 1];
            if (std::abs(current_spacing - spacing) > this->tolerance_) {
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

        if (std::abs(coord.x - expected_x) > this->tolerance_ || std::abs(coord.y - expected_y) > this->tolerance_ ||
            std::abs(coord.z - expected_z) > this->tolerance_) {
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

P3D_NAMESPACE_END
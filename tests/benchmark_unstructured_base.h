#pragma once

#include "benchmark_structured_base.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <random>
#include <array>
#include <filesystem>

namespace p3d {

/**
 * @brief Base class for unstructured data performance benchmarks
 *
 * Provides common functionality for benchmarking different data scales
 * and query sizes for unstructured (scattered) point cloud data using IDW interpolation.
 */
class UnstructuredBenchmarkBase : public BenchmarkBase {
  public:
    UnstructuredBenchmarkBase() = default;

    virtual ~UnstructuredBenchmarkBase() = default;

    /**
     * @brief Run all benchmarks for this data scale
     */
    void RunAllBenchmarks() {
        std::cout
            << "=== 3D Magnetic Field Data Interpolation Library Unstructured Data Performance Benchmarks ===\n\n";

        // Get data point count from derived class
        size_t data_count = GetDataPointCount();

        // Test different numbers of query points
        std::vector<size_t> query_sizes = {100, 1000, 10000};

        std::cout << "Test data scale: " << data_count << " scattered points\n";
        std::cout << std::string(60, '-') << "\n";

        // Create test data
        auto test_data = GenerateTestData(data_count);

        for (size_t query_size : query_sizes) {
            std::cout << "=== Testing with " << query_size << " query points ===" << std::endl;

            // Generate query points
            auto query_points = GenerateQueryPoints(query_size, test_data.grid_params);

            // CPU benchmark
            auto   cpu_results = BenchmarkCPU(test_data, query_points);
            double cpu_time    = cpu_results.first;

            // GPU benchmark
            auto   gpu_results = BenchmarkGPU(test_data, query_points);
            double gpu_time    = gpu_results.first;

            // Print CPU results
            std::cout << "[CPU, Unstructured, QuerySize=" << query_size << "] Time: " << std::fixed
                      << std::setprecision(3) << cpu_time << " ms, Throughput: " << std::fixed << std::setprecision(0)
                      << (query_size / (cpu_time / 1000.0)) << " q/s" << std::endl;

            if (gpu_time > 0) {
                double speedup = cpu_time / gpu_time;
                std::cout << "[GPU, Unstructured, QuerySize=" << query_size << "] Time: " << std::fixed
                          << std::setprecision(3) << gpu_time << " ms, Throughput: " << std::fixed
                          << std::setprecision(0) << (query_size / (gpu_time / 1000.0))
                          << " q/s, Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
            } else {
                std::cout << "[CPU, Unstructured, QuerySize=" << query_size << "] Time: " << std::fixed
                          << std::setprecision(3) << cpu_time << " ms, Throughput: " << std::fixed
                          << std::setprecision(0) << (query_size / (cpu_time / 1000.0)) << " q/s (GPU not available)"
                          << std::endl;
            }

            // Export VTK files for visualization
            ExportBenchmarkResults(test_data, query_points, cpu_results.second, gpu_results.second, {0, 0, 0},
                                   query_size);

            std::cout << std::endl;
        }
    }

  protected:
    /**
     * @brief Get the data dimensions for this benchmark (not used for unstructured)
     * @return Array of {width, height, depth} - returns {point_count, 1, 1}
     */
    std::array<size_t, 3> GetDataDimensions() const override { return {GetDataPointCount(), 1, 1}; }

    /**
     * @brief Get the number of data points for this benchmark
     * @return Number of scattered data points
     */
    virtual size_t GetDataPointCount() const = 0;

    /**
     * @brief Generate query points for benchmarking
     * @param count Number of points to generate
     * @param grid_params Grid parameters defining the domain (min/max bounds)
     * @return Vector of query points
     */
    virtual std::vector<Point3D> GenerateQueryPoints(size_t count, const GridParams& grid_params) override {
        std::vector<Point3D> points;
        points.reserve(count);

        std::uniform_real_distribution<float> dist_x(grid_params.min_bound.x, grid_params.max_bound.x);
        std::uniform_real_distribution<float> dist_y(grid_params.min_bound.y, grid_params.max_bound.y);
        std::uniform_real_distribution<float> dist_z(grid_params.min_bound.z, grid_params.max_bound.z);

        for (size_t i = 0; i < count; ++i) {
            points.push_back(Point3D(dist_x(rng_), dist_y(rng_), dist_z(rng_)));
        }

        return points;
    }

  private:
    struct TestData {
        std::vector<Point3D>           coordinates;
        std::vector<MagneticFieldData> field_data;
        GridParams                     grid_params;
    };

    void ExportBenchmarkResults(const TestData& test_data, const std::vector<Point3D>& query_points,
                                const std::vector<InterpolationResult>& cpu_results,
                                const std::vector<InterpolationResult>& gpu_results,
                                const std::array<size_t, 3>& /*data_size*/, size_t query_size) {
        // Create output directory
        std::filesystem::create_directories("benchmark_output");

        std::string type_suffix = GetBenchmarkType();
        std::string count_str   = std::to_string(GetDataPointCount());

        // Export input data points
        {
            std::string filename = "benchmark_output/input_" + count_str + type_suffix + ".vtk";
            MagneticFieldInterpolator::ExportInputPoints(test_data.coordinates, test_data.field_data,
                                                         ExportFormat::ParaviewVTK, filename);
        }

        // Export CPU results
        if (!cpu_results.empty()) {
            std::string filename =
                "benchmark_output/cpu_" + count_str + type_suffix + "_q" + std::to_string(query_size) + ".vtk";

            MagneticFieldInterpolator::ExportOutputPoints(ExportFormat::ParaviewVTK, query_points, cpu_results,
                                                          filename);
        }

        // Export GPU results
        if (!gpu_results.empty()) {
            std::string filename =
                "benchmark_output/gpu_" + count_str + type_suffix + "_q" + std::to_string(query_size) + ".vtk";
            MagneticFieldInterpolator::ExportOutputPoints(ExportFormat::ParaviewVTK, query_points, gpu_results,
                                                          filename);
        }
    }

    TestData GenerateTestData(size_t point_count) {
        TestData data;

        data.coordinates.reserve(point_count);
        data.field_data.reserve(point_count);

        // Set up domain bounds (similar to structured benchmarks)
        data.grid_params.origin     = Point3D(0.0f, 0.0f, 0.0f);
        data.grid_params.spacing    = Point3D(1.0f, 1.0f, 1.0f);
        data.grid_params.dimensions = {static_cast<uint32_t>(point_count), 1, 1};  // Not used for unstructured
        data.grid_params.update_bounds();

        // Generate random points within a reasonable domain
        std::uniform_real_distribution<float> pos_dist(-10.0f, 10.0f);
        std::uniform_real_distribution<float> field_dist(-1.0f, 1.0f);

        for (size_t i = 0; i < point_count; ++i) {
            Point3D coord(pos_dist(rng_), pos_dist(rng_), pos_dist(rng_));
            data.coordinates.push_back(coord);

            // Generate magnetic field data (similar pattern to structured)
            MagneticFieldData field(coord.x, coord.y, coord.z,  // Bx = x, By = y, Bz = z
                                    1.0f, 0.0f, 0.0f,           // dBx_dx=1, dBx_dy=0, dBx_dz=0
                                    0.0f, 1.0f, 0.0f,           // dBy_dx=0, dBy_dy=1, dBy_dz=0
                                    0.0f, 0.0f, 1.0f);          // dBz_dx=0, dBz_dy=0, dBz_dz=1
            data.field_data.push_back(field);
        }

        // Update bounds based on actual data
        if (!data.coordinates.empty()) {
            Point3D min_bound = data.coordinates[0];
            Point3D max_bound = data.coordinates[0];
            for (const auto& p : data.coordinates) {
                min_bound.x = std::min(min_bound.x, p.x);
                min_bound.y = std::min(min_bound.y, p.y);
                min_bound.z = std::min(min_bound.z, p.z);
                max_bound.x = std::max(max_bound.x, p.x);
                max_bound.y = std::max(max_bound.y, p.y);
                max_bound.z = std::max(max_bound.z, p.z);
            }
            data.grid_params.min_bound = min_bound;
            data.grid_params.max_bound = max_bound;
        }

        return data;
    }

    std::pair<double, std::vector<InterpolationResult>> BenchmarkCPU(const TestData&             test_data,
                                                                     const std::vector<Point3D>& query_points) {
        MagneticFieldInterpolator interp(false, 0, InterpolationMethod::IDW,
                                         ExtrapolationMethod::LinearExtrapolation);  // CPU mode with IDW interpolation

        // Load data
        ErrorCode err = interp.LoadFromMemory(test_data.coordinates.data(), test_data.field_data.data(),
                                              test_data.coordinates.size());
        if (err != ErrorCode::Success) {
            std::cerr << "CPU data loading failed: " << static_cast<int>(err) << std::endl;
            return {-1.0, {}};
        }

        // Warm up
        InterpolationResult dummy;
        interp.Query(query_points[0], dummy);

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<InterpolationResult> results(query_points.size());
        err = interp.QueryBatch(query_points.data(), results.data(), query_points.size());

        auto end = std::chrono::high_resolution_clock::now();

        if (err != ErrorCode::Success) {
            std::cerr << "CPU query failed: " << static_cast<int>(err) << std::endl;
            return {-1.0, {}};
        }

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return {duration.count() / 1000.0, results};  // Convert to milliseconds
    }

    std::pair<double, std::vector<InterpolationResult>> BenchmarkGPU(const TestData&             test_data,
                                                                     const std::vector<Point3D>& query_points) {
        MagneticFieldInterpolator interp(true, 0, InterpolationMethod::IDW,
                                         ExtrapolationMethod::LinearExtrapolation);  // GPU mode with IDW interpolation

        // Load data
        ErrorCode err = interp.LoadFromMemory(test_data.coordinates.data(), test_data.field_data.data(),
                                              test_data.coordinates.size());
        if (err != ErrorCode::Success) {
            return {-1.0, {}};  // GPU unavailable or initialization failed
        }

        // Warm up
        InterpolationResult dummy;
        interp.Query(query_points[0], dummy);

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<InterpolationResult> results(query_points.size());
        err = interp.QueryBatch(query_points.data(), results.data(), query_points.size());

        auto end = std::chrono::high_resolution_clock::now();

        if (err != ErrorCode::Success) {
            return {-1.0, {}};
        }

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return {duration.count() / 1000.0, results};  // Convert to milliseconds
    }
};

}  // namespace p3d

#pragma once

#include "point3d_interp/interpolator_api.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <random>
#include <array>
#include <filesystem>

namespace p3d {

/**
 * @brief Base class for performance benchmarks
 *
 * Provides common functionality for benchmarking different data scales
 * and query sizes. Eliminates code duplication across benchmark files.
 */
class BenchmarkBase {
  public:
    BenchmarkBase() : rng_(std::random_device{}()) {}

    virtual ~BenchmarkBase() = default;

    /**
     * @brief Run all benchmarks for this data scale
     */
    void RunAllBenchmarks() {
        std::cout << "=== 3D Magnetic Field Data Interpolation Library Performance Benchmarks ===\n\n";

        // Get data dimensions from derived class
        auto data_size = GetDataDimensions();

        // Test different numbers of query points
        std::vector<size_t> query_sizes = {100, 1000, 10000};

        std::cout << "Test data scale: " << data_size[0] << "x" << data_size[1] << "x" << data_size[2] << " ("
                  << (data_size[0] * data_size[1] * data_size[2]) << " points)\n";
        std::cout << std::string(60, '-') << "\n";

        // Create test data
        auto test_data = GenerateTestData(data_size);

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
            std::cout << "[CPU, Structured, QuerySize=" << query_size << "] Time: " << std::fixed
                      << std::setprecision(3) << cpu_time << " ms, Throughput: " << std::fixed << std::setprecision(0)
                      << (query_size / (cpu_time / 1000.0)) << " q/s" << std::endl;

            if (gpu_time > 0) {
                double speedup = cpu_time / gpu_time;
                std::cout << "[GPU, Structured, QuerySize=" << query_size << "] Time: " << std::fixed
                          << std::setprecision(3) << gpu_time << " ms, Throughput: " << std::fixed
                          << std::setprecision(0) << (query_size / (gpu_time / 1000.0))
                          << " q/s, Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
            } else {
                std::cout << "[CPU, Structured, QuerySize=" << query_size << "] Time: " << std::fixed
                          << std::setprecision(3) << cpu_time << " ms, Throughput: " << std::fixed
                          << std::setprecision(0) << (query_size / (cpu_time / 1000.0)) << " q/s (GPU not available)"
                          << std::endl;
            }

            // Export VTK files for visualization
            ExportBenchmarkResults(test_data, query_points, cpu_results.second, gpu_results.second, data_size,
                                   query_size);

            std::cout << std::endl;
        }
    }

  protected:
    /**
     * @brief Get the data dimensions for this benchmark
     * @return Array of {width, height, depth}
     */
    virtual std::array<size_t, 3> GetDataDimensions() const = 0;

    /**
     * @brief Get the benchmark type suffix for file naming
     * @return String suffix to append to filenames (e.g., "_out_of_domain")
     */
    virtual std::string GetBenchmarkType() const { return ""; }

    std::mt19937 rng_;

    /**
     * @brief Generate query points for benchmarking
     * @param count Number of points to generate
     * @param grid_params Grid parameters defining the domain
     * @return Vector of query points
     */
    virtual std::vector<Point3D> GenerateQueryPoints(size_t count, const GridParams& grid_params) {
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
                                const std::array<size_t, 3>& data_size, size_t query_size) {
        // Create output directory
        std::filesystem::create_directories("benchmark_output");

        std::string type_suffix = GetBenchmarkType();

        // Export input data points
        {
            std::string filename = "benchmark_output/input_" + std::to_string(data_size[0]) + "x" +
                                   std::to_string(data_size[1]) + "x" + std::to_string(data_size[2]) + type_suffix +
                                   ".vtk";
            MagneticFieldInterpolator::ExportInputPoints(test_data.coordinates, test_data.field_data,
                                                         ExportFormat::ParaviewVTK, filename);
        }

        // Export CPU results
        if (!cpu_results.empty()) {
            std::string filename = "benchmark_output/cpu_" + std::to_string(data_size[0]) + "x" +
                                   std::to_string(data_size[1]) + "x" + std::to_string(data_size[2]) + type_suffix +
                                   "_q" + std::to_string(query_size) + ".vtk";
            MagneticFieldInterpolator::ExportOutputPoints(ExportFormat::ParaviewVTK, query_points, cpu_results,
                                                          filename);
        }

        // Export GPU results
        if (!gpu_results.empty()) {
            std::string filename = "benchmark_output/gpu_" + std::to_string(data_size[0]) + "x" +
                                   std::to_string(data_size[1]) + "x" + std::to_string(data_size[2]) + type_suffix +
                                   "_q" + std::to_string(query_size) + ".vtk";
            MagneticFieldInterpolator::ExportOutputPoints(ExportFormat::ParaviewVTK, query_points, gpu_results,
                                                          filename);
        }
    }

    TestData GenerateTestData(const std::array<size_t, 3>& dimensions) {
        TestData data;

        data.grid_params.origin     = Point3D(0.0f, 0.0f, 0.0f);
        data.grid_params.spacing    = Point3D(1.0f, 1.0f, 1.0f);
        data.grid_params.dimensions = {static_cast<uint32_t>(dimensions[0]), static_cast<uint32_t>(dimensions[1]),
                                       static_cast<uint32_t>(dimensions[2])};
        data.grid_params.update_bounds();

        size_t total_points = dimensions[0] * dimensions[1] * dimensions[2];
        data.coordinates.reserve(total_points);
        data.field_data.reserve(total_points);

        // Generate magnetic field data for each grid point
        for (size_t k = 0; k < dimensions[2]; ++k) {
            for (size_t j = 0; j < dimensions[1]; ++j) {
                for (size_t i = 0; i < dimensions[0]; ++i) {
                    Point3D coord(data.grid_params.origin.x + i * data.grid_params.spacing.x,
                                  data.grid_params.origin.y + j * data.grid_params.spacing.y,
                                  data.grid_params.origin.z + k * data.grid_params.spacing.z);
                    data.coordinates.push_back(coord);

                    // Generate magnetic field data for each grid point
                    MagneticFieldData field(coord.x, coord.y, coord.z,  // Bx = x, By = y, Bz = z
                                            1.0f, 0.0f, 0.0f,           // dBx_dx=1, dBx_dy=0, dBx_dz=0
                                            0.0f, 1.0f, 0.0f,           // dBy_dx=0, dBy_dy=1, dBy_dz=0
                                            0.0f, 0.0f, 1.0f);          // dBz_dx=0, dBz_dy=0, dBz_dz=1
                    data.field_data.push_back(field);
                }
            }
        }

        return data;
    }

    std::pair<double, std::vector<InterpolationResult>> BenchmarkCPU(const TestData&             test_data,
                                                                     const std::vector<Point3D>& query_points) {
        MagneticFieldInterpolator interp(
            false, 0, InterpolationMethod::TricubicHermite,
            ExtrapolationMethod::LinearExtrapolation);  // CPU mode with linear extrapolation

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
        MagneticFieldInterpolator interp(
            true, 0, InterpolationMethod::TricubicHermite,
            ExtrapolationMethod::LinearExtrapolation);  // GPU mode with linear extrapolation

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

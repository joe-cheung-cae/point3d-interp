#include "point3d_interp/data_loader.h"
#include "point3d_interp/interpolator_exporter.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <random>
#include <fstream>
#include <cstdio>

using namespace p3d;

/**
 * @brief Data loading performance benchmark program
 */
class DataLoadingBenchmark {
  public:
    DataLoadingBenchmark() : rng_(std::random_device{}()) {}

    void RunBenchmark() {
        std::cout << "=== Data Loading Performance Benchmark ===\n\n";

        // Test different data sizes
        std::vector<std::array<size_t, 3>> sizes = {
            {10, 10, 10},  // ~1K points
            {20, 20, 20},  // ~8K points
            {30, 30, 30}   // ~27K points
        };

        for (const auto& size : sizes) {
            size_t total_points = size[0] * size[1] * size[2];
            std::cout << "Testing with " << total_points << " data points (" << size[0] << "x" << size[1] << "x"
                      << size[2] << ")\n";

            // Generate test data
            auto test_data = GenerateTestData(size);

            // Save to CSV and binary formats
            std::string csv_file = "benchmark_test.csv";
            std::string bin_file = "benchmark_test.bin";

            SaveTestDataToCSV(test_data, csv_file);
            SaveTestDataToBinary(test_data, bin_file);

            // Benchmark CSV loading
            double csv_time = BenchmarkCSVLoading(csv_file, total_points);

            // Benchmark binary loading
            double bin_time = BenchmarkBinaryLoading(bin_file, total_points);

            // Calculate speedup
            double speedup = csv_time / bin_time;

            std::cout << std::fixed << std::setprecision(3);
            std::cout << "  CSV loading:    " << csv_time << " ms\n";
            std::cout << "  Binary loading: " << bin_time << " ms\n";
            std::cout << "  Speedup:        " << speedup << "x\n\n";

            // Cleanup
            std::remove(csv_file.c_str());
            std::remove(bin_file.c_str());
        }
    }

  private:
    struct TestData {
        std::vector<Point3D>           coordinates;
        std::vector<MagneticFieldData> field_data;
        GridParams                     grid_params;
    };

    std::mt19937 rng_;

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

        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (size_t k = 0; k < dimensions[2]; ++k) {
            for (size_t j = 0; j < dimensions[1]; ++j) {
                for (size_t i = 0; i < dimensions[0]; ++i) {
                    Point3D coord(data.grid_params.origin.x + i * data.grid_params.spacing.x,
                                  data.grid_params.origin.y + j * data.grid_params.spacing.y,
                                  data.grid_params.origin.z + k * data.grid_params.spacing.z);
                    data.coordinates.push_back(coord);

                    MagneticFieldData field(dist(rng_), dist(rng_), dist(rng_), dist(rng_), dist(rng_), dist(rng_),
                                            dist(rng_), dist(rng_), dist(rng_), dist(rng_), dist(rng_), dist(rng_));
                    data.field_data.push_back(field);
                }
            }
        }

        return data;
    }

    void SaveTestDataToCSV(const TestData& data, const std::string& filename) {
        std::ofstream file(filename);
        file << "x,y,z,Bx,By,Bz,dBx_dx,dBx_dy,dBx_dz,dBy_dx,dBy_dy,dBy_dz,dBz_dx,dBz_dy,dBz_dz\n";

        for (size_t i = 0; i < data.coordinates.size(); ++i) {
            const auto& coord = data.coordinates[i];
            const auto& field = data.field_data[i];
            file << std::fixed << std::setprecision(6) << coord.x << "," << coord.y << "," << coord.z << "," << field.Bx
                 << "," << field.By << "," << field.Bz << "," << field.dBx_dx << "," << field.dBx_dy << ","
                 << field.dBx_dz << "," << field.dBy_dx << "," << field.dBy_dy << "," << field.dBy_dz << ","
                 << field.dBz_dx << "," << field.dBz_dy << "," << field.dBz_dz << "\n";
        }
    }

    void SaveTestDataToBinary(const TestData& data, const std::string& filename) {
        auto exporter = CreateExporter(ExportFormat::BinaryData);
        exporter->ExportInputPoints(data.coordinates, data.field_data, filename);
    }

    double BenchmarkCSVLoading(const std::string& filename, size_t expected_points) {
        std::vector<Point3D>           coordinates;
        std::vector<MagneticFieldData> field_data;
        GridParams                     grid_params;

        DataLoader loader;

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < 5; ++i) {  // Average over 5 runs
            coordinates.clear();
            field_data.clear();
            loader.LoadFromCSV(filename, coordinates, field_data, grid_params);
        }

        auto end      = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        return duration.count() / 5000.0 / 1000.0;  // Average in milliseconds
    }

    double BenchmarkBinaryLoading(const std::string& filename, size_t expected_points) {
        std::vector<Point3D>           coordinates;
        std::vector<MagneticFieldData> field_data;
        GridParams                     grid_params;

        DataLoader loader;

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < 5; ++i) {  // Average over 5 runs
            coordinates.clear();
            field_data.clear();
            loader.LoadFromBinary(filename, coordinates, field_data, grid_params);
        }

        auto end      = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        return duration.count() / 5000.0 / 1000.0;  // Average in milliseconds
    }
};

int main() {
    DataLoadingBenchmark benchmark;
    benchmark.RunBenchmark();
    return 0;
}
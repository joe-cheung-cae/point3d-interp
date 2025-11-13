#include "point3d_interp/api.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <random>

using namespace p3d;

/**
 * @brief Performance benchmark program
 */
class Benchmark {
public:
    Benchmark() : rng_(std::random_device{}()) {}

    void RunAllBenchmarks() {
        std::cout << "=== 3D Magnetic Field Data Interpolation Library Performance Benchmarks ===\n\n";

        // Test different scale datasets
        std::vector<std::array<size_t, 3>> data_sizes = {
            {10, 10, 10},     // 1,000 points
            {20, 20, 20},     // 8,000 points
            {30, 30, 30},     // 27,000 points
            {50, 50, 50}      // 125,000 points
        };

        // Test different numbers of query points
        std::vector<size_t> query_sizes = {100, 1000, 10000};

        for (const auto& data_size : data_sizes) {
            std::cout << "Test data scale: " << data_size[0] << "x" << data_size[1] << "x" << data_size[2]
                      << " (" << (data_size[0] * data_size[1] * data_size[2]) << " points)\n";
            std::cout << std::string(60, '-') << "\n";

            // Create test data
            auto test_data = GenerateTestData(data_size);

            for (size_t query_size : query_sizes) {
                std::cout << "Number of query points: " << query_size << "\n";

                // Generate query points
                auto query_points = GenerateQueryPoints(query_size, test_data.grid_params);

                // CPU benchmark
                double cpu_time = BenchmarkCPU(test_data, query_points);
                std::cout << "  CPU time: " << std::fixed << std::setprecision(3) << cpu_time << " ms";

                // GPU benchmark
                double gpu_time = BenchmarkGPU(test_data, query_points);
                if (gpu_time > 0) {
                    double speedup = cpu_time / gpu_time;
                    std::cout << "  GPU time: " << std::fixed << std::setprecision(3) << gpu_time << " ms";
                    std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x";
                } else {
                    std::cout << "  GPU: N/A (no GPU or initialization failed)";
                }

                // Calculate throughput
                double throughput = query_size / (cpu_time / 1000.0);
                std::cout << "  Throughput: " << std::fixed << std::setprecision(0) << throughput << " queries/second\n";

                std::cout << "\n";
            }
            std::cout << "\n";
        }

        // Memory usage test
        RunMemoryBenchmark();
    }

private:
    struct TestData {
        std::vector<Point3D> coordinates;
        std::vector<MagneticFieldData> field_data;
        GridParams grid_params;
    };

    std::mt19937 rng_;

    TestData GenerateTestData(const std::array<size_t, 3>& dimensions) {
        TestData data;

        data.grid_params.origin = Point3D(0.0f, 0.0f, 0.0f);
        data.grid_params.spacing = Point3D(1.0f, 1.0f, 1.0f);
        data.grid_params.dimensions = {static_cast<uint32_t>(dimensions[0]),
                                     static_cast<uint32_t>(dimensions[1]),
                                     static_cast<uint32_t>(dimensions[2])};
        data.grid_params.update_bounds();

        size_t total_points = dimensions[0] * dimensions[1] * dimensions[2];
        data.coordinates.reserve(total_points);
        data.field_data.reserve(total_points);

        // Geenerate magnetic field data for each grid point
        for (size_t k = 0; k < dimensions[2]; ++k) {
            for (size_t j = 0; j < dimensions[1]; ++j) {
                for (size_t i = 0; i < dimensions[0]; ++i) {
                    Point3D coord(
                        data.grid_params.origin.x + i * data.grid_params.spacing.x,
                        data.grid_params.origin.y + j * data.grid_params.spacing.y,
                        data.grid_params.origin.z + k * data.grid_params.spacing.z
                    );
                    data.coordinates.push_back(coord);

                    // Generate magnetic field data for each grid point
                    MagneticFieldData field(
                        coord.x + coord.y + coord.z + 1.0f,  // B = x + y + z + 1
                        1.0f, 1.0f, 1.0f                    // 常数梯度
                    );
                    data.field_data.push_back(field);
                }
            }
        }

        return data;
    }

    std::vector<Point3D> GenerateQueryPoints(size_t count, const GridParams& grid_params) {
        std::vector<Point3D> points;
        points.reserve(count);

        std::uniform_real_distribution<float> dist_x(grid_params.min_bound.x, grid_params.max_bound.x);
        std::uniform_real_distribution<float> dist_y(grid_params.min_bound.y, grid_params.max_bound.y);
        std::uniform_real_distribution<float> dist_z(grid_params.min_bound.z, grid_params.max_bound.z);

        for (size_t i = 0; i < count; ++i) {
            points.push_back(Point3D(
                dist_x(rng_), dist_y(rng_), dist_z(rng_)
            ));
        }

        return points;
    }

    double BenchmarkCPU(const TestData& test_data, const std::vector<Point3D>& query_points) {
        MagneticFieldInterpolator interp(false);  // CPU模式

        // 加载数据
        ErrorCode err = interp.LoadFromMemory(
            test_data.coordinates.data(),
            test_data.field_data.data(),
            test_data.coordinates.size()
        );
        if (err != ErrorCode::Success) {
            std::cerr << "CPU数据加载失败: " << static_cast<int>(err) << std::endl;
            return -1.0;
        }

        // 预热
        InterpolationResult dummy;
        interp.Query(query_points[0], dummy);

        // 基准测试
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<InterpolationResult> results(query_points.size());
        err = interp.QueryBatch(query_points.data(), results.data(), query_points.size());

        auto end = std::chrono::high_resolution_clock::now();

        if (err != ErrorCode::Success) {
            std::cerr << "CPU查询失败: " << static_cast<int>(err) << std::endl;
            return -1.0;
        }

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return duration.count() / 1000.0;  // 转换为毫秒
    }

    double BenchmarkGPU(const TestData& test_data, const std::vector<Point3D>& query_points) {
        MagneticFieldInterpolator interp(true);  // GPU模式

        // 加载数据
        ErrorCode err = interp.LoadFromMemory(
            test_data.coordinates.data(),
            test_data.field_data.data(),
            test_data.coordinates.size()
        );
        if (err != ErrorCode::Success) {
            return -1.0;  // GPU不可用或初始化失败
        }

        // 预热
        InterpolationResult dummy;
        interp.Query(query_points[0], dummy);

        // 基准测试
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<InterpolationResult> results(query_points.size());
        err = interp.QueryBatch(query_points.data(), results.data(), query_points.size());

        auto end = std::chrono::high_resolution_clock::now();

        if (err != ErrorCode::Success) {
            return -1.0;
        }

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return duration.count() / 1000.0;  // 转换为毫秒
    }

    void RunMemoryBenchmark() {
        std::cout << "=== 内存使用基准测试 ===\n\n";

        std::vector<std::array<size_t, 3>> sizes = {
            {10, 10, 10},    // ~0.4MB
            {50, 50, 50},    // ~50MB
            {100, 100, 100}  // ~800MB (可能超出GPU内存)
        };

        for (const auto& size : sizes) {
            size_t total_points = size[0] * size[1] * size[2];
            size_t memory_mb = total_points * sizeof(MagneticFieldData) / (1024 * 1024);

            std::cout << "数据规模: " << size[0] << "x" << size[1] << "x" << size[2]
                      << " (" << total_points << " 点, ~" << memory_mb << " MB)\n";

            auto test_data = GenerateTestData(size);

            // CPU测试
            {
                MagneticFieldInterpolator interp(false);
                auto start = std::chrono::high_resolution_clock::now();
                ErrorCode err = interp.LoadFromMemory(
                    test_data.coordinates.data(),
                    test_data.field_data.data(),
                    total_points
                );
                auto end = std::chrono::high_resolution_clock::now();

                if (err == ErrorCode::Success) {
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                    std::cout << "  CPU加载时间: " << duration.count() << " ms\n";
                } else {
                    std::cout << "  CPU加载失败\n";
                }
            }

            // GPU测试
            {
                MagneticFieldInterpolator interp(true);
                auto start = std::chrono::high_resolution_clock::now();
                ErrorCode err = interp.LoadFromMemory(
                    test_data.coordinates.data(),
                    test_data.field_data.data(),
                    total_points
                );
                auto end = std::chrono::high_resolution_clock::now();

                if (err == ErrorCode::Success) {
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                    std::cout << "  GPU加载时间: " << duration.count() << " ms\n";
                } else {
                    std::cout << "  GPU加载失败 (可能是内存不足)\n";
                }
            }

            std::cout << "\n";
        }
    }
};

int main() {
    Benchmark benchmark;
    benchmark.RunAllBenchmarks();
    return 0;
}
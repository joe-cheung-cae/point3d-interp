#include "point3d_interp/api.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <random>
#include <numeric>

int main() {
    using namespace p3d;

    std::cout << "=== 3D Magnetic Field Data Interpolation Time Profile ===\n" << std::endl;

    // Configuration
    const std::vector<size_t> query_sizes    = {100, 1000, 10000, 1000000};
    const int                 num_iterations = 5;
    const std::string         data_file      = "../data/sample_magnetic_field.csv";

    // Load data
    std::cout << "Loading magnetic field data from: " << data_file << std::endl;

    // Create temporary interpolator to get grid parameters
    MagneticFieldInterpolator temp_interp(false);
    ErrorCode                 err = temp_interp.LoadFromCSV(data_file);
    if (err != ErrorCode::Success) {
        std::cerr << "Data loading failed: " << ErrorCodeToString(err) << std::endl;
        return 1;
    }

    const auto& params = temp_interp.GetGridParams();
    std::cout << "Data loaded successfully!" << std::endl;
    std::cout << "Number of data points: " << temp_interp.GetDataPointCount() << std::endl;
    std::cout << "Grid dimensions: " << params.dimensions[0] << " x " << params.dimensions[1] << " x "
              << params.dimensions[2] << std::endl;
    std::cout << "Grid bounds: [" << params.min_bound.x << ", " << params.max_bound.x << "] x [" << params.min_bound.y
              << ", " << params.max_bound.y << "] x [" << params.min_bound.z << ", " << params.max_bound.z << "]"
              << std::endl;
    std::cout << std::endl;

    // Random number generation setup
    std::random_device                    rd;
    std::mt19937                          gen(rd());
    std::uniform_real_distribution<float> dist_x(params.min_bound.x, params.max_bound.x);
    std::uniform_real_distribution<float> dist_y(params.min_bound.y, params.max_bound.y);
    std::uniform_real_distribution<float> dist_z(params.min_bound.z, params.max_bound.z);

    // Function to benchmark interpolator with specific query points
    auto benchmark_interpolator = [&](bool use_gpu, const std::string& name,
                                      const std::vector<Point3D>& query_points) -> double {
        MagneticFieldInterpolator interp(use_gpu);

        err = interp.LoadFromCSV(data_file);
        if (err != ErrorCode::Success) {
            std::cerr << name << " interpolator initialization failed: " << ErrorCodeToString(err) << std::endl;
            return -1.0;
        }

        // Warm up
        InterpolationResult dummy;
        interp.Query(query_points[0], dummy);

        // Benchmark iterations
        std::vector<double> times;
        times.reserve(num_iterations);

        for (int iter = 0; iter < num_iterations; ++iter) {
            std::vector<InterpolationResult> results(query_points.size());

            auto start = std::chrono::high_resolution_clock::now();
            err        = interp.QueryBatch(query_points.data(), results.data(), query_points.size());
            auto end   = std::chrono::high_resolution_clock::now();

            if (err != ErrorCode::Success) {
                std::cerr << name << " query failed at iteration " << iter << ": " << ErrorCodeToString(err)
                          << std::endl;
                return -1.0;
            }

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            times.push_back(duration.count() / 1000.0);  // Convert to milliseconds
        }

        // Calculate average time
        double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        return avg_time;
    };

    // Test different query sizes
    for (size_t query_size : query_sizes) {
        std::cout << "=== Testing with " << query_size << " query points ===" << std::endl;

        // Generate query points for this size
        std::vector<Point3D> query_points;
        query_points.reserve(query_size);

        for (size_t i = 0; i < query_size; ++i) {
            query_points.emplace_back(dist_x(gen), dist_y(gen), dist_z(gen));
        }

        // Benchmark CPU
        std::cout << "Benchmarking CPU interpolation..." << std::endl;
        double cpu_time = benchmark_interpolator(false, "CPU", query_points);
        if (cpu_time < 0) {
            std::cerr << "CPU benchmark failed." << std::endl;
            continue;
        }
        std::cout << "CPU benchmark completed." << std::endl;

        // Benchmark GPU
        std::cout << "Benchmarking GPU interpolation..." << std::endl;
        double gpu_time = benchmark_interpolator(true, "GPU", query_points);
        if (gpu_time < 0) {
            std::cout << "GPU benchmark failed (GPU may not be available)." << std::endl;
            std::cout << "CPU time: " << std::fixed << std::setprecision(3) << cpu_time << " ms" << std::endl;
            std::cout << "Throughput: " << std::fixed << std::setprecision(0) << (query_size / (cpu_time / 1000.0))
                      << " queries/second" << std::endl;
            std::cout << std::endl;
            continue;
        }
        std::cout << "GPU benchmark completed." << std::endl;

        // Calculate speedup
        double speedup = cpu_time / gpu_time;

        // Print results
        std::cout << "Results for " << query_size << " query points:" << std::endl;
        std::cout << "  CPU time: " << std::fixed << std::setprecision(3) << cpu_time << " ms" << std::endl;
        std::cout << "  GPU time: " << std::fixed << std::setprecision(3) << gpu_time << " ms" << std::endl;
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        std::cout << "  Throughput:" << std::endl;
        std::cout << "    CPU: " << std::fixed << std::setprecision(0) << (query_size / (cpu_time / 1000.0))
                  << " queries/second" << std::endl;
        std::cout << "    GPU: " << std::fixed << std::setprecision(0) << (query_size / (gpu_time / 1000.0))
                  << " queries/second" << std::endl;
        std::cout << std::endl;
    }

    std::cout << "=== Time Profile Complete ===" << std::endl;

    return 0;
}
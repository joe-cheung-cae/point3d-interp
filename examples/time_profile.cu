#include "point3d_interp/api.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <random>
#include <numeric>
#include <cuda_runtime.h>

int main() {
    using namespace p3d;

    std::cout << "=============================================\n";
    std::cout << "=== Benchmarking Structured Interpolation ===\n";
    std::cout << "=============================================\n" << std::endl;

    // Configuration
    const std::vector<size_t> query_sizes    = {100, 1000, 10000};
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
    auto benchmark_interpolator_structured = [&](bool use_gpu, const std::string& name,
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

            // For GPU, get kernel-only time if available
            double measured_time;
            if (use_gpu) {
                float kernel_time_ms;
                if (interp.GetLastKernelTime(kernel_time_ms) == ErrorCode::Success) {
                    measured_time = kernel_time_ms;
                } else {
                    // Fallback to total time if kernel timing not available
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                    measured_time = duration.count() / 1000.0;
                }
            } else {
                // CPU timing
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                measured_time = duration.count() / 1000.0;
            }

            times.push_back(measured_time);
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
        double cpu_time = benchmark_interpolator_structured(false, "CPU", query_points);
        if (cpu_time < 0) {
            std::cerr << "CPU benchmark failed." << std::endl;
            continue;
        }
        std::cout << "CPU benchmark completed." << std::endl;

        // Benchmark GPU
        std::cout << "Benchmarking GPU interpolation..." << std::endl;
        double gpu_time = benchmark_interpolator_structured(true, "GPU", query_points);
        if (gpu_time < 0) {
            std::cout << "[CPU, Structured, QuerySize=" << query_size << "] Time: " << std::fixed
                      << std::setprecision(3) << cpu_time << " ms, Throughput: " << std::fixed << std::setprecision(0)
                      << (query_size / (cpu_time / 1000.0)) << " q/s (GPU not available)" << std::endl;
            std::cout << std::endl;
            continue;
        }
        std::cout << "GPU benchmark completed." << std::endl;

        // Calculate speedup
        double speedup = cpu_time / gpu_time;

        // Print results
        std::cout << "[CPU, Structured, QuerySize=" << query_size << "] Time: " << std::fixed << std::setprecision(3)
                  << cpu_time << " ms, Throughput: " << std::fixed << std::setprecision(0)
                  << (query_size / (cpu_time / 1000.0)) << " q/s" << std::endl;
        std::cout << "[GPU, Structured, QuerySize=" << query_size << "] Time: " << std::fixed << std::setprecision(3)
                  << gpu_time << " ms, Throughput: " << std::fixed << std::setprecision(0)
                  << (query_size / (gpu_time / 1000.0)) << " q/s, Speedup: " << std::fixed << std::setprecision(2)
                  << speedup << "x" << std::endl;
        std::cout << std::endl;
    }

    // === Benchmarking Unstructured Interpolation ===
    std::cout << "===============================================\n";
    std::cout << "=== Benchmarking Unstructured Interpolation ===\n";
    std::cout << "===============================================\n" << std::endl;

    // Generate synthetic unstructured data
    const size_t                   num_data_points = 1000;
    std::vector<Point3D>           data_points;
    std::vector<MagneticFieldData> data_field;
    data_points.reserve(num_data_points);
    data_field.reserve(num_data_points);

    // Random generators for data points and field values
    std::uniform_real_distribution<float> pos_dist(-10.0f, 10.0f);
    std::uniform_real_distribution<float> field_dist(-1.0f, 1.0f);

    for (size_t i = 0; i < num_data_points; ++i) {
        data_points.emplace_back(pos_dist(gen), pos_dist(gen), pos_dist(gen));
        data_field.emplace_back(field_dist(gen), field_dist(gen), field_dist(gen));
    }

    // Calculate bounds of the data
    Point3D min_bound = data_points[0];
    Point3D max_bound = data_points[0];
    for (const auto& p : data_points) {
        min_bound.x = std::min(min_bound.x, p.x);
        min_bound.y = std::min(min_bound.y, p.y);
        min_bound.z = std::min(min_bound.z, p.z);
        max_bound.x = std::max(max_bound.x, p.x);
        max_bound.y = std::max(max_bound.y, p.y);
        max_bound.z = std::max(max_bound.z, p.z);
    }

    std::cout << "Generated " << num_data_points << " unstructured data points" << std::endl;
    std::cout << "Data bounds: [" << min_bound.x << ", " << max_bound.x << "] x [" << min_bound.y << ", " << max_bound.y
              << "] x [" << min_bound.z << ", " << max_bound.z << "]" << std::endl;
    std::cout << std::endl;

    // Random number generation setup for query points
    std::uniform_real_distribution<float> query_dist_x(min_bound.x, max_bound.x);
    std::uniform_real_distribution<float> query_dist_y(min_bound.y, max_bound.y);
    std::uniform_real_distribution<float> query_dist_z(min_bound.z, max_bound.z);

    // Function to benchmark unstructured interpolator
    auto benchmark_interpolator_unstructured = [&](bool use_gpu, const std::string& name,
                                                   const std::vector<Point3D>& query_points) -> double {
        std::cout << "Creating " << name << " interpolator (use_gpu=" << use_gpu << ")" << std::endl;
        MagneticFieldInterpolator interp(use_gpu, 0, InterpolationMethod::IDW);

        ErrorCode err = interp.LoadFromMemory(data_points.data(), data_field.data(), num_data_points);
        if (err != ErrorCode::Success) {
            std::cerr << name << " interpolator initialization failed: " << ErrorCodeToString(err) << std::endl;
            return -1.0;
        }
        std::cout << "Data loaded successfully. Data points: " << interp.GetDataPointCount() << std::endl;

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
                cudaError_t cuda_err = cudaGetLastError();
                if (cuda_err != cudaSuccess) {
                    std::cerr << "CUDA error: " << cudaGetErrorString(cuda_err) << std::endl;
                }
                return -1.0;
            }

            // For GPU, get kernel-only time if available
            double measured_time;
            if (use_gpu) {
                float kernel_time_ms;
                if (interp.GetLastKernelTime(kernel_time_ms) == ErrorCode::Success) {
                    measured_time = kernel_time_ms;
                } else {
                    // Fallback to total time if kernel timing not available
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                    measured_time = duration.count() / 1000.0;
                }
            } else {
                // CPU timing
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                measured_time = duration.count() / 1000.0;
            }

            times.push_back(measured_time);
        }

        // Calculate average time
        double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        return avg_time;
    };

    // Test different query sizes for unstructured (limited to avoid GPU memory issues)
    const std::vector<size_t> unstructured_query_sizes = {100, 1000, 10000};
    for (size_t query_size : unstructured_query_sizes) {
        std::cout << "=== Testing unstructured with " << query_size << " query points ===" << std::endl;

        // Generate query points for this size
        std::vector<Point3D> query_points;
        query_points.reserve(query_size);

        for (size_t i = 0; i < query_size; ++i) {
            query_points.emplace_back(query_dist_x(gen), query_dist_y(gen), query_dist_z(gen));
        }

        // Benchmark CPU
        std::cout << "Benchmarking CPU unstructured interpolation..." << std::endl;
        double cpu_time = benchmark_interpolator_unstructured(false, "CPU Unstructured", query_points);
        if (cpu_time < 0) {
            std::cerr << "CPU unstructured benchmark failed.\n\n";
            continue;
        }
        std::cout << "CPU unstructured benchmark completed.\n\n";

        // Benchmark GPU
        std::cout << "\nBenchmarking GPU unstructured interpolation..." << std::endl;
        double gpu_time = benchmark_interpolator_unstructured(true, "GPU Unstructured", query_points);
        if (gpu_time < 0) {
            std::cout << "[CPU, Unstructured, QuerySize=" << query_size << "] Time: " << std::fixed
                      << std::setprecision(3) << cpu_time << " ms, Throughput: " << std::fixed << std::setprecision(0)
                      << (query_size / (cpu_time / 1000.0)) << " q/s (GPU not available)" << std::endl;
            std::cout << std::endl;
            continue;
        }
        std::cout << "GPU unstructured benchmark completed.\n\n";

        // Calculate speedup
        double speedup = cpu_time / gpu_time;

        // Print results
        std::cout << "[CPU, Unstructured, QuerySize=" << query_size << "] Time: " << std::fixed << std::setprecision(3)
                  << cpu_time << " ms, Throughput: " << std::fixed << std::setprecision(0)
                  << (query_size / (cpu_time / 1000.0)) << " q/s" << std::endl;
        std::cout << "[GPU, Unstructured, QuerySize=" << query_size << "] Time: " << std::fixed << std::setprecision(3)
                  << gpu_time << " ms, Throughput: " << std::fixed << std::setprecision(0)
                  << (query_size / (gpu_time / 1000.0)) << " q/s, Speedup: " << std::fixed << std::setprecision(2)
                  << speedup << "x" << std::endl;
        std::cout << std::endl;
    }

    std::cout << "=== Time Profile Complete ===" << std::endl;

    return 0;
}
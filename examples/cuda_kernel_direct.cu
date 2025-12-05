#include "point3d_interp/interpolator_api.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

int main() {
    using namespace p3d;

    std::cout << "=== Direct CUDA Kernel Interface Example ===\n" << std::endl;

    // Create interpolator with GPU support
    MagneticFieldInterpolator interp(true);  // Use GPU

    // Load data
    std::cout << "Loading magnetic field data..." << std::endl;
    try {
        interp.LoadFromCSV("../data/sample_magnetic_field.csv");
    } catch (const std::exception& e) {
        std::cerr << "Data loading failed: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Data loaded successfully!\n" << std::endl;

    // Get GPU device pointers for direct CUDA access
    const Point3D*           d_grid_points = interp.GetDeviceGridPoints();
    const MagneticFieldData* d_field_data  = interp.GetDeviceFieldData();

    if (!d_grid_points || !d_field_data) {
        std::cerr << "GPU device pointers not available. Make sure GPU is enabled and data is loaded." << std::endl;
        return 1;
    }

    std::cout << "GPU device pointers obtained successfully!" << std::endl;
    std::cout << "Grid points pointer: " << d_grid_points << std::endl;
    std::cout << "Field data pointer: " << d_field_data << std::endl;
    std::cout << std::endl;

    // Prepare query points
    const size_t                     num_queries = 10000;
    std::vector<Point3D>             host_query_points(num_queries);
    std::vector<InterpolationResult> host_results(num_queries);

    // Generate random query points within the grid bounds
    const auto& params = interp.GetGridParams();
    for (size_t i = 0; i < num_queries; ++i) {
        Point3D& point = host_query_points[i];
        point.x        = params.min_bound.x + (params.max_bound.x - params.min_bound.x) * (rand() / double(RAND_MAX));
        point.y        = params.min_bound.y + (params.max_bound.y - params.min_bound.y) * (rand() / double(RAND_MAX));
        point.z        = params.min_bound.z + (params.max_bound.z - params.min_bound.z) * (rand() / double(RAND_MAX));
    }

    std::cout << "Generated " << num_queries << " random query points" << std::endl;
    std::cout << "Grid bounds: [" << params.min_bound.x << ", " << params.max_bound.x << "] x [" << params.min_bound.y
              << ", " << params.max_bound.y << "] x [" << params.min_bound.z << ", " << params.max_bound.z << "]"
              << std::endl;
    std::cout << std::endl;

    // Allocate GPU memory for query points and results
    Point3D*             d_query_points;
    InterpolationResult* d_results;

    cudaError_t cuda_err;
    cuda_err = cudaMalloc(&d_query_points, num_queries * sizeof(Point3D));
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for query points: " << cudaGetErrorString(cuda_err) << std::endl;
        return 1;
    }

    cuda_err = cudaMalloc(&d_results, num_queries * sizeof(InterpolationResult));
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for results: " << cudaGetErrorString(cuda_err) << std::endl;
        cudaFree(d_query_points);
        return 1;
    }

    // Copy query points to GPU
    cuda_err =
        cudaMemcpy(d_query_points, host_query_points.data(), num_queries * sizeof(Point3D), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to copy query points to GPU: " << cudaGetErrorString(cuda_err) << std::endl;
        cudaFree(d_query_points);
        cudaFree(d_results);
        return 1;
    }

    std::cout << "GPU memory allocated and data copied successfully!" << std::endl;
    std::cout << std::endl;

    // Get optimal kernel launch configuration
    KernelConfig config;
    interp.GetOptimalKernelConfig(num_queries, config);

    std::cout << "Optimal kernel configuration:" << std::endl;
    std::cout << "Block dimensions: (" << config.block_x << ", " << config.block_y << ", " << config.block_z << ")"
              << std::endl;
    std::cout << "Grid dimensions: (" << config.grid_x << ", " << config.grid_y << ", " << config.grid_z << ")"
              << std::endl;
    std::cout << std::endl;

    // Launch interpolation kernel directly
    std::cout << "Launching CUDA interpolation kernel..." << std::endl;

    // Calculate shared memory size for kernel
    const size_t shared_mem_size = sizeof(GridParams);

    auto start = std::chrono::high_resolution_clock::now();

    // Call the kernel directly
    p3d::cuda::
        TricubicHermiteInterpolationKernel<<<dim3(config.grid_x, config.grid_y, config.grid_z),
                                             dim3(config.block_x, config.block_y, config.block_z), shared_mem_size>>>(
            d_query_points, d_field_data, params, d_results, num_queries, 0);  // 0 = ExtrapolationMethod::None

    // Check for kernel launch errors
    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(cuda_err) << std::endl;
        cudaFree(d_query_points);
        cudaFree(d_results);
        return 1;
    }

    // Wait for kernel completion
    cuda_err = cudaDeviceSynchronize();
    if (cuda_err != cudaSuccess) {
        std::cerr << "CUDA kernel synchronization error: " << cudaGetErrorString(cuda_err) << std::endl;
        cudaFree(d_query_points);
        cudaFree(d_results);
        return 1;
    }

    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Kernel execution completed in " << duration.count() << " milliseconds" << std::endl;
    std::cout << std::endl;

    // Copy results back to host
    cuda_err =
        cudaMemcpy(host_results.data(), d_results, num_queries * sizeof(InterpolationResult), cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to copy results from GPU: " << cudaGetErrorString(cuda_err) << std::endl;
        cudaFree(d_query_points);
        cudaFree(d_results);
        return 1;
    }

    std::cout << "Results copied back to host successfully!" << std::endl;
    std::cout << std::endl;

    // Display some results
    std::cout << "Sample interpolation results:" << std::endl;
    size_t valid_count = 0;
    for (size_t i = 0; i < std::min(size_t(10), num_queries); ++i) {
        const auto& point  = host_query_points[i];
        const auto& result = host_results[i];
        if (result.valid) {
            valid_count++;
            std::cout << "Point " << i + 1 << ": (" << std::fixed << std::setprecision(3) << point.x << ", " << point.y
                      << ", " << point.z << ") -> "
                      << "B = (" << std::setprecision(6) << result.data.Bx << ", " << result.data.By << ", "
                      << result.data.Bz << ")" << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "Total query points: " << num_queries << std::endl;
    std::cout << "Valid interpolations: " << valid_count << std::endl;
    std::cout << "Success rate: " << std::fixed << std::setprecision(1) << (valid_count * 100.0 / num_queries) << "%"
              << std::endl;

    // Clean up GPU memory
    cudaFree(d_query_points);
    cudaFree(d_results);

    std::cout << std::endl;
    std::cout << "=== Direct CUDA Kernel Example Completed Successfully ===" << std::endl;

    return 0;
}
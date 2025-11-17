#include <gtest/gtest.h>
#include "point3d_interp/api.h"
#include <fstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

namespace p3d {
namespace test {

class GPUTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create test data programmatically (same as CPU tests)
        CreateTestData();
    }

    void TearDown() override {
        // Clean up
    }

    void CreateTestData() {
        // Create test grid (3x3x3)
        grid_params_.origin     = Point3D(0.0f, 0.0f, 0.0f);
        grid_params_.spacing    = Point3D(1.0f, 1.0f, 1.0f);
        grid_params_.dimensions = {3, 3, 3};
        grid_params_.update_bounds();

        // Generate coordinates and field data
        size_t total_points = 3 * 3 * 3;
        coordinates_.reserve(total_points);
        field_data_.reserve(total_points);

        for (int k = 0; k < 3; ++k) {
            for (int j = 0; j < 3; ++j) {
                for (int i = 0; i < 3; ++i) {
                    float x = i * 1.0f;
                    float y = j * 1.0f;
                    float z = k * 1.0f;
                    coordinates_.push_back(Point3D(x, y, z));
                    // B = (x,y,z) with gradients for linear field
                    field_data_.push_back(MagneticFieldData(x, y, z,           // B = (x, y, z)
                                                            1.0f, 0.0f, 0.0f,  // dBx_dx, dBx_dy, dBx_dz
                                                            0.0f, 1.0f, 0.0f,  // dBy_dx, dBy_dy, dBy_dz
                                                            0.0f, 0.0f, 1.0f   // dBz_dx, dBz_dy, dBz_dz
                                                            ));
                }
            }
        }
    }

    GridParams                     grid_params_;
    std::vector<Point3D>           coordinates_;
    std::vector<MagneticFieldData> field_data_;
};

// Test GPU initialization and basic functionality
TEST_F(GPUTest, GPUInitialization) {
    MagneticFieldInterpolator interp(true, 0);  // use_gpu=true, device_id=0

    ErrorCode err = interp.LoadFromMemory(coordinates_.data(), field_data_.data(), coordinates_.size());
    EXPECT_EQ(err, ErrorCode::Success);
    EXPECT_TRUE(interp.IsDataLoaded());
    EXPECT_EQ(interp.GetDataPointCount(), 27u);  // 3x3x3 grid
}

// Test single point GPU query
TEST_F(GPUTest, GPUSinglePointQuery) {
    MagneticFieldInterpolator interp(true, 0);

    interp.LoadFromMemory(coordinates_.data(), field_data_.data(), coordinates_.size());

    Point3D             query_point(0.5f, 0.5f, 0.5f);
    InterpolationResult result;

    ErrorCode err = interp.Query(query_point, result);
    EXPECT_EQ(err, ErrorCode::Success);
    EXPECT_TRUE(result.valid);

    // For linear field Bx=x, By=y, Bz=z, interpolated value should be close to query point
    EXPECT_NEAR(result.data.Bx, 0.5f, 1e-3f);
    EXPECT_NEAR(result.data.By, 0.5f, 1e-3f);
    EXPECT_NEAR(result.data.Bz, 0.5f, 1e-3f);

    // Export query point and result for visualization
    std::vector<Point3D>             query_points = {query_point};
    std::vector<InterpolationResult> results      = {result};
    err =
        interp.ExportOutputPoints(ExportFormat::ParaviewVTK, query_points, results, "test_output/gpu_single_point.vtk");
    EXPECT_EQ(err, ErrorCode::Success);
}

// Test batch GPU query
TEST_F(GPUTest, GPUBatchQuery) {
    MagneticFieldInterpolator interp(true, 0);
    interp.LoadFromMemory(coordinates_.data(), field_data_.data(), coordinates_.size());

    std::vector<Point3D> query_points = {{0.5f, 0.5f, 0.5f}, {1.2f, 0.8f, 1.5f}, {0.3f, 1.7f, 0.9f}};

    std::vector<InterpolationResult> results(query_points.size());

    ErrorCode err = interp.QueryBatch(query_points.data(), results.data(), query_points.size());
    EXPECT_EQ(err, ErrorCode::Success);

    for (const auto& result : results) {
        EXPECT_TRUE(result.valid);
        // Check reasonable ranges
        EXPECT_GE(result.data.Bx, 0.0f);
        EXPECT_LE(result.data.Bx, 3.0f);
        EXPECT_GE(result.data.By, 0.0f);
        EXPECT_LE(result.data.By, 3.0f);
        EXPECT_GE(result.data.Bz, 0.0f);
        EXPECT_LE(result.data.Bz, 3.0f);
    }

    // Export batch query points and results for visualization
    err =
        interp.ExportOutputPoints(ExportFormat::ParaviewVTK, query_points, results, "test_output/gpu_batch_query.vtk");
    EXPECT_EQ(err, ErrorCode::Success);
}

// Test CPU/GPU consistency
TEST_F(GPUTest, CPU_GPU_Consistency) {
    // CPU interpolator
    MagneticFieldInterpolator cpu_interp(false, 0);  // use_gpu=false
    cpu_interp.LoadFromMemory(coordinates_.data(), field_data_.data(), coordinates_.size());

    // GPU interpolator
    MagneticFieldInterpolator gpu_interp(true, 0);  // use_gpu=true
    gpu_interp.LoadFromMemory(coordinates_.data(), field_data_.data(), coordinates_.size());

    // Test points
    std::vector<Point3D> test_points = {
        {0.5f, 0.5f, 0.5f}, {1.2f, 0.8f, 1.5f}, {0.3f, 1.7f, 0.9f}, {2.0f, 1.3f, 0.7f}, {0.1f, 0.2f, 0.3f}};

    std::vector<InterpolationResult> cpu_results, gpu_results;

    for (const auto& point : test_points) {
        InterpolationResult cpu_result, gpu_result;

        ErrorCode cpu_err = cpu_interp.Query(point, cpu_result);
        ErrorCode gpu_err = gpu_interp.Query(point, gpu_result);

        EXPECT_EQ(cpu_err, ErrorCode::Success);
        EXPECT_EQ(gpu_err, ErrorCode::Success);
        EXPECT_EQ(cpu_result.valid, gpu_result.valid);

        cpu_results.push_back(cpu_result);
        gpu_results.push_back(gpu_result);

        if (cpu_result.valid && gpu_result.valid) {
            // Compare results with tolerance for floating point precision
            EXPECT_NEAR(cpu_result.data.Bx, gpu_result.data.Bx, 1e-5f);
            EXPECT_NEAR(cpu_result.data.By, gpu_result.data.By, 1e-5f);
            EXPECT_NEAR(cpu_result.data.Bz, gpu_result.data.Bz, 1e-5f);

            std::cout << std::fixed << std::setprecision(8);
            std::cout << "Point (" << point.x << "," << point.y << "," << point.z << "): "
                      << "CPU Bx=" << cpu_result.data.Bx << " GPU Bx=" << gpu_result.data.Bx
                      << " Diff=" << std::abs(cpu_result.data.Bx - gpu_result.data.Bx) << std::endl;
        }
    }

    // Export CPU and GPU results for comparison visualization
    ErrorCode err = cpu_interp.ExportOutputPoints(ExportFormat::ParaviewVTK, test_points, cpu_results,
                                                  "test_output/cpu_consistency.vtk");
    EXPECT_EQ(err, ErrorCode::Success);
    err = gpu_interp.ExportOutputPoints(ExportFormat::ParaviewVTK, test_points, gpu_results,
                                        "test_output/gpu_consistency.vtk");
    EXPECT_EQ(err, ErrorCode::Success);
}

// Test GPU out of bounds handling
TEST_F(GPUTest, GPUOutOfBounds) {
    MagneticFieldInterpolator interp(true, 0);
    interp.LoadFromMemory(coordinates_.data(), field_data_.data(), coordinates_.size());

    Point3D             out_of_bounds(-1.0f, 0.5f, 0.5f);
    InterpolationResult result;

    ErrorCode err = interp.Query(out_of_bounds, result);
    EXPECT_EQ(err, ErrorCode::Success);
    EXPECT_FALSE(result.valid);
}

// Test invalid GPU device ID
TEST_F(GPUTest, InvalidGPUDevice) {
    // Try device ID 999 (likely invalid)
    MagneticFieldInterpolator interp(true, 999);

    ErrorCode err = interp.LoadFromMemory(coordinates_.data(), field_data_.data(), coordinates_.size());
    // Should either fail gracefully or succeed if device ID is ignored
    // For now, just check it doesn't crash
    SUCCEED();
}

// Test GPU Hermite interpolation accuracy
TEST_F(GPUTest, GPUHermiteInterpolationAccuracy) {
    MagneticFieldInterpolator gpu_interp(true, 0);
    gpu_interp.LoadFromMemory(coordinates_.data(), field_data_.data(), coordinates_.size());

    // Test multiple query points for Hermite interpolation accuracy
    std::vector<Point3D> test_points = {{0.5f, 0.5f, 0.5f}, {1.2f, 0.8f, 1.5f}, {0.3f, 1.7f, 0.9f}};

    std::vector<InterpolationResult> results;

    for (const auto& point : test_points) {
        InterpolationResult result;
        ErrorCode           err = gpu_interp.Query(point, result);
        EXPECT_EQ(err, ErrorCode::Success);
        EXPECT_TRUE(result.valid);
        results.push_back(result);

        // For linear field Bx=x, By=y, Bz=z with correct gradients,
        // Hermite interpolation should be very accurate
        float expected_Bx = point.x;
        float expected_By = point.y;
        float expected_Bz = point.z;

        // Log results for debugging
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "GPU Hermite test point: (" << point.x << ", " << point.y << ", " << point.z << ")" << std::endl;
        std::cout << "  Expected: Bx=" << expected_Bx << ", By=" << expected_By << ", Bz=" << expected_Bz << std::endl;
        std::cout << "  GPU Result: Bx=" << result.data.Bx << ", By=" << result.data.By << ", Bz=" << result.data.Bz
                  << std::endl;
        std::cout << "  Errors: Bx=" << std::abs(result.data.Bx - expected_Bx)
                  << ", By=" << std::abs(result.data.By - expected_By)
                  << ", Bz=" << std::abs(result.data.Bz - expected_Bz) << std::endl
                  << std::endl;

        // Hermite interpolation should be very accurate for linear fields
        EXPECT_NEAR(result.data.Bx, expected_Bx, 1e-4f);
        EXPECT_NEAR(result.data.By, expected_By, 1e-4f);
        EXPECT_NEAR(result.data.Bz, expected_Bz, 1e-4f);
    }

    // Export Hermite interpolation results for visualization
    ErrorCode err = gpu_interp.ExportOutputPoints(ExportFormat::ParaviewVTK, test_points, results,
                                                  "test_output/gpu_hermite_accuracy.vtk");
    EXPECT_EQ(err, ErrorCode::Success);
}

// Test GPU gradient verification - compare with CPU to ensure gradients are used
TEST_F(GPUTest, GPUGradientVerification) {
    // CPU interpolator
    MagneticFieldInterpolator cpu_interp(false, 0);
    cpu_interp.LoadFromMemory(coordinates_.data(), field_data_.data(), coordinates_.size());

    // GPU interpolator
    MagneticFieldInterpolator gpu_interp(true, 0);
    gpu_interp.LoadFromMemory(coordinates_.data(), field_data_.data(), coordinates_.size());

    // Test points that would benefit from gradient information
    std::vector<Point3D> test_points = {{0.25f, 0.25f, 0.25f}, {0.75f, 0.75f, 0.75f}, {1.25f, 1.25f, 1.25f}};

    std::vector<InterpolationResult> cpu_results, gpu_results;

    for (const auto& point : test_points) {
        InterpolationResult cpu_result, gpu_result;

        ErrorCode cpu_err = cpu_interp.Query(point, cpu_result);
        ErrorCode gpu_err = gpu_interp.Query(point, gpu_result);

        EXPECT_EQ(cpu_err, ErrorCode::Success);
        EXPECT_EQ(gpu_err, ErrorCode::Success);
        EXPECT_EQ(cpu_result.valid, gpu_result.valid);

        cpu_results.push_back(cpu_result);
        gpu_results.push_back(gpu_result);

        if (cpu_result.valid && gpu_result.valid) {
            // CPU and GPU should produce nearly identical results
            EXPECT_NEAR(cpu_result.data.Bx, gpu_result.data.Bx, 1e-5f);
            EXPECT_NEAR(cpu_result.data.By, gpu_result.data.By, 1e-5f);
            EXPECT_NEAR(cpu_result.data.Bz, gpu_result.data.Bz, 1e-5f);

            std::cout << std::fixed << std::setprecision(8);
            std::cout << "Gradient verification point (" << point.x << "," << point.y << "," << point.z << "): "
                      << "CPU Bx=" << cpu_result.data.Bx << " GPU Bx=" << gpu_result.data.Bx
                      << " Diff=" << std::abs(cpu_result.data.Bx - gpu_result.data.Bx) << std::endl;
        }
    }

    // Export gradient verification results for visualization
    ErrorCode err = cpu_interp.ExportOutputPoints(ExportFormat::ParaviewVTK, test_points, cpu_results,
                                                  "test_output/cpu_gradient_verification.vtk");
    EXPECT_EQ(err, ErrorCode::Success);
    err = gpu_interp.ExportOutputPoints(ExportFormat::ParaviewVTK, test_points, gpu_results,
                                        "test_output/gpu_gradient_verification.vtk");
    EXPECT_EQ(err, ErrorCode::Success);
}

// Test GPU boundary interpolation
TEST_F(GPUTest, GPUBoundaryInterpolation) {
    MagneticFieldInterpolator gpu_interp(true, 0);
    gpu_interp.LoadFromMemory(coordinates_.data(), field_data_.data(), coordinates_.size());

    // Test points very close to grid boundaries
    std::vector<Point3D> boundary_points = {
        {0.01f, 0.01f, 0.01f},  // Near origin
        {1.99f, 1.99f, 1.99f},  // Near max boundary
        {0.0f, 0.0f, 0.0f},     // Exactly at grid point
        {2.0f, 2.0f, 2.0f}      // Exactly at grid point
    };

    std::vector<InterpolationResult> results;

    for (const auto& point : boundary_points) {
        InterpolationResult result;
        ErrorCode           err = gpu_interp.Query(point, result);
        EXPECT_EQ(err, ErrorCode::Success);
        EXPECT_TRUE(result.valid);
        results.push_back(result);

        // Check that results are reasonable
        EXPECT_GE(result.data.Bx, -0.1f);
        EXPECT_LE(result.data.Bx, 2.1f);
        EXPECT_GE(result.data.By, -0.1f);
        EXPECT_LE(result.data.By, 2.1f);
        EXPECT_GE(result.data.Bz, -0.1f);
        EXPECT_LE(result.data.Bz, 2.1f);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "GPU boundary point (" << point.x << "," << point.y << "," << point.z << "): "
                  << "Bx=" << result.data.Bx << ", By=" << result.data.By << ", Bz=" << result.data.Bz << std::endl;
    }

    // Export boundary interpolation results for visualization
    ErrorCode err = gpu_interp.ExportOutputPoints(ExportFormat::ParaviewVTK, boundary_points, results,
                                                  "test_output/gpu_boundary_interpolation.vtk");
    EXPECT_EQ(err, ErrorCode::Success);
}

// Test GPU performance regression - ensure queries are reasonably fast
TEST_F(GPUTest, GPUPerformanceRegression) {
    MagneticFieldInterpolator gpu_interp(true, 0);
    gpu_interp.LoadFromMemory(coordinates_.data(), field_data_.data(), coordinates_.size());

    // Generate many query points for performance testing
    std::vector<Point3D>                  query_points;
    std::mt19937                          rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 2.0f);
    for (int i = 0; i < 1000; ++i) {
        query_points.push_back(Point3D(dist(rng), dist(rng), dist(rng)));
    }

    std::vector<InterpolationResult> results(query_points.size());

    // Time the batch query
    auto      start = std::chrono::high_resolution_clock::now();
    ErrorCode err   = gpu_interp.QueryBatch(query_points.data(), results.data(), query_points.size());
    auto      end   = std::chrono::high_resolution_clock::now();

    EXPECT_EQ(err, ErrorCode::Success);

    auto   duration           = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double queries_per_second = query_points.size() / (duration.count() / 1000.0);

    std::cout << "GPU performance: " << query_points.size() << " queries in " << duration.count() << " ms ("
              << std::fixed << std::setprecision(0) << queries_per_second << " queries/second)" << std::endl;

    // Basic performance check - should be reasonably fast (at least 1000 queries/second)
    EXPECT_GE(queries_per_second, 1000.0);
}

// Test GPU with different grid configurations
TEST_F(GPUTest, GPUDifferentGridConfigs) {
    // Create a larger 4x4x4 grid programmatically
    GridParams large_params;
    large_params.origin     = Point3D(0.0f, 0.0f, 0.0f);
    large_params.spacing    = Point3D(0.5f, 0.5f, 0.5f);
    large_params.dimensions = {4, 4, 4};
    large_params.update_bounds();

    std::vector<Point3D>           large_coords;
    std::vector<MagneticFieldData> large_data;
    size_t                         large_total = 4 * 4 * 4;
    large_coords.reserve(large_total);
    large_data.reserve(large_total);

    for (int k = 0; k < 4; ++k) {
        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < 4; ++i) {
                float x = i * 0.5f;
                float y = j * 0.5f;
                float z = k * 0.5f;
                large_coords.push_back(Point3D(x, y, z));
                large_data.push_back(MagneticFieldData(x, y, z, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f));
            }
        }
    }

    MagneticFieldInterpolator gpu_interp(true, 0);
    ErrorCode err = gpu_interp.LoadFromMemory(large_coords.data(), large_data.data(), large_coords.size());
    EXPECT_EQ(err, ErrorCode::Success);
    EXPECT_TRUE(gpu_interp.IsDataLoaded());
    EXPECT_EQ(gpu_interp.GetDataPointCount(), 64u);  // 4x4x4 grid

    // Test query on larger grid
    Point3D             query_point(0.75f, 0.75f, 0.75f);
    InterpolationResult result;
    err = gpu_interp.Query(query_point, result);
    EXPECT_EQ(err, ErrorCode::Success);
    EXPECT_TRUE(result.valid);

    // Should interpolate correctly
    EXPECT_NEAR(result.data.Bx, 0.75f, 1e-3f);
    EXPECT_NEAR(result.data.By, 0.75f, 1e-3f);
    EXPECT_NEAR(result.data.Bz, 0.75f, 1e-3f);

    // Export different grid configuration results for visualization
    std::vector<Point3D>             query_points = {query_point};
    std::vector<InterpolationResult> results      = {result};
    err = gpu_interp.ExportOutputPoints(ExportFormat::ParaviewVTK, query_points, results,
                                        "test_output/gpu_different_grid.vtk");
    EXPECT_EQ(err, ErrorCode::Success);
}

// Test GPU with quadratic field to verify Hermite interpolation benefits
TEST_F(GPUTest, GPUQuadraticFieldInterpolation) {
    // Create quadratic field data programmatically
    std::vector<Point3D>           quad_coords;
    std::vector<MagneticFieldData> quad_data;
    size_t                         total = 3 * 3 * 3;
    quad_coords.reserve(total);
    quad_data.reserve(total);

    for (int k = 0; k < 3; ++k) {
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                float x = i * 1.0f;
                float y = j * 1.0f;
                float z = k * 1.0f;
                // Quadratic field: Bx = x^2, By = y^2, Bz = z^2
                float Bx = x * x;
                float By = y * y;
                float Bz = z * z;
                // Gradients: dBx_dx = 2x, dBy_dy = 2y, dBz_dz = 2z
                quad_coords.push_back(Point3D(x, y, z));
                quad_data.push_back(
                    MagneticFieldData(Bx, By, Bz, 2 * x, 0.0f, 0.0f, 0.0f, 2 * y, 0.0f, 0.0f, 0.0f, 2 * z));
            }
        }
    }

    MagneticFieldInterpolator gpu_interp(true, 0);
    gpu_interp.LoadFromMemory(quad_coords.data(), quad_data.data(), quad_coords.size());

    // Test interpolation at midpoints
    std::vector<Point3D> test_points = {{0.5f, 0.5f, 0.5f}, {1.5f, 1.5f, 1.5f}};

    std::vector<InterpolationResult> results;

    for (const auto& point : test_points) {
        InterpolationResult result;
        ErrorCode           err = gpu_interp.Query(point, result);
        EXPECT_EQ(err, ErrorCode::Success);
        EXPECT_TRUE(result.valid);
        results.push_back(result);

        // Analytical solution for quadratic field
        float expected_Bx = point.x * point.x;
        float expected_By = point.y * point.y;
        float expected_Bz = point.z * point.z;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "GPU quadratic point (" << point.x << "," << point.y << "," << point.z << "): "
                  << "Expected Bx=" << expected_Bx << " Got Bx=" << result.data.Bx
                  << " Error=" << std::abs(result.data.Bx - expected_Bx) << std::endl;

        // Hermite interpolation should provide good accuracy for quadratic fields
        EXPECT_NEAR(result.data.Bx, expected_Bx, 1e-2f);
        EXPECT_NEAR(result.data.By, expected_By, 1e-2f);
        EXPECT_NEAR(result.data.Bz, expected_Bz, 1e-2f);
    }

    // Export quadratic field interpolation results for visualization
    ErrorCode err = gpu_interp.ExportOutputPoints(ExportFormat::ParaviewVTK, test_points, results,
                                                  "test_output/gpu_quadratic_field.vtk");
    EXPECT_EQ(err, ErrorCode::Success);
}

// Test GPU memory management with move semantics
TEST_F(GPUTest, GPUMoveSemantics) {
    MagneticFieldInterpolator interp1(true, 0);
    interp1.LoadFromMemory(coordinates_.data(), field_data_.data(), coordinates_.size());

    // Move constructor
    MagneticFieldInterpolator interp2(std::move(interp1));
    EXPECT_TRUE(interp2.IsDataLoaded());
    // interp1 should be invalid after move

    // Test query on moved interpolator
    Point3D             point(0.5f, 0.5f, 0.5f);
    InterpolationResult result;
    ErrorCode           err = interp2.Query(point, result);
    EXPECT_EQ(err, ErrorCode::Success);
    EXPECT_TRUE(result.valid);
}

}  // namespace test
}  // namespace p3d
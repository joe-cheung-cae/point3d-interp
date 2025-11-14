#include <gtest/gtest.h>
#include "point3d_interp/cpu_interpolator.h"
#include "point3d_interp/grid_structure.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace p3d {
namespace test {

// Test CPU and GPU interpolation result consistency
TEST(AccuracyTest, CPUInterpolationConsistency) {
    // Create test grid (3x3x3)
    GridParams params;
    params.origin     = Point3D(0.0f, 0.0f, 0.0f);
    params.spacing    = Point3D(1.0f, 1.0f, 1.0f);
    params.dimensions = {3, 3, 3};
    params.update_bounds();

    RegularGrid3D grid(params);

    // Set test data for known function: Bx = 2x, By = 2y, Bz = 2z
    // Derivatives: dBx_dx=2, dBx_dy=0, dBx_dz=0, dBy_dx=0, dBy_dy=2, dBy_dz=0, dBz_dx=0, dBz_dy=0, dBz_dz=2
    auto& field_data = const_cast<std::vector<MagneticFieldData>&>(grid.getFieldData());
    for (size_t i = 0; i < field_data.size(); ++i) {
        const auto& coord = grid.getCoordinates()[i];
        float       x = coord.x, y = coord.y, z = coord.z;
        field_data[i] = MagneticFieldData(2 * x, 2 * y, 2 * z,  // B = (2x, 2y, 2z)
                                          2.0f, 0.0f, 0.0f,     // dBx_dx, dBx_dy, dBx_dz
                                          0.0f, 2.0f, 0.0f,     // dBy_dx, dBy_dy, dBy_dz
                                          0.0f, 0.0f, 2.0f      // dBz_dx, dBz_dy, dBz_dz
        );
    }

    CPUInterpolator cpu_interp(grid);

    // Test multiple query points
    std::vector<Point3D> test_points = {{0.5f, 0.5f, 0.5f}, {1.2f, 0.8f, 1.5f}, {0.3f, 1.7f, 0.9f}, {2.1f, 1.3f, 0.7f}};

    for (const auto& point : test_points) {
        InterpolationResult result = cpu_interp.query(point);
        ASSERT_TRUE(result.valid);

        // Calculate analytical solution
        float x = point.x, y = point.y, z = point.z;
        float expected_Bx = 2 * x;
        float expected_By = 2 * y;
        float expected_Bz = 2 * z;

        // Log results
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Query point: (" << point.x << ", " << point.y << ", " << point.z << ")" << std::endl;
        std::cout << "  Expected: Bx=" << expected_Bx << ", By=" << expected_By << ", Bz=" << expected_Bz << std::endl;
        std::cout << "  Interpolated: Bx=" << result.data.Bx << ", By=" << result.data.By << ", Bz=" << result.data.Bz
                  << std::endl;
        std::cout << "  Errors: Bx=" << std::abs(result.data.Bx - expected_Bx)
                  << ", By=" << std::abs(result.data.By - expected_By)
                  << ", Bz=" << std::abs(result.data.Bz - expected_Bz) << std::endl
                  << std::endl;

        // Check interpolation accuracy
        EXPECT_NEAR(result.data.Bx, expected_Bx, 1e-3f);
        EXPECT_NEAR(result.data.By, expected_By, 1e-3f);
        EXPECT_NEAR(result.data.Bz, expected_Bz, 1e-3f);
    }
}

// Test interpolation accuracy for boundary cases
TEST(AccuracyTest, BoundaryInterpolation) {
    // Create simple 2x2x2 grid
    GridParams params;
    params.origin     = Point3D(0.0f, 0.0f, 0.0f);
    params.spacing    = Point3D(1.0f, 1.0f, 1.0f);
    params.dimensions = {2, 2, 2};
    params.update_bounds();

    RegularGrid3D grid(params);

    // Set linear field: Bx = 1, By = 2, Bz = 3
    auto& field_data = const_cast<std::vector<MagneticFieldData>&>(grid.getFieldData());
    for (size_t i = 0; i < field_data.size(); ++i) {
        field_data[i] = MagneticFieldData(1.0f, 2.0f, 3.0f,  // B = (1, 2, 3)
                                          0.0f, 0.0f, 0.0f,  // dBx_dx, dBx_dy, dBx_dz (constant)
                                          0.0f, 0.0f, 0.0f,  // dBy_dx, dBy_dy, dBy_dz
                                          0.0f, 0.0f, 0.0f   // dBz_dx, dBz_dy, dBz_dz
        );
    }

    CPUInterpolator cpu_interp(grid);

    // Test multiple points within cells
    std::vector<Point3D> test_points;
    for (float x = 0.1f; x < 1.0f; x += 0.2f) {
        for (float y = 0.1f; y < 1.0f; y += 0.2f) {
            for (float z = 0.1f; z < 1.0f; z += 0.2f) {
                test_points.push_back(Point3D(x, y, z));
            }
        }
    }

    for (const auto& point : test_points) {
        InterpolationResult result = cpu_interp.query(point);
        ASSERT_TRUE(result.valid);

        // Log results
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Boundary test point: (" << point.x << ", " << point.y << ", " << point.z << ")" << std::endl;
        std::cout << "  Expected: Bx=1.0, By=2.0, Bz=3.0" << std::endl;
        std::cout << "  Interpolated: Bx=" << result.data.Bx << ", By=" << result.data.By << ", Bz=" << result.data.Bz
                  << std::endl;
        std::cout << "  Errors: Bx=" << std::abs(result.data.Bx - 1.0f) << ", By=" << std::abs(result.data.By - 2.0f)
                  << ", Bz=" << std::abs(result.data.Bz - 3.0f) << std::endl
                  << std::endl;

        // For linear fields, Hermite interpolation should be exact
        EXPECT_NEAR(result.data.Bx, 1.0f, 1e-6f);
        EXPECT_NEAR(result.data.By, 2.0f, 1e-6f);
        EXPECT_NEAR(result.data.Bz, 3.0f, 1e-6f);
    }
}

// Test exactness of grid point interpolation
TEST(AccuracyTest, GridPointExactness) {
    // Create test grid
    GridParams params;
    params.origin     = Point3D(0.0f, 0.0f, 0.0f);
    params.spacing    = Point3D(0.5f, 0.5f, 0.5f);
    params.dimensions = {5, 5, 5};
    params.update_bounds();

    RegularGrid3D grid(params);

    // Set random data
    auto& field_data = const_cast<std::vector<MagneticFieldData>&>(grid.getFieldData());
    for (auto& data : field_data) {
        data = MagneticFieldData(static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX,
                                 static_cast<float>(rand()) / RAND_MAX, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                 0.0f);
    }

    CPUInterpolator cpu_interp(grid);

    // Test all grid points
    const auto& coordinates = grid.getCoordinates();
    for (size_t i = 0; i < coordinates.size(); ++i) {
        const Point3D&      point  = coordinates[i];
        InterpolationResult result = cpu_interp.query(point);

        ASSERT_TRUE(result.valid);

        // Grid point interpolation should exactly match original data
        const MagneticFieldData& original = field_data[i];

        // Log results for first few points
        if (i < 5) {
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "Grid point " << i << ": (" << point.x << ", " << point.y << ", " << point.z << ")"
                      << std::endl;
            std::cout << "  Original: Bx=" << original.Bx << ", By=" << original.By << ", Bz=" << original.Bz
                      << std::endl;
            std::cout << "  Interpolated: Bx=" << result.data.Bx << ", By=" << result.data.By
                      << ", Bz=" << result.data.Bz << std::endl;
            std::cout << "  Match: "
                      << (result.data.Bx == original.Bx && result.data.By == original.By &&
                                  result.data.Bz == original.Bz
                              ? "Yes"
                              : "No")
                      << std::endl
                      << std::endl;
        }

        EXPECT_FLOAT_EQ(result.data.Bx, original.Bx);
        EXPECT_FLOAT_EQ(result.data.By, original.By);
        EXPECT_FLOAT_EQ(result.data.Bz, original.Bz);
    }
}

// Test numerical stability
TEST(AccuracyTest, NumericalStability) {
    // Create large range grid
    GridParams params;
    params.origin     = Point3D(-100.0f, -100.0f, -100.0f);
    params.spacing    = Point3D(10.0f, 10.0f, 10.0f);
    params.dimensions = {21, 21, 21};  // From -100 to +100
    params.update_bounds();

    RegularGrid3D grid(params);

    // Set simple function: Bx = 1, By = 1, Bz = 1
    auto& field_data = const_cast<std::vector<MagneticFieldData>&>(grid.getFieldData());
    for (size_t i = 0; i < field_data.size(); ++i) {
        field_data[i] = MagneticFieldData(1.0f, 1.0f, 1.0f,   // B = (1, 1, 1)
                                          0.0f, 0.0f, 0.0f,   // dBx_dx, dBx_dy, dBx_dz
                                          0.0f, 0.0f, 0.0f,   // dBy_dx, dBy_dy, dBy_dz
                                          0.0f, 0.0f, 0.0f);  // dBz_dx, dBz_dy, dBz_dz
    }

    CPUInterpolator cpu_interp(grid);

    // Test points far from origin
    std::vector<Point3D> test_points = {{50.0f, 25.0f, -75.0f}, {-30.0f, 80.0f, 10.0f}, {15.0f, -45.0f, 60.0f}};

    for (const auto& point : test_points) {
        InterpolationResult result = cpu_interp.query(point);
        ASSERT_TRUE(result.valid);

        // Log results
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Stability test point: (" << point.x << ", " << point.y << ", " << point.z << ")" << std::endl;
        std::cout << "  Expected: Bx=1.0, By=1.0, Bz=1.0" << std::endl;
        std::cout << "  Interpolated: Bx=" << result.data.Bx << ", By=" << result.data.By << ", Bz=" << result.data.Bz
                  << std::endl;
        std::cout << "  Errors: Bx=" << std::abs(result.data.Bx - 1.0f) << ", By=" << std::abs(result.data.By - 1.0f)
                  << ", Bz=" << std::abs(result.data.Bz - 1.0f) << std::endl
                  << std::endl;

        // Check numerical stability
        EXPECT_NEAR(result.data.Bx, 1.0f, 1e-2f);
        EXPECT_NEAR(result.data.By, 1.0f, 1e-2f);
        EXPECT_NEAR(result.data.Bz, 1.0f, 1e-2f);
    }
}

// Test consistency between batch and single point interpolation
TEST(AccuracyTest, BatchVsSingleConsistency) {
    // Create test grid
    GridParams params;
    params.origin     = Point3D(0.0f, 0.0f, 0.0f);
    params.spacing    = Point3D(1.0f, 1.0f, 1.0f);
    params.dimensions = {4, 4, 4};
    params.update_bounds();

    RegularGrid3D grid(params);

    // Set test data
    auto& field_data = const_cast<std::vector<MagneticFieldData>&>(grid.getFieldData());
    for (auto& data : field_data) {
        data = MagneticFieldData(
            static_cast<float>(rand()) / RAND_MAX * 2.0f, static_cast<float>(rand()) / RAND_MAX * 2.0f,
            static_cast<float>(rand()) / RAND_MAX * 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }

    CPUInterpolator cpu_interp(grid);

    // Generate test points
    std::vector<Point3D> test_points;
    for (int i = 0; i < 50; ++i) {
        test_points.push_back(Point3D(static_cast<float>(rand()) / RAND_MAX * 3.0f,
                                      static_cast<float>(rand()) / RAND_MAX * 3.0f,
                                      static_cast<float>(rand()) / RAND_MAX * 3.0f));
    }

    // Single point query
    std::vector<InterpolationResult> single_results;
    for (const auto& point : test_points) {
        single_results.push_back(cpu_interp.query(point));
    }

    // Batch query
    std::vector<InterpolationResult> batch_results(test_points.size());
    batch_results = cpu_interp.queryBatch(test_points);

    // Compare results
    ASSERT_EQ(single_results.size(), batch_results.size());
    for (size_t i = 0; i < single_results.size(); ++i) {
        const auto& single = single_results[i];
        const auto& batch  = batch_results[i];

        // Log first few comparisons
        if (i < 5) {
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "Batch vs Single comparison " << i << ":" << std::endl;
            std::cout << "  Single: valid=" << single.valid << ", Bx=" << single.data.Bx << ", By=" << single.data.By
                      << ", Bz=" << single.data.Bz << std::endl;
            std::cout << "  Batch:  valid=" << batch.valid << ", Bx=" << batch.data.Bx << ", By=" << batch.data.By
                      << ", Bz=" << batch.data.Bz << std::endl;
            std::cout << "  Match: "
                      << (single.valid == batch.valid && (!single.valid || (single.data.Bx == batch.data.Bx &&
                                                                            single.data.By == batch.data.By &&
                                                                            single.data.Bz == batch.data.Bz))
                              ? "Yes"
                              : "No")
                      << std::endl
                      << std::endl;
        }

        EXPECT_EQ(single.valid, batch.valid);

        if (single.valid && batch.valid) {
            EXPECT_FLOAT_EQ(single.data.Bx, batch.data.Bx);
            EXPECT_FLOAT_EQ(single.data.By, batch.data.By);
            EXPECT_FLOAT_EQ(single.data.Bz, batch.data.Bz);
        }
    }
}

}  // namespace test
}  // namespace p3d
#include <gtest/gtest.h>
#include "point3d_interp/cpu_interpolator.h"
#include "point3d_interp/grid_structure.h"
#include <vector>
#include <cmath>

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

    // Set test data for known function: f(x,y,z) = x^2 + y^2 + z^2
    // Gradient: (2x, 2y, 2z)
    auto& field_data = const_cast<std::vector<MagneticFieldData>&>(grid.getFieldData());
    for (size_t i = 0; i < field_data.size(); ++i) {
        const auto& coord = grid.getCoordinates()[i];
        float       x = coord.x, y = coord.y, z = coord.z;
        field_data[i] = MagneticFieldData(x * x + y * y + z * z,  // B = x² + y² + z²
                                          2 * x, 2 * y, 2 * z     // Gradient = (2x, 2y, 2z)
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
        float expected_B  = x * x + y * y + z * z;
        float expected_Bx = 2 * x;
        float expected_By = 2 * y;
        float expected_Bz = 2 * z;

        // Check interpolation accuracy (trilinear interpolation should have good accuracy)
        EXPECT_NEAR(result.data.field_strength, expected_B, 1e-3f);
        EXPECT_NEAR(result.data.gradient_x, expected_Bx, 1e-3f);
        EXPECT_NEAR(result.data.gradient_y, expected_By, 1e-3f);
        EXPECT_NEAR(result.data.gradient_z, expected_Bz, 1e-3f);
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

    // Set linear field: f(x,y,z) = x + 2*y + 3*z + 1
    auto& field_data = const_cast<std::vector<MagneticFieldData>&>(grid.getFieldData());
    for (size_t i = 0; i < field_data.size(); ++i) {
        const auto& coord = grid.getCoordinates()[i];
        float       x = coord.x, y = coord.y, z = coord.z;
        field_data[i] = MagneticFieldData(x + 2 * y + 3 * z + 1,  // B = x + 2y + 3z + 1
                                          1.0f, 2.0f, 3.0f        // Gradient = (1, 2, 3)
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

        // Calculate analytical solution
        float expected_B = point.x + 2 * point.y + 3 * point.z + 1;

        // For linear fields, trilinear interpolation should be exact
        EXPECT_NEAR(result.data.field_strength, expected_B, 1e-6f);
        EXPECT_NEAR(result.data.gradient_x, 1.0f, 1e-6f);
        EXPECT_NEAR(result.data.gradient_y, 2.0f, 1e-6f);
        EXPECT_NEAR(result.data.gradient_z, 3.0f, 1e-6f);
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
                                 static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX);
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
        EXPECT_FLOAT_EQ(result.data.field_strength, original.field_strength);
        EXPECT_FLOAT_EQ(result.data.gradient_x, original.gradient_x);
        EXPECT_FLOAT_EQ(result.data.gradient_y, original.gradient_y);
        EXPECT_FLOAT_EQ(result.data.gradient_z, original.gradient_z);
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

    // Set simple function
    auto& field_data = const_cast<std::vector<MagneticFieldData>&>(grid.getFieldData());
    for (size_t i = 0; i < field_data.size(); ++i) {
        const auto& coord = grid.getCoordinates()[i];
        field_data[i]     = MagneticFieldData(coord.x + coord.y + coord.z,  // B = x + y + z
                                              1.0f, 1.0f, 1.0f);
    }

    CPUInterpolator cpu_interp(grid);

    // Test points far from origin
    std::vector<Point3D> test_points = {{50.0f, 25.0f, -75.0f}, {-30.0f, 80.0f, 10.0f}, {15.0f, -45.0f, 60.0f}};

    for (const auto& point : test_points) {
        InterpolationResult result = cpu_interp.query(point);
        ASSERT_TRUE(result.valid);

        // Calculate analytical solution
        float expected_B = point.x + point.y + point.z;

        // Check numerical stability
        EXPECT_NEAR(result.data.field_strength, expected_B, 1e-2f);  // Relaxed precision requirement
        EXPECT_NEAR(result.data.gradient_x, 1.0f, 1e-2f);
        EXPECT_NEAR(result.data.gradient_y, 1.0f, 1e-2f);
        EXPECT_NEAR(result.data.gradient_z, 1.0f, 1e-2f);
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
        data = MagneticFieldData(static_cast<float>(rand()) / RAND_MAX * 10.0f, static_cast<float>(rand()) / RAND_MAX * 2.0f,
                                 static_cast<float>(rand()) / RAND_MAX * 2.0f, static_cast<float>(rand()) / RAND_MAX * 2.0f);
    }

    CPUInterpolator cpu_interp(grid);

    // Generate test points
    std::vector<Point3D> test_points;
    for (int i = 0; i < 50; ++i) {
        test_points.push_back(Point3D(static_cast<float>(rand()) / RAND_MAX * 3.0f, static_cast<float>(rand()) / RAND_MAX * 3.0f,
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

        EXPECT_EQ(single.valid, batch.valid);

        if (single.valid && batch.valid) {
            EXPECT_FLOAT_EQ(single.data.field_strength, batch.data.field_strength);
            EXPECT_FLOAT_EQ(single.data.gradient_x, batch.data.gradient_x);
            EXPECT_FLOAT_EQ(single.data.gradient_y, batch.data.gradient_y);
            EXPECT_FLOAT_EQ(single.data.gradient_z, batch.data.gradient_z);
        }
    }
}

}  // namespace test
}  // namespace p3d
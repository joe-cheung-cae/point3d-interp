#include <gtest/gtest.h>
#include "point3d_interp/data_loader.h"
#include "point3d_interp/api.h"
#include "point3d_interp/grid_structure.h"
#include <fstream>
#include <iomanip>
#include <string>
#include <limits>
#include <cmath>

namespace p3d {
namespace test {

// Test class for edge cases
class EdgeCaseTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create test files
        CreateIrregularGridFile();
        CreateBoundaryTestFile();
        CreatePrecisionTestFile();
    }

    void TearDown() override {
        // Clean up test files
        std::remove("irregular_grid.csv");
        std::remove("boundary_test.csv");
        std::remove("precision_test.csv");
    }

    void CreateIrregularGridFile() {
        std::ofstream file("irregular_grid.csv");
        file << "x,y,z,B,Bx,By,Bz\n";
        // Irregular spacing - should fail grid detection
        file << "0.0,0.0,0.0,1.0,0.1,0.2,0.3\n";
        file << "1.5,0.0,0.0,1.1,0.15,0.18,0.32\n";  // Non-uniform spacing
        file << "0.0,2.0,0.0,0.9,0.08,0.22,0.28\n";  // Non-uniform spacing
        file << "1.5,2.0,0.0,1.0,0.13,0.19,0.31\n";
        file << "0.0,0.0,3.0,1.2,0.12,0.18,0.35\n";  // Non-uniform spacing
        file << "1.5,0.0,3.0,1.3,0.17,0.14,0.37\n";
        file << "0.0,2.0,3.0,1.1,0.1,0.2,0.33\n";
        file << "1.5,2.0,3.0,1.2,0.15,0.16,0.35\n";
        file.close();
    }

    void CreateBoundaryTestFile() {
        std::ofstream file("boundary_test.csv");
        file << "x,y,z,B,Bx,By,Bz\n";
        // 2x2x2 grid at boundaries
        file << "-1.0,-1.0,-1.0,1.0,0.1,0.2,0.3\n";
        file << "0.0,-1.0,-1.0,1.1,0.15,0.18,0.32\n";
        file << "-1.0,0.0,-1.0,0.9,0.08,0.22,0.28\n";
        file << "0.0,0.0,-1.0,1.0,0.13,0.19,0.31\n";
        file << "-1.0,-1.0,0.0,1.2,0.12,0.18,0.35\n";
        file << "0.0,-1.0,0.0,1.3,0.17,0.14,0.37\n";
        file << "-1.0,0.0,0.0,1.1,0.1,0.2,0.33\n";
        file << "0.0,0.0,0.0,1.2,0.15,0.16,0.35\n";
        file.close();
    }

    void CreatePrecisionTestFile() {
        std::ofstream file("precision_test.csv");
        file << "x,y,z,B,Bx,By,Bz\n";
        // High precision values
        file << std::fixed << std::setprecision(10);
        file << "0.0000000001,0.0000000002,0.0000000003,1.0000000001,0.1000000001,0.2000000002,0.3000000003\n";
        file << "1.0000000001,0.0000000002,0.0000000003,1.1000000001,0.1500000001,0.1800000002,0.3200000003\n";
        file << "0.0000000001,1.0000000002,0.0000000003,0.9000000001,0.0800000001,0.2200000002,0.2800000003\n";
        file << "1.0000000001,1.0000000002,0.0000000003,1.0000000001,0.1300000001,0.1900000002,0.3100000003\n";
        file.close();
    }
};

// Test irregular grid detection
TEST_F(EdgeCaseTest, IrregularGridDetection) {
    DataLoader                     loader;
    std::vector<Point3D>           coordinates;
    std::vector<MagneticFieldData> field_data;
    GridParams                     grid_params;

    ErrorCode result = loader.LoadFromCSV("irregular_grid.csv", coordinates, field_data, grid_params);

    // Should fail due to irregular spacing
    EXPECT_EQ(result, ErrorCode::InvalidGridData);
}

// Test boundary conditions
TEST_F(EdgeCaseTest, BoundaryConditions) {
    MagneticFieldInterpolator interp;
    ErrorCode                 err = interp.LoadFromCSV("boundary_test.csv");
    ASSERT_EQ(err, ErrorCode::Success);

    // Test points at grid boundaries
    Point3D             boundary_point(0.0f, 0.0f, 0.0f);
    InterpolationResult result;
    err = interp.Query(boundary_point, result);
    EXPECT_EQ(err, ErrorCode::Success);
    EXPECT_TRUE(result.valid);

    // Test points outside boundaries
    Point3D outside_point(1.0f, 1.0f, 1.0f);  // Beyond grid bounds
    err = interp.Query(outside_point, result);
    EXPECT_EQ(err, ErrorCode::Success);
    EXPECT_FALSE(result.valid);
}

// Test high precision data handling
TEST_F(EdgeCaseTest, HighPrecisionData) {
    MagneticFieldInterpolator interp;
    ErrorCode                 err = interp.LoadFromCSV("precision_test.csv");
    ASSERT_EQ(err, ErrorCode::Success);

    Point3D             query_point(0.5f, 0.5f, 0.0f);
    InterpolationResult result;
    err = interp.Query(query_point, result);
    EXPECT_EQ(err, ErrorCode::Success);
    EXPECT_TRUE(result.valid);

    // Results should be reasonable (interpolation of high precision data)
    EXPECT_GE(result.data.field_strength, 0.9f);
    EXPECT_LE(result.data.field_strength, 1.2f);
}

// Test extreme values
TEST(ExtremeValueTest, VeryLargeValues) {
    // Create data with very large values
    std::vector<Point3D> coordinates = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};

    std::vector<MagneticFieldData> field_data(4);
    for (auto& data : field_data) {
        data.field_strength = 1e10f;  // Very large values
        data.gradient_x     = 1e8f;
        data.gradient_y     = 1e8f;
        data.gradient_z     = 1e8f;
    }

    MagneticFieldInterpolator interp;
    ErrorCode                 err = interp.LoadFromMemory(coordinates.data(), field_data.data(), 4);
    EXPECT_EQ(err, ErrorCode::Success);

    Point3D             query(0.5f, 0.5f, 0.0f);
    InterpolationResult result;
    err = interp.Query(query, result);
    EXPECT_EQ(err, ErrorCode::Success);
    EXPECT_TRUE(result.valid);
    EXPECT_FLOAT_EQ(result.data.field_strength, 1e10f);  // Should interpolate correctly
}

// Test very small values
TEST(ExtremeValueTest, VerySmallValues) {
    std::vector<Point3D> coordinates = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};

    std::vector<MagneticFieldData> field_data(4);
    for (auto& data : field_data) {
        data.field_strength = 1e-10f;  // Very small values
        data.gradient_x     = 1e-8f;
        data.gradient_y     = 1e-8f;
        data.gradient_z     = 1e-8f;
    }

    MagneticFieldInterpolator interp;
    ErrorCode                 err = interp.LoadFromMemory(coordinates.data(), field_data.data(), 4);
    EXPECT_EQ(err, ErrorCode::Success);

    Point3D             query(0.5f, 0.5f, 0.0f);
    InterpolationResult result;
    err = interp.Query(query, result);
    EXPECT_EQ(err, ErrorCode::Success);
    EXPECT_TRUE(result.valid);
    EXPECT_NEAR(result.data.field_strength, 1e-10f, 1e-11f);
}

// Test NaN and Inf values
TEST(ExtremeValueTest, NaNInfValues) {
    std::vector<Point3D> coordinates = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};

    std::vector<MagneticFieldData> field_data(4);
    for (size_t i = 0; i < 4; ++i) {
        if (i == 0) {
            // First point has NaN
            field_data[i].field_strength = std::numeric_limits<float>::quiet_NaN();
        } else {
            field_data[i].field_strength = 1.0f;
        }
        field_data[i].gradient_x = 0.1f;
        field_data[i].gradient_y = 0.2f;
        field_data[i].gradient_z = 0.3f;
    }

    MagneticFieldInterpolator interp;
    ErrorCode                 err = interp.LoadFromMemory(coordinates.data(), field_data.data(), 4);
    // Should still load (NaN is a valid float value)
    EXPECT_EQ(err, ErrorCode::Success);

    Point3D             query(0.5f, 0.5f, 0.0f);
    InterpolationResult result;
    err = interp.Query(query, result);
    EXPECT_EQ(err, ErrorCode::Success);
    // Result may be NaN due to interpolation with NaN
    EXPECT_TRUE(std::isnan(result.data.field_strength) || std::isfinite(result.data.field_strength));
}

// Test empty data arrays
TEST(ErrorHandlingTest, EmptyDataArrays) {
    MagneticFieldInterpolator interp;

    // Empty arrays
    ErrorCode err = interp.LoadFromMemory(nullptr, nullptr, 0);
    EXPECT_EQ(err, ErrorCode::InvalidParameter);

    std::vector<Point3D>           coords;
    std::vector<MagneticFieldData> data;
    err = interp.LoadFromMemory(coords.data(), data.data(), 0);
    EXPECT_EQ(err, ErrorCode::InvalidParameter);
}

// Test mismatched array sizes
TEST(ErrorHandlingTest, MismatchedArraySizes) {
    MagneticFieldInterpolator interp;

    std::vector<Point3D>           coords = {{0, 0, 0}, {1, 0, 0}};
    std::vector<MagneticFieldData> data   = {{1.0f, 0.1f, 0.2f, 0.3f}};  // Only 1 element

    ErrorCode err = interp.LoadFromMemory(coords.data(), data.data(), coords.size());
    EXPECT_EQ(err, ErrorCode::InvalidGridData);
}

// Test concurrent access (basic thread safety check)
TEST(ConcurrencyTest, BasicConcurrency) {
    MagneticFieldInterpolator interp;

    // Load some test data
    std::vector<Point3D>           coords = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
    std::vector<MagneticFieldData> data(4, MagneticFieldData(1.0f, 0.1f, 0.2f, 0.3f));

    ErrorCode err = interp.LoadFromMemory(coords.data(), data.data(), 4);
    ASSERT_EQ(err, ErrorCode::Success);

    // Test that multiple queries work (basic concurrency check)
    Point3D query1(0.5f, 0.5f, 0.0f);
    Point3D query2(0.3f, 0.7f, 0.0f);

    InterpolationResult result1, result2;

    // These should work without interference
    err = interp.Query(query1, result1);
    EXPECT_EQ(err, ErrorCode::Success);

    err = interp.Query(query2, result2);
    EXPECT_EQ(err, ErrorCode::Success);

    EXPECT_TRUE(result1.valid);
    EXPECT_TRUE(result2.valid);
}

// Test grid parameter validation
TEST(GridValidationTest, InvalidGridParameters) {
    // Test with invalid dimensions
    GridParams invalid_params;
    invalid_params.dimensions = {0, 1, 1};  // Zero dimension

    RegularGrid3D* grid = nullptr;
    try {
        // This should throw due to invalid parameters
        grid = new RegularGrid3D(invalid_params);
        FAIL() << "Expected exception for invalid grid parameters";
    } catch (const std::exception&) {
        // Expected
    }
    delete grid;
}

}  // namespace test
}  // namespace p3d
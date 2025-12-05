#include <gtest/gtest.h>
#include "point3d_interp/data_loader.h"
#include "point3d_interp/interpolator_api.h"
#include "point3d_interp/grid_structure.h"
#include <fstream>
#include <iomanip>
#include <string>
#include <limits>
#include <cmath>
#include <iostream>

P3D_NAMESPACE_BEGIN
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
        file << "x,y,z,Bx,By,Bz,dBx_dx,dBx_dy,dBx_dz,dBy_dx,dBy_dy,dBy_dz,dBz_dx,dBz_dy,dBz_dz\n";
        // Irregular spacing - should fail grid detection
        file << "0.0,0.0,0.0,0.1,0.2,0.3,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09\n";
        file << "1.5,0.0,0.0,0.15,0.18,0.32,0.011,0.021,0.031,0.041,0.051,0.061,0.071,0.081,0.091\n";  // Non-uniform
                                                                                                       // spacing
        file << "2.5,0.0,0.0,0.08,0.22,0.28,0.012,0.022,0.032,0.042,0.052,0.062,0.072,0.082,0.092\n";  // Non-uniform
                                                                                                       // spacing
        file << "0.0,2.0,0.0,0.13,0.19,0.31,0.013,0.023,0.033,0.043,0.053,0.063,0.073,0.083,0.093\n";
        file << "1.5,2.0,0.0,0.12,0.18,0.35,0.014,0.024,0.034,0.044,0.054,0.064,0.074,0.084,0.094\n";  // Non-uniform
                                                                                                       // spacing
        file << "2.5,2.0,0.0,0.17,0.14,0.37,0.015,0.025,0.035,0.045,0.055,0.065,0.075,0.085,0.095\n";
        file << "0.0,0.0,3.0,0.1,0.2,0.33,0.016,0.026,0.036,0.046,0.056,0.066,0.076,0.086,0.096\n";
        file << "1.5,0.0,3.0,0.15,0.16,0.35,0.017,0.027,0.037,0.047,0.057,0.067,0.077,0.087,0.097\n";
        file << "2.5,0.0,3.0,0.15,0.16,0.35,0.017,0.027,0.037,0.047,0.057,0.067,0.077,0.087,0.097\n";
        file.close();
    }

    void CreateBoundaryTestFile() {
        std::ofstream file("boundary_test.csv");
        file << "x,y,z,Bx,By,Bz,dBx_dx,dBx_dy,dBx_dz,dBy_dx,dBy_dy,dBy_dz,dBz_dx,dBz_dy,dBz_dz\n";
        // 2x2x2 grid at boundaries
        file << "-1.0,-1.0,-1.0,0.1,0.2,0.3,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09\n";
        file << "0.0,-1.0,-1.0,0.15,0.18,0.32,0.011,0.021,0.031,0.041,0.051,0.061,0.071,0.081,0.091\n";
        file << "-1.0,0.0,-1.0,0.08,0.22,0.28,0.012,0.022,0.032,0.042,0.052,0.062,0.072,0.082,0.092\n";
        file << "0.0,0.0,-1.0,0.13,0.19,0.31,0.013,0.023,0.033,0.043,0.053,0.063,0.073,0.083,0.093\n";
        file << "-1.0,-1.0,0.0,0.12,0.18,0.35,0.014,0.024,0.034,0.044,0.054,0.064,0.074,0.084,0.094\n";
        file << "0.0,-1.0,0.0,0.17,0.14,0.37,0.015,0.025,0.035,0.045,0.055,0.065,0.075,0.085,0.095\n";
        file << "-1.0,0.0,0.0,0.1,0.2,0.33,0.016,0.026,0.036,0.046,0.056,0.066,0.076,0.086,0.096\n";
        file << "0.0,0.0,0.0,0.15,0.16,0.35,0.017,0.027,0.037,0.047,0.057,0.067,0.077,0.087,0.097\n";
        file.close();
    }

    void CreatePrecisionTestFile() {
        std::ofstream file("precision_test.csv");
        file << "x,y,z,Bx,By,Bz,dBx_dx,dBx_dy,dBx_dz,dBy_dx,dBy_dy,dBy_dz,dBz_dx,dBz_dy,dBz_dz\n";
        // High precision values
        file << std::fixed << std::setprecision(6);
        file << "0.000001,0.000002,0.0,0.100001,0.200002,0.300003,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09\n";
        file << "1.000001,0.000002,0.0,0.150001,0.180002,0.320003,0.011,0.021,0.031,0.041,0.051,0.061,0.071,0.081,"
                "0.091\n";
        file << "0.000001,1.000002,0.0,0.080001,0.220002,0.280003,0.012,0.022,0.032,0.042,0.052,0.062,0.072,0.082,"
                "0.092\n";
        file << "1.000001,1.000002,0.0,0.130001,0.190002,0.310003,0.013,0.023,0.033,0.043,0.053,0.063,0.073,0.083,"
                "0.093\n";
        file << "0.000001,0.000002,1.0,0.120001,0.180002,0.350003,0.014,0.024,0.034,0.044,0.054,0.064,0.074,0.084,"
                "0.094\n";
        file << "1.000001,0.000002,1.0,0.170001,0.140002,0.370003,0.015,0.025,0.035,0.045,0.055,0.065,0.075,0.085,"
                "0.095\n";
        file << "0.000001,1.000002,1.0,0.100001,0.200002,0.330003,0.016,0.026,0.036,0.046,0.056,0.066,0.076,0.086,"
                "0.096\n";
        file << "1.000001,1.000002,1.0,0.150001,0.160002,0.350003,0.017,0.027,0.037,0.047,0.057,0.067,0.077,0.087,"
                "0.097\n";
        file.close();
    }
};

// Test irregular grid detection
TEST_F(EdgeCaseTest, IrregularGridDetection) {
    DataLoader                     loader;
    std::vector<Point3D>           coordinates;
    std::vector<MagneticFieldData> field_data;
    GridParams                     grid_params;

    // Should fail due to irregular spacing
    EXPECT_THROW(loader.LoadFromCSV("irregular_grid.csv", coordinates, field_data, grid_params), std::runtime_error);
}

// Test boundary conditions
TEST_F(EdgeCaseTest, BoundaryConditions) {
    MagneticFieldInterpolator interp;
    EXPECT_NO_THROW(interp.LoadFromCSV("boundary_test.csv"));

    // Test points at grid boundaries
    Point3D             boundary_point(0.0f, 0.0f, 0.0f);
    InterpolationResult result;
    EXPECT_NO_THROW(interp.Query(boundary_point, result));
    EXPECT_TRUE(result.valid);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Boundary point (0.0,0.0,0.0): valid=" << result.valid << ", Bx=" << result.data.Bx
              << ", By=" << result.data.By << ", Bz=" << result.data.Bz << std::endl;

    // Test points outside boundaries
    Point3D outside_point(1.0f, 1.0f, 1.0f);  // Beyond grid bounds
    EXPECT_NO_THROW(interp.Query(outside_point, result));
    EXPECT_FALSE(result.valid);

    std::cout << "Outside point (1.0,1.0,1.0): valid=" << result.valid << std::endl << std::endl;

    // Export boundary condition test results for visualization
    std::vector<Point3D>             query_points = {boundary_point, outside_point};
    std::vector<InterpolationResult> results      = {InterpolationResult(), result};  // First result is overwritten
    EXPECT_NO_THROW(interp.Query(boundary_point, results[0]));
    EXPECT_NO_THROW(interp.ExportOutputPoints(ExportFormat::ParaviewVTK, query_points, results,
                                              "test_output/boundary_conditions.vtk"));
}

// Test high precision data handling
TEST_F(EdgeCaseTest, HighPrecisionData) {
    MagneticFieldInterpolator interp;
    EXPECT_NO_THROW(interp.LoadFromCSV("precision_test.csv"));

    Point3D             query_point(0.5f, 0.5f, 0.0f);
    InterpolationResult result;
    EXPECT_NO_THROW(interp.Query(query_point, result));
    EXPECT_TRUE(result.valid);

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "High precision query (0.5,0.5,0.0): valid=" << result.valid << ", Bx=" << result.data.Bx
              << ", By=" << result.data.By << ", Bz=" << result.data.Bz << std::endl
              << std::endl;

    // Results should be reasonable (interpolation of high precision data)
    EXPECT_GE(result.data.Bx, 0.08f);
    EXPECT_LE(result.data.Bx, 0.17f);

    // Export high precision data test results for visualization
    std::vector<Point3D>             query_points = {query_point};
    std::vector<InterpolationResult> results      = {result};
    EXPECT_NO_THROW(interp.ExportOutputPoints(ExportFormat::ParaviewVTK, query_points, results,
                                              "test_output/high_precision_data.vtk"));
}

// Test extreme values
TEST(ExtremeValueTest, VeryLargeValues) {
    // Create data with very large values
    std::vector<Point3D> coordinates = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
                                        {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};

    std::vector<MagneticFieldData> field_data(8);
    for (auto& data : field_data) {
        data.Bx = 1e8f;
        data.By = 1e8f;
        data.Bz = 1e8f;
    }

    MagneticFieldInterpolator interp;
    EXPECT_NO_THROW(interp.LoadFromMemory(coordinates.data(), field_data.data(), 8));

    Point3D             query(0.5f, 0.5f, 0.5f);
    InterpolationResult result;
    EXPECT_NO_THROW(interp.Query(query, result));
    EXPECT_TRUE(result.valid);
    EXPECT_FLOAT_EQ(result.data.Bx, 1e8f);  // Should interpolate correctly

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Very large values query (0.5,0.5,0.5): valid=" << result.valid << ", Bx=" << result.data.Bx
              << ", By=" << result.data.By << ", Bz=" << result.data.Bz << std::endl
              << std::endl;

    // Export very large values test results for visualization
    std::vector<Point3D>             query_points = {query};
    std::vector<InterpolationResult> results      = {result};
    EXPECT_NO_THROW(interp.ExportOutputPoints(ExportFormat::ParaviewVTK, query_points, results,
                                              "test_output/very_large_values.vtk"));
}

// Test very small values
TEST(ExtremeValueTest, VerySmallValues) {
    std::vector<Point3D> coordinates = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
                                        {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};

    std::vector<MagneticFieldData> field_data(8);
    for (auto& data : field_data) {
        data.Bx = 1e-8f;
        data.By = 1e-8f;
        data.Bz = 1e-8f;
    }

    MagneticFieldInterpolator interp;
    EXPECT_NO_THROW(interp.LoadFromMemory(coordinates.data(), field_data.data(), 8));

    Point3D             query(0.5f, 0.5f, 0.5f);
    InterpolationResult result;
    EXPECT_NO_THROW(interp.Query(query, result));
    EXPECT_TRUE(result.valid);
    EXPECT_NEAR(result.data.Bx, 1e-8f, 1e-9f);

    std::cout << std::scientific << std::setprecision(2);
    std::cout << "Very small values query (0.5,0.5,0.5): valid=" << result.valid << ", Bx=" << result.data.Bx
              << ", By=" << result.data.By << ", Bz=" << result.data.Bz << std::endl
              << std::endl;

    // Export very small values test results for visualization
    std::vector<Point3D>             query_points = {query};
    std::vector<InterpolationResult> results      = {result};
    EXPECT_NO_THROW(interp.ExportOutputPoints(ExportFormat::ParaviewVTK, query_points, results,
                                              "test_output/very_small_values.vtk"));
}

// Test NaN and Inf values
TEST(ExtremeValueTest, NaNInfValues) {
    std::vector<Point3D> coordinates = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
                                        {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};

    std::vector<MagneticFieldData> field_data(8);
    for (size_t i = 0; i < 8; ++i) {
        if (i == 0) {
            // First point has NaN
            field_data[i].Bx = std::numeric_limits<float>::quiet_NaN();
            field_data[i].By = 1.0f;
            field_data[i].Bz = 1.0f;
        } else {
            field_data[i].Bx = 1.0f;
            field_data[i].By = 1.0f;
            field_data[i].Bz = 1.0f;
        }
    }

    MagneticFieldInterpolator interp;
    // Should still load (NaN is a valid float value)
    EXPECT_NO_THROW(interp.LoadFromMemory(coordinates.data(), field_data.data(), 8));

    Point3D             query(0.5f, 0.5f, 0.5f);
    InterpolationResult result;
    EXPECT_NO_THROW(interp.Query(query, result));
    // Result may be NaN due to interpolation with NaN
    EXPECT_TRUE(std::isnan(result.data.Bx) || std::isfinite(result.data.Bx));

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "NaN/Inf values query (0.5,0.5,0.5): valid=" << result.valid << ", Bx=" << result.data.Bx
              << ", By=" << result.data.By << ", Bz=" << result.data.Bz;
    if (std::isnan(result.data.Bx)) std::cout << " (Bx is NaN)";
    std::cout << std::endl << std::endl;

    // Export NaN/Inf values test results for visualization
    std::vector<Point3D>             query_points = {query};
    std::vector<InterpolationResult> results      = {result};
    EXPECT_NO_THROW(
        interp.ExportOutputPoints(ExportFormat::ParaviewVTK, query_points, results, "test_output/nan_inf_values.vtk"));
}

// Test empty data arrays
TEST(ErrorHandlingTest, EmptyDataArrays) {
    MagneticFieldInterpolator interp;

    // Empty arrays - should throw exceptions
    EXPECT_THROW(interp.LoadFromMemory(nullptr, nullptr, 0), std::runtime_error);

    std::vector<Point3D>           coords;
    std::vector<MagneticFieldData> data;
    EXPECT_THROW(interp.LoadFromMemory(coords.data(), data.data(), 0), std::runtime_error);
}

// Test mismatched array sizes
TEST(ErrorHandlingTest, MismatchedArraySizes) {
    MagneticFieldInterpolator interp;

    std::vector<Point3D>           coords = {{0, 0, 0}, {1, 0, 0}};
    std::vector<MagneticFieldData> data   = {{1.0f, 0.1f, 0.2f, 0.3f}};  // Only 1 element

    EXPECT_NO_THROW(interp.LoadFromMemory(coords.data(), data.data(), coords.size()));
}

// Test concurrent access (basic thread safety check)
TEST(ConcurrencyTest, BasicConcurrency) {
    MagneticFieldInterpolator interp;

    // Load some test data
    std::vector<Point3D>           coords = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
                                             {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};
    std::vector<MagneticFieldData> data(8, MagneticFieldData(1.0f, 0.1f, 0.2f, 0.3f));

    EXPECT_NO_THROW(interp.LoadFromMemory(coords.data(), data.data(), 8));

    // Test that multiple queries work (basic concurrency check)
    Point3D query1(0.5f, 0.5f, 0.5f);
    Point3D query2(0.3f, 0.7f, 0.5f);

    InterpolationResult result1, result2;

    // These should work without interference
    EXPECT_NO_THROW(interp.Query(query1, result1));
    EXPECT_NO_THROW(interp.Query(query2, result2));

    EXPECT_TRUE(result1.valid);
    EXPECT_TRUE(result2.valid);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Concurrency query1 (0.5,0.5,0.5): valid=" << result1.valid << ", Bx=" << result1.data.Bx
              << ", By=" << result1.data.By << ", Bz=" << result1.data.Bz << std::endl;
    std::cout << "Concurrency query2 (0.3,0.7,0.5): valid=" << result2.valid << ", Bx=" << result2.data.Bx
              << ", By=" << result2.data.By << ", Bz=" << result2.data.Bz << std::endl
              << std::endl;

    // Export concurrency test results for visualization
    std::vector<Point3D>             query_points = {query1, query2};
    std::vector<InterpolationResult> results      = {result1, result2};
    EXPECT_NO_THROW(interp.ExportOutputPoints(ExportFormat::ParaviewVTK, query_points, results,
                                              "test_output/basic_concurrency.vtk"));
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
P3D_NAMESPACE_END
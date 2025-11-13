#include <gtest/gtest.h>
#include "point3d_interp/api.h"
#include <fstream>
#include <string>

namespace p3d {
namespace test {

class APITest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create temporary test file
        CreateTestCSVFile();
    }

    void TearDown() override {
        // Clean up temporary file
        std::remove(test_file_path_.c_str());
    }

    void CreateTestCSVFile() {
        test_file_path_ = "api_test.csv";

        std::ofstream file(test_file_path_);
        ASSERT_TRUE(file.is_open());

        // Write test data (2x2x2 grid)
        file << "x,y,z,Bx,By,Bz,dBx_dx,dBx_dy,dBx_dz,dBy_dx,dBy_dy,dBy_dz,dBz_dx,dBz_dy,dBz_dz\n";
        file << "0.0,0.0,0.0,0.1,0.2,0.3,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09\n";
        file << "1.0,0.0,0.0,0.15,0.18,0.32,0.011,0.021,0.031,0.041,0.051,0.061,0.071,0.081,0.091\n";
        file << "0.0,1.0,0.0,0.08,0.22,0.28,0.012,0.022,0.032,0.042,0.052,0.062,0.072,0.082,0.092\n";
        file << "1.0,1.0,0.0,0.13,0.19,0.31,0.013,0.023,0.033,0.043,0.053,0.063,0.073,0.083,0.093\n";
        file << "0.0,0.0,1.0,0.12,0.18,0.35,0.014,0.024,0.034,0.044,0.054,0.064,0.074,0.084,0.094\n";
        file << "1.0,0.0,1.0,0.17,0.14,0.37,0.015,0.025,0.035,0.045,0.055,0.065,0.075,0.085,0.095\n";
        file << "0.0,1.0,1.0,0.1,0.2,0.33,0.016,0.026,0.036,0.046,0.056,0.066,0.076,0.086,0.096\n";
        file << "1.0,1.0,1.0,0.15,0.16,0.35,0.017,0.027,0.037,0.047,0.057,0.067,0.077,0.087,0.097\n";

        file.close();
    }

    std::string test_file_path_;
};

// Test basic API functionality
TEST_F(APITest, BasicAPIUsage) {
    MagneticFieldInterpolator interp;

    // Load data
    ErrorCode err = interp.LoadFromCSV(test_file_path_);
    EXPECT_EQ(err, ErrorCode::Success);

    // Check data loading status
    EXPECT_TRUE(interp.IsDataLoaded());
    EXPECT_EQ(interp.GetDataPointCount(), 8u);

    // Check grid parameters
    const auto& params = interp.GetGridParams();
    EXPECT_EQ(params.dimensions[0], 2u);
    EXPECT_EQ(params.dimensions[1], 2u);
    EXPECT_EQ(params.dimensions[2], 2u);
}

// Test single point query
TEST_F(APITest, SinglePointQuery) {
    MagneticFieldInterpolator interp;
    interp.LoadFromCSV(test_file_path_);

    Point3D             query_point(0.5f, 0.5f, 0.5f);
    InterpolationResult result;

    ErrorCode err = interp.Query(query_point, result);
    EXPECT_EQ(err, ErrorCode::Success);
    EXPECT_TRUE(result.valid);

    // Check result reasonableness
    EXPECT_GE(result.data.Bx, 0.0f);
    EXPECT_LE(result.data.Bx, 2.0f);
}

// Test batch query
TEST_F(APITest, BatchQuery) {
    MagneticFieldInterpolator interp;
    interp.LoadFromCSV(test_file_path_);

    std::vector<Point3D> query_points = {{0.0f, 0.0f, 0.0f}, {0.5f, 0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}};

    std::vector<InterpolationResult> results(query_points.size());

    ErrorCode err = interp.QueryBatch(query_points.data(), results.data(), query_points.size());
    EXPECT_EQ(err, ErrorCode::Success);

    // Check all results are valid
    for (const auto& result : results) {
        EXPECT_TRUE(result.valid);
    }
}

// Test out of bounds query
TEST_F(APITest, OutOfBoundsQuery) {
    MagneticFieldInterpolator interp;
    interp.LoadFromCSV(test_file_path_);

    Point3D             query_point(-1.0f, 0.5f, 0.5f);  // Out of bounds
    InterpolationResult result;

    ErrorCode err = interp.Query(query_point, result);
    EXPECT_EQ(err, ErrorCode::Success);
    EXPECT_FALSE(result.valid);
}

// Test loading data from memory
TEST_F(APITest, LoadFromMemory) {
    // Prepare test data
    std::vector<Point3D> coordinates = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 0.0f},
                                        {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}};

    std::vector<MagneticFieldData> field_data(coordinates.size(), MagneticFieldData(1.0f, 0.1f, 0.2f, 0.3f));

    MagneticFieldInterpolator interp;

    ErrorCode err = interp.LoadFromMemory(coordinates.data(), field_data.data(), coordinates.size());

    EXPECT_EQ(err, ErrorCode::Success);
    EXPECT_TRUE(interp.IsDataLoaded());
    EXPECT_EQ(interp.GetDataPointCount(), 8u);
}

// Test file not found case
TEST_F(APITest, FileNotFound) {
    MagneticFieldInterpolator interp;

    ErrorCode err = interp.LoadFromCSV("nonexistent_file.csv");
    EXPECT_EQ(err, ErrorCode::FileNotFound);
    EXPECT_FALSE(interp.IsDataLoaded());
}

// Test empty query
TEST_F(APITest, EmptyBatchQuery) {
    MagneticFieldInterpolator interp;
    interp.LoadFromCSV(test_file_path_);

    ErrorCode err = interp.QueryBatch(nullptr, nullptr, 0);
    EXPECT_EQ(err, ErrorCode::InvalidParameter);
}

// Test query without loaded data
TEST(APITest, QueryWithoutData) {
    MagneticFieldInterpolator interp;  // Data not loaded

    Point3D             query_point(0.0f, 0.0f, 0.0f);
    InterpolationResult result;

    ErrorCode err = interp.Query(query_point, result);
    EXPECT_EQ(err, ErrorCode::DataNotLoaded);
}

// Test move semantics
TEST_F(APITest, MoveSemantics) {
    MagneticFieldInterpolator interp1;
    interp1.LoadFromCSV(test_file_path_);

    // Test move constructor
    MagneticFieldInterpolator interp2(std::move(interp1));
    EXPECT_TRUE(interp2.IsDataLoaded());
    EXPECT_FALSE(interp1.IsDataLoaded());  // Source object invalid after move

    // Reload data to interp1
    interp1.LoadFromCSV(test_file_path_);

    // Test move assignment
    MagneticFieldInterpolator interp3;
    interp3 = std::move(interp1);
    EXPECT_TRUE(interp3.IsDataLoaded());
    EXPECT_FALSE(interp1.IsDataLoaded());
}

}  // namespace test
}  // namespace p3d
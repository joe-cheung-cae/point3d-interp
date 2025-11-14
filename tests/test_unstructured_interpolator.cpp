#include <gtest/gtest.h>
#include "point3d_interp/unstructured_interpolator.h"
#include "point3d_interp/types.h"
#include <vector>
#include <cmath>

namespace p3d {

class UnstructuredInterpolatorTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create a simple unstructured dataset
        coordinates_ = {
            {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {0.5f, 0.5f, 1.0f}};

        field_data_ = {
            {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},  // Bx=1, By=0, Bz=0
            {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},  // Bx=0, By=1, Bz=0
            {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},  // Bx=0, By=0, Bz=1
            {1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},  // Bx=1, By=1, Bz=1
            {0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}   // Bx=0.5, By=0.5, Bz=0.5
        };
    }

    std::vector<Point3D>           coordinates_;
    std::vector<MagneticFieldData> field_data_;
};

TEST_F(UnstructuredInterpolatorTest, Constructor) {
    EXPECT_NO_THROW({ UnstructuredInterpolator interp(coordinates_, field_data_); });
}

TEST_F(UnstructuredInterpolatorTest, ConstructorWithPower) {
    EXPECT_NO_THROW({ UnstructuredInterpolator interp(coordinates_, field_data_, 3.0f); });
}

TEST_F(UnstructuredInterpolatorTest, ConstructorWithMaxNeighbors) {
    EXPECT_NO_THROW({ UnstructuredInterpolator interp(coordinates_, field_data_, 2.0f, 3); });
}

TEST_F(UnstructuredInterpolatorTest, ConstructorMismatchedSizes) {
    std::vector<Point3D>           coords = {{0, 0, 0}};
    std::vector<MagneticFieldData> data   = {{0, 0, 0}, {0, 0, 0}};  // Different size

    EXPECT_THROW({ UnstructuredInterpolator interp(coords, data); }, std::invalid_argument);
}

TEST_F(UnstructuredInterpolatorTest, ConstructorEmptyData) {
    std::vector<Point3D>           coords;
    std::vector<MagneticFieldData> data;

    EXPECT_THROW({ UnstructuredInterpolator interp(coords, data); }, std::invalid_argument);
}

TEST_F(UnstructuredInterpolatorTest, ConstructorInvalidPower) {
    EXPECT_THROW({ UnstructuredInterpolator interp(coordinates_, field_data_, -1.0f); }, std::invalid_argument);
}

TEST_F(UnstructuredInterpolatorTest, QueryExactMatch) {
    UnstructuredInterpolator interp(coordinates_, field_data_);

    // Query at exact data point
    InterpolationResult result = interp.query(coordinates_[0]);

    EXPECT_TRUE(result.valid);
    EXPECT_NEAR(result.data.Bx, 1.0f, 1e-6f);
    EXPECT_NEAR(result.data.By, 0.0f, 1e-6f);
    EXPECT_NEAR(result.data.Bz, 0.0f, 1e-6f);
}

TEST_F(UnstructuredInterpolatorTest, QueryInterpolation) {
    UnstructuredInterpolator interp(coordinates_, field_data_);

    // Query at center of square (0.5, 0.5, 0.0)
    Point3D             query_point(0.5f, 0.5f, 0.0f);
    InterpolationResult result = interp.query(query_point);

    EXPECT_TRUE(result.valid);
    // IDW should give some weighted average
    EXPECT_GE(result.data.Bx, 0.0f);
    EXPECT_GE(result.data.By, 0.0f);
    EXPECT_GE(result.data.Bz, 0.0f);
    EXPECT_LE(result.data.Bx, 1.0f);
    EXPECT_LE(result.data.By, 1.0f);
    EXPECT_LE(result.data.Bz, 1.0f);
}

TEST_F(UnstructuredInterpolatorTest, QueryBatch) {
    UnstructuredInterpolator interp(coordinates_, field_data_);

    std::vector<Point3D> query_points = {{0.0f, 0.0f, 0.0f}, {0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 0.0f}};

    auto results = interp.queryBatch(query_points);

    EXPECT_EQ(results.size(), 3);
    EXPECT_TRUE(results[0].valid);
    EXPECT_TRUE(results[1].valid);
    EXPECT_TRUE(results[2].valid);

    // Check exact matches
    EXPECT_NEAR(results[0].data.Bx, 1.0f, 1e-6f);
    EXPECT_NEAR(results[2].data.Bx, 1.0f, 1e-6f);
    EXPECT_NEAR(results[2].data.By, 1.0f, 1e-6f);
    EXPECT_NEAR(results[2].data.Bz, 1.0f, 1e-6f);
}

TEST_F(UnstructuredInterpolatorTest, GetDataCount) {
    UnstructuredInterpolator interp(coordinates_, field_data_);
    EXPECT_EQ(interp.getDataCount(), 5);
}

TEST_F(UnstructuredInterpolatorTest, GetPower) {
    Real                     power = 3.0f;
    UnstructuredInterpolator interp(coordinates_, field_data_, power);
    EXPECT_EQ(interp.getPower(), power);
}

TEST_F(UnstructuredInterpolatorTest, GetMaxNeighbors) {
    size_t                   max_neighbors = 3;
    UnstructuredInterpolator interp(coordinates_, field_data_, 2.0f, max_neighbors);
    EXPECT_EQ(interp.getMaxNeighbors(), max_neighbors);
}

TEST_F(UnstructuredInterpolatorTest, MoveConstructor) {
    UnstructuredInterpolator interp1(coordinates_, field_data_);
    UnstructuredInterpolator interp2(std::move(interp1));

    // interp1 should be in moved-from state, but we can still query interp2
    InterpolationResult result = interp2.query(coordinates_[0]);
    EXPECT_TRUE(result.valid);
}

TEST_F(UnstructuredInterpolatorTest, MoveAssignment) {
    UnstructuredInterpolator interp1(coordinates_, field_data_);
    UnstructuredInterpolator interp2({{0, 0, 0}}, {{0, 0, 0}});

    interp2 = std::move(interp1);

    InterpolationResult result = interp2.query(coordinates_[0]);
    EXPECT_TRUE(result.valid);
}

}  // namespace p3d
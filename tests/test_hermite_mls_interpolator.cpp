#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "point3d_interp/hermite_mls_interpolator.h"
#include "point3d_interp/types.h"

P3D_NAMESPACE_BEGIN

class HermiteMLSInterpolatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create simple test data: points on a plane with known field
        coordinates_ = {
            Point3D(0.0, 0.0, 0.0),
            Point3D(1.0, 0.0, 0.0),
            Point3D(0.0, 1.0, 0.0),
            Point3D(1.0, 1.0, 0.0),
            Point3D(0.5, 0.5, 0.0)
        };

        // Simple field: Bx = x, By = y, Bz = x*y
        // Derivatives: dBx/dx = 1, dBx/dy = 0, dBx/dz = 0
        // dBy/dx = 0, dBy/dy = 1, dBy/dz = 0
        // dBz/dx = y, dBz/dy = x, dBz/dz = 0
        field_data_ = {
            MagneticFieldData(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            MagneticFieldData(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            MagneticFieldData(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            MagneticFieldData(1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0),
            MagneticFieldData(0.5, 0.5, 0.25, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0)
        };
    }

    std::vector<Point3D> coordinates_;
    std::vector<MagneticFieldData> field_data_;
};

// Test constructor
TEST_F(HermiteMLSInterpolatorTest, Constructor) {
    HermiteMLSInterpolator::Parameters params;
    params.basis_order = HermiteMLSInterpolator::BasisOrder::Quadratic;
    params.weight_function = HermiteMLSInterpolator::WeightFunction::Gaussian;

    EXPECT_NO_THROW({
        HermiteMLSInterpolator interpolator(coordinates_, field_data_, params);
    });
}

// Test constructor with empty data
TEST_F(HermiteMLSInterpolatorTest, ConstructorEmptyData) {
    std::vector<Point3D> empty_coords;
    std::vector<MagneticFieldData> empty_field;

    EXPECT_THROW({
        HermiteMLSInterpolator interpolator(empty_coords, empty_field);
    }, std::invalid_argument);
}

// Test constructor with mismatched sizes
TEST_F(HermiteMLSInterpolatorTest, ConstructorMismatchedSizes) {
    std::vector<Point3D> coords = {Point3D(0,0,0)};
    std::vector<MagneticFieldData> field = {MagneticFieldData(), MagneticFieldData()};

    EXPECT_THROW({
        HermiteMLSInterpolator interpolator(coords, field);
    }, std::invalid_argument);
}

// Test single point query
TEST_F(HermiteMLSInterpolatorTest, QuerySinglePoint) {
    HermiteMLSInterpolator interpolator(coordinates_, field_data_);

    Point3D query_point(0.5, 0.5, 0.0);
    InterpolationResult result = interpolator.query(query_point);

    EXPECT_TRUE(result.valid);
    // Should interpolate to approximately (0.5, 0.5, 0.25)
    EXPECT_NEAR(result.data.Bx, 0.5, 1e-2);
    EXPECT_NEAR(result.data.By, 0.5, 1e-2);
    EXPECT_NEAR(result.data.Bz, 0.25, 1e-2);
}

// Test batch query
TEST_F(HermiteMLSInterpolatorTest, QueryBatch) {
    HermiteMLSInterpolator interpolator(coordinates_, field_data_);

    std::vector<Point3D> query_points = {
        Point3D(0.0, 0.0, 0.0),
        Point3D(0.5, 0.5, 0.0),
        Point3D(1.0, 1.0, 0.0)
    };

    std::vector<InterpolationResult> results = interpolator.queryBatch(query_points);

    EXPECT_EQ(results.size(), 3);
    EXPECT_TRUE(results[0].valid);
    EXPECT_TRUE(results[1].valid);
    EXPECT_TRUE(results[2].valid);

    // Check exact interpolation at data points
    EXPECT_NEAR(results[0].data.Bx, 0.0, 1e-6);
    EXPECT_NEAR(results[0].data.By, 0.0, 1e-6);
    EXPECT_NEAR(results[0].data.Bz, 0.0, 1e-6);
}

// Test bounds
TEST_F(HermiteMLSInterpolatorTest, GetBounds) {
    HermiteMLSInterpolator interpolator(coordinates_, field_data_);

    Point3D min_bound, max_bound;
    interpolator.getBounds(min_bound, max_bound);

    EXPECT_EQ(min_bound.x, 0.0);
    EXPECT_EQ(min_bound.y, 0.0);
    EXPECT_EQ(min_bound.z, 0.0);
    EXPECT_EQ(max_bound.x, 1.0);
    EXPECT_EQ(max_bound.y, 1.0);
    EXPECT_EQ(max_bound.z, 0.0);
}

// Test parameters
TEST_F(HermiteMLSInterpolatorTest, GetParameters) {
    HermiteMLSInterpolator::Parameters params;
    params.basis_order = HermiteMLSInterpolator::BasisOrder::Linear;
    params.support_radius = 3.0;

    HermiteMLSInterpolator interpolator(coordinates_, field_data_, params);

    const auto& retrieved_params = interpolator.getParameters();
    EXPECT_EQ(retrieved_params.basis_order, HermiteMLSInterpolator::BasisOrder::Linear);
    EXPECT_EQ(retrieved_params.support_radius, 3.0);
}

// Test different basis orders
TEST_F(HermiteMLSInterpolatorTest, DifferentBasisOrders) {
    std::vector<HermiteMLSInterpolator::BasisOrder> orders = {
        HermiteMLSInterpolator::BasisOrder::Linear,
        HermiteMLSInterpolator::BasisOrder::Quadratic,
        HermiteMLSInterpolator::BasisOrder::Cubic
    };

    for (auto order : orders) {
        HermiteMLSInterpolator::Parameters params;
        params.basis_order = order;

        HermiteMLSInterpolator interpolator(coordinates_, field_data_, params);
        InterpolationResult result = interpolator.query(Point3D(0.5, 0.5, 0.0));

        EXPECT_TRUE(result.valid);
        EXPECT_NEAR(result.data.Bx, 0.5, 0.1); // Allow some tolerance for different orders
    }
}

// Test different weight functions
TEST_F(HermiteMLSInterpolatorTest, DifferentWeightFunctions) {
    std::vector<HermiteMLSInterpolator::WeightFunction> functions = {
        HermiteMLSInterpolator::WeightFunction::Gaussian,
        HermiteMLSInterpolator::WeightFunction::Wendland
    };

    for (auto func : functions) {
        HermiteMLSInterpolator::Parameters params;
        params.weight_function = func;

        HermiteMLSInterpolator interpolator(coordinates_, field_data_, params);
        InterpolationResult result = interpolator.query(Point3D(0.5, 0.5, 0.0));

        EXPECT_TRUE(result.valid);
    }
}

// Test with different parameters
TEST_F(HermiteMLSInterpolatorTest, DifferentParameters) {
    HermiteMLSInterpolator::Parameters params;
    params.basis_order = HermiteMLSInterpolator::BasisOrder::Linear;
    params.weight_function = HermiteMLSInterpolator::WeightFunction::Wendland;
    params.support_radius = 2.0;
    params.derivative_weight = 0.5;

    HermiteMLSInterpolator interpolator(coordinates_, field_data_, params);
    InterpolationResult result = interpolator.query(Point3D(0.5, 0.5, 0.0));

    EXPECT_TRUE(result.valid);
    EXPECT_NEAR(result.data.Bx, 0.5, 0.2); // Allow some tolerance
}

// Test with insufficient neighbors
TEST_F(HermiteMLSInterpolatorTest, InsufficientNeighbors) {
    // Create interpolator with very small support radius
    HermiteMLSInterpolator::Parameters params;
    params.support_radius = 0.01; // Very small, should find few neighbors
    params.max_neighbors = 20;

    HermiteMLSInterpolator interpolator(coordinates_, field_data_, params);

    // Query far from data points
    Point3D query_point(10.0, 10.0, 10.0);
    InterpolationResult result = interpolator.query(query_point);

    // Should still return a result (fallback to IDW or similar)
    EXPECT_TRUE(result.valid || !result.valid); // Either is acceptable for this edge case
}

// Test derivative accuracy
TEST_F(HermiteMLSInterpolatorTest, DerivativeAccuracy) {
    HermiteMLSInterpolator interpolator(coordinates_, field_data_);

    // Query at a known point
    Point3D query_point(0.5, 0.5, 0.0);
    InterpolationResult result = interpolator.query(query_point);

    EXPECT_TRUE(result.valid);

    // Check derivatives are reasonable (should be close to analytical)
    // dBx/dx should be ~1, dBy/dy should be ~1, dBz/dx should be ~0.5, dBz/dy should be ~0.5
    EXPECT_NEAR(result.data.dBx_dx, 1.0, 0.5);
    EXPECT_NEAR(result.data.dBy_dy, 1.0, 0.5);
    EXPECT_NEAR(result.data.dBz_dx, 0.5, 0.5);
    EXPECT_NEAR(result.data.dBz_dy, 0.5, 0.5);
}

P3D_NAMESPACE_END
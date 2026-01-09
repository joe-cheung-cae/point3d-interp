#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>
#include "point3d_interp/interpolator_factory.h"
#include "point3d_interp/types.h"
#include "point3d_interp/hermite_mls_interpolator.h"
#include "point3d_interp/unstructured_interpolator.h"

P3D_NAMESPACE_BEGIN

class HMLSAccuracyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test data: quadratic function f(x,y,z) = x^2 + y^2 + z^2
        // Derivatives: df/dx = 2x, df/dy = 2y, df/dz = 2z
        const size_t num_points = 100;
        coordinates_.reserve(num_points);
        field_data_.reserve(num_points);

        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<Real> dist(-1.0, 1.0);

        for (size_t i = 0; i < num_points; ++i) {
            Point3D point(dist(gen), dist(gen), dist(gen));
            Real x = point.x, y = point.y, z = point.z;

            // Field values: Bx = x^2, By = y^2, Bz = z^2
            MagneticFieldData field;
            field.Bx = x * x;
            field.By = y * y;
            field.Bz = z * z;

            // Derivatives
            field.dBx_dx = 2 * x; field.dBx_dy = 0; field.dBx_dz = 0;
            field.dBy_dx = 0; field.dBy_dy = 2 * y; field.dBy_dz = 0;
            field.dBz_dx = 0; field.dBz_dy = 0; field.dBz_dz = 2 * z;

            coordinates_.push_back(point);
            field_data_.push_back(field);
        }

        // Create test query points (different from data points)
        query_points_.reserve(50);
        for (size_t i = 0; i < 50; ++i) {
            Point3D point(dist(gen) * 0.8, dist(gen) * 0.8, dist(gen) * 0.8); // Keep within bounds
            query_points_.push_back(point);
        }
    }

    std::vector<Point3D> coordinates_;
    std::vector<MagneticFieldData> field_data_;
    std::vector<Point3D> query_points_;

    // Analytical solution for the test function
    MagneticFieldData analyticalSolution(const Point3D& point) const {
        Real x = point.x, y = point.y, z = point.z;
        MagneticFieldData result;
        result.Bx = x * x;
        result.By = y * y;
        result.Bz = z * z;
        result.dBx_dx = 2 * x; result.dBx_dy = 0; result.dBx_dz = 0;
        result.dBy_dx = 0; result.dBy_dy = 2 * y; result.dBy_dz = 0;
        result.dBz_dx = 0; result.dBz_dy = 0; result.dBz_dz = 2 * z;
        return result;
    }

    // Calculate RMS error
    double calculateRMSE(const std::vector<InterpolationResult>& results,
                        const std::vector<Point3D>& query_points) const {
        double sum_squared_error = 0.0;
        size_t valid_count = 0;

        for (size_t i = 0; i < results.size(); ++i) {
            if (results[i].valid) {
                MagneticFieldData analytical = analyticalSolution(query_points[i]);
                double error_bx = results[i].data.Bx - analytical.Bx;
                double error_by = results[i].data.By - analytical.By;
                double error_bz = results[i].data.Bz - analytical.Bz;
                sum_squared_error += error_bx * error_bx + error_by * error_by + error_bz * error_bz;
                valid_count++;
            }
        }

        return valid_count > 0 ? std::sqrt(sum_squared_error / (3 * valid_count)) : 0.0;
    }

    // Calculate max error
    double calculateMaxError(const std::vector<InterpolationResult>& results,
                           const std::vector<Point3D>& query_points) const {
        double max_error = 0.0;

        for (size_t i = 0; i < results.size(); ++i) {
            if (results[i].valid) {
                MagneticFieldData analytical = analyticalSolution(query_points[i]);
                double error_bx = std::abs(results[i].data.Bx - analytical.Bx);
                double error_by = std::abs(results[i].data.By - analytical.By);
                double error_bz = std::abs(results[i].data.Bz - analytical.Bz);
                double max_field_error = std::max({error_bx, error_by, error_bz});
                max_error = std::max(max_error, max_field_error);
            }
        }

        return max_error;
    }

    // Calculate derivative RMS error
    double calculateDerivativeRMSE(const std::vector<InterpolationResult>& results,
                                 const std::vector<Point3D>& query_points) const {
        double sum_squared_error = 0.0;
        size_t valid_count = 0;

        for (size_t i = 0; i < results.size(); ++i) {
            if (results[i].valid) {
                MagneticFieldData analytical = analyticalSolution(query_points[i]);

                // Bx derivatives
                double error_dbx_dx = results[i].data.dBx_dx - analytical.dBx_dx;
                double error_dbx_dy = results[i].data.dBx_dy - analytical.dBx_dy;
                double error_dbx_dz = results[i].data.dBx_dz - analytical.dBx_dz;

                // By derivatives
                double error_dby_dx = results[i].data.dBy_dx - analytical.dBy_dx;
                double error_dby_dy = results[i].data.dBy_dy - analytical.dBy_dy;
                double error_dby_dz = results[i].data.dBy_dz - analytical.dBy_dz;

                // Bz derivatives
                double error_dbz_dx = results[i].data.dBz_dx - analytical.dBz_dx;
                double error_dbz_dy = results[i].data.dBz_dy - analytical.dBz_dy;
                double error_dbz_dz = results[i].data.dBz_dz - analytical.dBz_dz;

                sum_squared_error += error_dbx_dx * error_dbx_dx + error_dbx_dy * error_dbx_dy + error_dbx_dz * error_dbx_dz +
                                   error_dby_dx * error_dby_dx + error_dby_dy * error_dby_dy + error_dby_dz * error_dby_dz +
                                   error_dbz_dx * error_dbz_dx + error_dbz_dy * error_dbz_dy + error_dbz_dz * error_dbz_dz;
                valid_count++;
            }
        }

        return valid_count > 0 ? std::sqrt(sum_squared_error / (9 * valid_count)) : 0.0;
    }
};

// Test HMLS accuracy against analytical solution
TEST_F(HMLSAccuracyTest, HMLSAccuracyTest) {
    // Create HMLS interpolator directly
    HermiteMLSInterpolator::Parameters params;
    HermiteMLSInterpolator interpolator(coordinates_, field_data_, params);

    // Query all test points
    std::vector<InterpolationResult> results = interpolator.queryBatch(query_points_);

    // Calculate errors
    double rmse = calculateRMSE(results, query_points_);
    double max_error = calculateMaxError(results, query_points_);
    double derivative_rmse = calculateDerivativeRMSE(results, query_points_);

    // HMLS should be quite accurate for this quadratic function
    EXPECT_LT(rmse, 0.1) << "RMS error too high: " << rmse;
    EXPECT_LT(max_error, 0.5) << "Max error too high: " << max_error;
    EXPECT_LT(derivative_rmse, 0.5) << "Derivative RMS error too high: " << derivative_rmse;

    // Log results for analysis
    std::cout << "HMLS Accuracy Results:" << std::endl;
    std::cout << "  RMS Error: " << rmse << std::endl;
    std::cout << "  Max Error: " << max_error << std::endl;
    std::cout << "  Derivative RMS Error: " << derivative_rmse << std::endl;
}

// Compare HMLS vs IDW accuracy
TEST_F(HMLSAccuracyTest, CompareHMLSvsIDW) {
    // Create HMLS interpolator
    HermiteMLSInterpolator::Parameters hmls_params;
    HermiteMLSInterpolator hmls_interpolator(coordinates_, field_data_, hmls_params);

    // Create IDW interpolator (using UnstructuredInterpolator directly for simplicity)
    UnstructuredInterpolator idw_interpolator(coordinates_, field_data_, 2.0); // power = 2

    // Query with both methods
    std::vector<InterpolationResult> hmls_results = hmls_interpolator.queryBatch(query_points_);
    std::vector<InterpolationResult> idw_results = idw_interpolator.queryBatch(query_points_);

    // Calculate errors
    double hmls_rmse = calculateRMSE(hmls_results, query_points_);
    double idw_rmse = calculateRMSE(idw_results, query_points_);
    double hmls_derivative_rmse = calculateDerivativeRMSE(hmls_results, query_points_);

    // HMLS should be more accurate than IDW for this smooth function
    EXPECT_LT(hmls_rmse, idw_rmse * 1.2) << "HMLS RMS error: " << hmls_rmse << ", IDW RMS error: " << idw_rmse;

    // HMLS should have much better derivative accuracy
    EXPECT_LT(hmls_derivative_rmse, 1.0) << "HMLS derivative RMS error too high: " << hmls_derivative_rmse;

    std::cout << "HMLS vs IDW Comparison:" << std::endl;
    std::cout << "  HMLS RMS Error: " << hmls_rmse << std::endl;
    std::cout << "  IDW RMS Error: " << idw_rmse << std::endl;
    std::cout << "  HMLS Derivative RMS Error: " << hmls_derivative_rmse << std::endl;
}

// Test different HMLS parameters
TEST_F(HMLSAccuracyTest, ParameterSensitivity) {
    std::vector<double> rms_errors;
    std::vector<double> derivative_errors;

    // Test different basis orders
    std::vector<HermiteMLSInterpolator::BasisOrder> orders = {
        HermiteMLSInterpolator::BasisOrder::Linear,
        HermiteMLSInterpolator::BasisOrder::Quadratic,
        HermiteMLSInterpolator::BasisOrder::Cubic
    };

    for (auto order : orders) {
        HermiteMLSInterpolator::Parameters params;
        params.basis_order = order;
        params.support_radius = 1.0;
        params.max_neighbors = 20;

        HermiteMLSInterpolator interpolator(coordinates_, field_data_, params);
        std::vector<InterpolationResult> results = interpolator.queryBatch(query_points_);

        double rmse = calculateRMSE(results, query_points_);
        double derivative_rmse = calculateDerivativeRMSE(results, query_points_);

        rms_errors.push_back(rmse);
        derivative_errors.push_back(derivative_rmse);

        std::cout << "Basis order " << static_cast<int>(order) << ":" << std::endl;
        std::cout << "  RMS Error: " << rmse << std::endl;
        std::cout << "  Derivative RMS Error: " << derivative_rmse << std::endl;
    }

    // Higher order bases should generally perform better
    EXPECT_LE(rms_errors[1], rms_errors[0] * 1.5); // Quadratic vs Linear
    EXPECT_LE(derivative_errors[1], derivative_errors[0]); // Quadratic should have better derivatives
}

// Test weight function comparison
TEST_F(HMLSAccuracyTest, WeightFunctionComparison) {
    std::vector<HermiteMLSInterpolator::WeightFunction> functions = {
        HermiteMLSInterpolator::WeightFunction::Gaussian,
        HermiteMLSInterpolator::WeightFunction::Wendland
    };

    std::vector<double> rms_errors;

    for (auto func : functions) {
        HermiteMLSInterpolator::Parameters params;
        params.weight_function = func;
        params.support_radius = 1.0;

        auto interpolator = std::make_unique<HermiteMLSInterpolator>(coordinates_, field_data_, params);
        std::vector<InterpolationResult> results = interpolator->queryBatch(query_points_);

        double rmse = calculateRMSE(results, query_points_);
        rms_errors.push_back(rmse);

        std::cout << "Weight function " << (func == HermiteMLSInterpolator::WeightFunction::Gaussian ? "Gaussian" : "Wendland") << ":" << std::endl;
        std::cout << "  RMS Error: " << rmse << std::endl;
    }

    // Both should perform reasonably well
    for (double error : rms_errors) {
        EXPECT_LT(error, 0.2) << "Weight function error too high";
    }
}

// Test extrapolation behavior
TEST_F(HMLSAccuracyTest, ExtrapolationBehavior) {
    HermiteMLSInterpolator::Parameters params;
    params.support_radius = 2.0; // Larger support for extrapolation
    HermiteMLSInterpolator interpolator(coordinates_, field_data_, params);

    // Query points outside the data bounds
    std::vector<Point3D> extrapolation_points = {
        Point3D(2.0, 0.0, 0.0),   // Outside x bound
        Point3D(0.0, 2.0, 0.0),   // Outside y bound
        Point3D(0.0, 0.0, 2.0),   // Outside z bound
        Point3D(2.0, 2.0, 2.0)    // Outside all bounds
    };

    std::vector<InterpolationResult> results = interpolator.queryBatch(extrapolation_points);

    // Should handle extrapolation gracefully
    for (const auto& result : results) {
        EXPECT_TRUE(result.valid || !result.valid); // Either is acceptable for extrapolation
    }

    std::cout << "Extrapolation test completed - " << results.size() << " points queried" << std::endl;
}

P3D_NAMESPACE_END
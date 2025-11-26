#include "point3d_interp/interpolator_api.h"
#include <iostream>
#include <iomanip>
#include <chrono>

int main() {
    using namespace p3d;

    std::cout << "=== 3D Magnetic Field Data Interpolation Library Example ===\n" << std::endl;

    // Create interpolator instance
    MagneticFieldInterpolator interp(true);  // Use GPU

    // Load data
    std::cout << "Loading magnetic field data..." << std::endl;
    ErrorCode err = interp.LoadFromCSV("../data/sample_magnetic_field.csv");
    if (err != ErrorCode::Success) {
        std::cerr << "Data loading failed: " << ErrorCodeToString(err) << std::endl;
        return 1;
    }

    std::cout << "Data loaded successfully!" << std::endl;
    std::cout << "Number of data points: " << interp.GetDataPointCount() << std::endl;

    // Display grid parameters
    const auto& params = interp.GetGridParams();
    std::cout << "Grid parameters:" << std::endl;
    std::cout << "  Dimensions: " << params.dimensions[0] << " x " << params.dimensions[1] << " x "
              << params.dimensions[2] << std::endl;
    std::cout << "  Spacing: (" << params.spacing.x << ", " << params.spacing.y << ", " << params.spacing.z << ")"
              << std::endl;
    std::cout << "  Range: [" << params.min_bound.x << ", " << params.max_bound.x << "] x [" << params.min_bound.y
              << ", " << params.max_bound.y << "] x [" << params.min_bound.z << ", " << params.max_bound.z << "]"
              << std::endl;
    std::cout << std::endl;

    // Single-point interpolation test
    std::cout << "=== Single-Point Interpolation Test ===" << std::endl;

    Point3D             query_point(1.5, 1.5, 1.5);  // Grid center point
    InterpolationResult result;

    auto start = std::chrono::high_resolution_clock::now();
    err        = interp.Query(query_point, result);
    auto end   = std::chrono::high_resolution_clock::now();

    if (err == ErrorCode::Success && result.valid) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Query point: (" << query_point.x << ", " << query_point.y << ", " << query_point.z << ")"
                  << std::endl;
        std::cout << "Interpolation result:" << std::endl;
        std::cout << "  Magnetic field vector: (" << result.data.Bx << ", " << result.data.By << ", " << result.data.Bz
                  << ")" << std::endl;
        std::cout << "Query time: " << duration.count() << " microseconds" << std::endl;
    } else {
        std::cout << "Interpolation failed: " << ErrorCodeToString(err) << std::endl;
    }

    std::cout << std::endl;

    // Batch interpolation test
    std::cout << "=== Batch Interpolation Test ===" << std::endl;

    const size_t                     num_queries = 1000;
    std::vector<Point3D>             query_points;
    std::vector<InterpolationResult> results;

    // Generate random query points
    query_points.reserve(num_queries);
    results.resize(num_queries);

    for (size_t i = 0; i < num_queries; ++i) {
        Point3D point(params.min_bound.x + (params.max_bound.x - params.min_bound.x) * (rand() / double(RAND_MAX)),
                      params.min_bound.y + (params.max_bound.y - params.min_bound.y) * (rand() / double(RAND_MAX)),
                      params.min_bound.z + (params.max_bound.z - params.min_bound.z) * (rand() / double(RAND_MAX)));
        query_points.push_back(point);
    }

    start = std::chrono::high_resolution_clock::now();
    err   = interp.QueryBatch(query_points.data(), results.data(), num_queries);
    end   = std::chrono::high_resolution_clock::now();

    if (err == ErrorCode::Success) {
        auto   duration   = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        double throughput = num_queries / (duration.count() / 1000.0);

        std::cout << "Batch query of " << num_queries << " points" << std::endl;
        std::cout << "Total time: " << duration.count() << " milliseconds" << std::endl;
        std::cout << "Throughput: " << std::fixed << std::setprecision(0) << throughput << " queries/second"
                  << std::endl;

        // Display first 5 results
        std::cout << "\nFirst 5 query results:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), num_queries); ++i) {
            const auto& point = query_points[i];
            const auto& res   = results[i];
            if (res.valid) {
                std::cout << "  Point " << i + 1 << ": (" << std::fixed << std::setprecision(2) << point.x << ", "
                          << point.y << ", " << point.z << ") -> Bx = " << std::setprecision(4) << res.data.Bx
                          << std::endl;
            }
        }
    } else {
        std::cout << "Batch interpolation failed: " << ErrorCodeToString(err) << std::endl;
    }

    std::cout << std::endl;
    std::cout << "=== Example Program End ===" << std::endl;

    return 0;
}
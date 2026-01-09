#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include "point3d_interp/hermite_mls_interpolator.h"
#include "point3d_interp/types.h"

P3D_NAMESPACE_BEGIN

static void BM_HMLS_InsideDomain_8000(benchmark::State& state) {
    // Generate test data
    const size_t num_data_points = 8000;
    const size_t num_query_points = 1000;

    std::vector<Point3D> data_points;
    std::vector<MagneticFieldData> field_data;
    std::vector<Point3D> query_points;

    // Fixed seed for reproducible results
    std::mt19937 gen(42);
    std::uniform_real_distribution<Real> dist(-1.0, 1.0);

    // Generate data points
    data_points.reserve(num_data_points);
    field_data.reserve(num_data_points);
    for (size_t i = 0; i < num_data_points; ++i) {
        Point3D point(dist(gen), dist(gen), dist(gen));
        data_points.push_back(point);

        // Simple field: Bx = x, By = y, Bz = z
        MagneticFieldData field;
        field.Bx = point.x;
        field.By = point.y;
        field.Bz = point.z;
        // Derivatives
        field.dBx_dx = 1.0; field.dBx_dy = 0.0; field.dBx_dz = 0.0;
        field.dBy_dx = 0.0; field.dBy_dy = 1.0; field.dBy_dz = 0.0;
        field.dBz_dx = 0.0; field.dBz_dy = 0.0; field.dBz_dz = 1.0;
        field_data.push_back(field);
    }

    // Generate query points within the data bounds
    query_points.reserve(num_query_points);
    for (size_t i = 0; i < num_query_points; ++i) {
        query_points.emplace_back(dist(gen) * 0.8, dist(gen) * 0.8, dist(gen) * 0.8);
    }

    // Create interpolator
    HermiteMLSInterpolator::Parameters params;
    params.basis_order = HermiteMLSInterpolator::BasisOrder::Quadratic;
    params.support_radius = 0.3;
    params.max_neighbors = 20;
    HermiteMLSInterpolator interpolator(data_points, field_data, params);

    // Benchmark
    for (auto _ : state) {
        std::vector<InterpolationResult> results = interpolator.queryBatch(query_points);
        benchmark::DoNotOptimize(results);
    }

    state.SetItemsProcessed(num_query_points * state.iterations());
    state.SetLabel("HMLS 8000 data points, 1000 queries");
}

BENCHMARK(BM_HMLS_InsideDomain_8000)->Unit(benchmark::kMillisecond);

P3D_NAMESPACE_END
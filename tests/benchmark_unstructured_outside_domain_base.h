#pragma once

#include "benchmark_unstructured_base.h"

P3D_NAMESPACE_BEGIN

/**
 * @brief Base class for unstructured data performance benchmarks with queries outside the domain
 */
class UnstructuredBenchmarkOutsideDomainBase : public UnstructuredBenchmarkBase {
  protected:
    std::vector<Point3D> GenerateQueryPoints(size_t count, const GridParams& grid_params) override {
        std::vector<Point3D> points;
        points.reserve(count);

        // Generate points outside the domain bounds
        float margin = 5.0f;  // Distance outside the domain

        std::uniform_real_distribution<float> dist_x_low(grid_params.min_bound.x - margin,
                                                         grid_params.min_bound.x - 0.1f);
        std::uniform_real_distribution<float> dist_x_high(grid_params.max_bound.x + 0.1f,
                                                          grid_params.max_bound.x + margin);
        std::uniform_real_distribution<float> dist_y_low(grid_params.min_bound.y - margin,
                                                         grid_params.min_bound.y - 0.1f);
        std::uniform_real_distribution<float> dist_y_high(grid_params.max_bound.y + 0.1f,
                                                          grid_params.max_bound.y + margin);
        std::uniform_real_distribution<float> dist_z_low(grid_params.min_bound.z - margin,
                                                         grid_params.min_bound.z - 0.1f);
        std::uniform_real_distribution<float> dist_z_high(grid_params.max_bound.z + 0.1f,
                                                          grid_params.max_bound.z + margin);

        std::uniform_int_distribution<int> side_dist(0, 1);  // 0 for below min, 1 for above max

        for (size_t i = 0; i < count; ++i) {
            float x = side_dist(rng_) ? dist_x_high(rng_) : dist_x_low(rng_);
            float y = side_dist(rng_) ? dist_y_high(rng_) : dist_y_low(rng_);
            float z = side_dist(rng_) ? dist_z_high(rng_) : dist_z_low(rng_);
            points.push_back(Point3D(x, y, z));
        }

        return points;
    }

    std::string GetBenchmarkType() const override { return "_unstructured_out_of_domain"; }
};

P3D_NAMESPACE_END

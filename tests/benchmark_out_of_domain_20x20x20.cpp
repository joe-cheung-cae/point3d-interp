#include "benchmark_base.h"

/**
 * @brief Performance benchmark for interpolation points outside the domain (20x20x20 data)
 */
class BenchmarkOutOfDomain20x20x20 : public p3d::BenchmarkBase {
  protected:
    std::array<size_t, 3> GetDataDimensions() const override {
        return {20, 20, 20};  // 8,000 points
    }

    std::vector<p3d::Point3D> GenerateQueryPoints(size_t count, const p3d::GridParams& grid_params) override {
        std::vector<p3d::Point3D> points;
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
            points.push_back(p3d::Point3D(x, y, z));
        }

        return points;
    }
};

int main() {
    BenchmarkOutOfDomain20x20x20 benchmark;
    benchmark.RunAllBenchmarks();
    return 0;
}
#include "point3d_interp/interpolator_api.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <random>

using namespace p3d;

/**
 * @brief Memory usage benchmark program
 */
class Benchmark {
  public:
    Benchmark() : rng_(std::random_device{}()) {}

    void RunMemoryBenchmark() {
        std::cout << "=== Memory usage benchmark ===\n\n";

        std::vector<std::array<size_t, 3>> sizes = {
            {10, 10, 10},    // ~0.4MB
            {50, 50, 50},    // ~50MB
            {100, 100, 100}  // ~800MB (may exceed GPU memory)
        };

        for (const auto& size : sizes) {
            size_t total_points = size[0] * size[1] * size[2];
            size_t memory_mb    = total_points * sizeof(MagneticFieldData) / (1024 * 1024);

            std::cout << "Data scale: " << size[0] << "x" << size[1] << "x" << size[2] << " (" << total_points
                      << " points, ~" << memory_mb << " MB)\n";

            auto test_data = GenerateTestData(size);

            // CPU test
            {
                MagneticFieldInterpolator interp(false);
                auto                      start = std::chrono::high_resolution_clock::now();
                try {
                    interp.LoadFromMemory(test_data.coordinates.data(), test_data.field_data.data(), total_points);
                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                    std::cout << "  CPU loading time: " << duration.count() << " ms\n";
                } catch (const std::exception& e) {
                    std::cout << "  CPU loading failed: " << e.what() << "\n";
                }
            }

            // GPU test
            {
                MagneticFieldInterpolator interp(true);
                auto                      start = std::chrono::high_resolution_clock::now();
                try {
                    interp.LoadFromMemory(test_data.coordinates.data(), test_data.field_data.data(), total_points);
                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                    std::cout << "  GPU loading time: " << duration.count() << " ms\n";
                } catch (const std::exception& e) {
                    std::cout << "  GPU loading failed (possibly insufficient memory): " << e.what() << "\n";
                }
            }

            std::cout << "\n";
        }
    }

  private:
    struct TestData {
        std::vector<Point3D>           coordinates;
        std::vector<MagneticFieldData> field_data;
        GridParams                     grid_params;
    };

    std::mt19937 rng_;

    TestData GenerateTestData(const std::array<size_t, 3>& dimensions) {
        TestData data;

        data.grid_params.origin     = Point3D(0.0f, 0.0f, 0.0f);
        data.grid_params.spacing    = Point3D(1.0f, 1.0f, 1.0f);
        data.grid_params.dimensions = {static_cast<uint32_t>(dimensions[0]), static_cast<uint32_t>(dimensions[1]),
                                       static_cast<uint32_t>(dimensions[2])};
        data.grid_params.update_bounds();

        size_t total_points = dimensions[0] * dimensions[1] * dimensions[2];
        data.coordinates.reserve(total_points);
        data.field_data.reserve(total_points);

        // Generate magnetic field data for each grid point
        for (size_t k = 0; k < dimensions[2]; ++k) {
            for (size_t j = 0; j < dimensions[1]; ++j) {
                for (size_t i = 0; i < dimensions[0]; ++i) {
                    Point3D coord(data.grid_params.origin.x + i * data.grid_params.spacing.x,
                                  data.grid_params.origin.y + j * data.grid_params.spacing.y,
                                  data.grid_params.origin.z + k * data.grid_params.spacing.z);
                    data.coordinates.push_back(coord);

                    // Generate magnetic field data for each grid point
                    MagneticFieldData field(coord.x + coord.y + coord.z + 1.0f,  // B = x + y + z + 1
                                            1.0f, 1.0f, 1.0f                     // Constant gradient
                    );
                    data.field_data.push_back(field);
                }
            }
        }

        return data;
    }
};

int main() {
    Benchmark benchmark;
    benchmark.RunMemoryBenchmark();
    return 0;
}
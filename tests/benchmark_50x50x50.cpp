#include "benchmark_base.h"

/**
 * @brief Performance benchmark program for 50x50x50 data scale
 */
class Benchmark : public p3d::BenchmarkBase {
  protected:
    std::array<size_t, 3> GetDataDimensions() const override {
        return {50, 50, 50};  // 125,000 points
    }
};

int main() {
    Benchmark benchmark;
    benchmark.RunAllBenchmarks();
    return 0;
}
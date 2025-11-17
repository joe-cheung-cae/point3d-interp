#include "benchmark_base.h"

/**
 * @brief Performance benchmark program for 10x10x10 data scale
 */
class Benchmark : public p3d::BenchmarkBase {
  protected:
    std::array<size_t, 3> GetDataDimensions() const override {
        return {10, 10, 10};  // 1,000 points
    }
};

int main() {
    Benchmark benchmark;
    benchmark.RunAllBenchmarks();
    return 0;
}
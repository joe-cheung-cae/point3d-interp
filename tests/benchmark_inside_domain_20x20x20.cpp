#include "benchmark_base.h"

/**
 * @brief Performance benchmark program for 20x20x20 data scale
 */
class Benchmark : public p3d::BenchmarkBase {
  protected:
    std::array<size_t, 3> GetDataDimensions() const override {
        return {20, 20, 20};  // 8,000 points
    }

    std::string GetBenchmarkType() const override { return "_in_domain"; }
};

int main() {
    Benchmark benchmark;
    benchmark.RunAllBenchmarks();
    return 0;
}
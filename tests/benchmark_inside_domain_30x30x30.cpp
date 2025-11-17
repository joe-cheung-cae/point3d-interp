#include "benchmark_base.h"

/**
 * @brief Performance benchmark program for 30x30x30 data scale
 */
class Benchmark : public p3d::BenchmarkBase {
  protected:
    std::array<size_t, 3> GetDataDimensions() const override {
        return {30, 30, 30};  // 27,000 points
    }

    std::string GetBenchmarkType() const override { return "_in_domain"; }
};

int main() {
    Benchmark benchmark;
    benchmark.RunAllBenchmarks();
    return 0;
}
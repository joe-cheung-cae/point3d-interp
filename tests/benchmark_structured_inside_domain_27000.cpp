#include "benchmark_base.h"

/**
 * @brief Performance benchmark program for structured data with 27,000 points (inside domain)
 */
class BenchmarkStructuredInsideDomain27000 : public p3d::BenchmarkBase {
  protected:
    std::array<size_t, 3> GetDataDimensions() const override {
        return {30, 30, 30};  // 27,000 points
    }

    std::string GetBenchmarkType() const override { return "_structured_in_domain"; }
};

int main() {
    BenchmarkStructuredInsideDomain27000 benchmark;
    benchmark.RunAllBenchmarks();
    return 0;
}
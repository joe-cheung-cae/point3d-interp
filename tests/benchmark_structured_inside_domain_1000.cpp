#include "benchmark_structured_base.h"

/**
 * @brief Performance benchmark program for structured data with 1,000 points (inside domain)
 */
class BenchmarkStructuredInsideDomain1000 : public p3d::BenchmarkBase {
  protected:
    std::array<size_t, 3> GetDataDimensions() const override {
        return {10, 10, 10};  // 1,000 points
    }

    std::string GetBenchmarkType() const override { return "_structured_in_domain"; }
};

int main() {
    BenchmarkStructuredInsideDomain1000 benchmark;
    benchmark.RunAllBenchmarks();
    return 0;
}
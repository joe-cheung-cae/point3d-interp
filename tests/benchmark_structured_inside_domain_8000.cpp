#include "benchmark_structured_base.h"

/**
 * @brief Performance benchmark program for structured data with 8,000 points (inside domain)
 */
class BenchmarkStructuredInsideDomain8000 : public p3d::BenchmarkBase {
  protected:
    std::array<size_t, 3> GetDataDimensions() const override {
        return {20, 20, 20};  // 8,000 points
    }

    std::string GetBenchmarkType() const override { return "_structured_in_domain"; }
};

int main() {
    BenchmarkStructuredInsideDomain8000 benchmark;
    benchmark.RunAllBenchmarks();
    return 0;
}
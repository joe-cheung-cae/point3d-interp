#include "benchmark_base.h"

/**
 * @brief Performance benchmark program for structured data with 125,000 points (inside domain)
 */
class BenchmarkStructuredInsideDomain125000 : public p3d::BenchmarkBase {
  protected:
    std::array<size_t, 3> GetDataDimensions() const override {
        return {50, 50, 50};  // 125,000 points
    }

    std::string GetBenchmarkType() const override { return "_structured_in_domain"; }
};

int main() {
    BenchmarkStructuredInsideDomain125000 benchmark;
    benchmark.RunAllBenchmarks();
    return 0;
}
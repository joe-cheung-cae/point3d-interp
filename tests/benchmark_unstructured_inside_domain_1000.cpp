#include "benchmark_unstructured_base.h"

/**
 * @brief Performance benchmark program for unstructured data with 1,000 points (inside domain)
 */
class BenchmarkUnstructuredInsideDomain1000 : public p3d::UnstructuredBenchmarkBase {
  protected:
    size_t GetDataPointCount() const override {
        return 1000;  // 1,000 scattered points
    }

    std::string GetBenchmarkType() const override { return "_unstructured_in_domain"; }
};

int main() {
    BenchmarkUnstructuredInsideDomain1000 benchmark;
    benchmark.RunAllBenchmarks();
    return 0;
}
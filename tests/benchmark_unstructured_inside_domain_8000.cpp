#include "benchmark_unstructured_base.h"

/**
 * @brief Performance benchmark program for unstructured data with 8,000 points (inside domain)
 */
class BenchmarkUnstructuredInsideDomain8000 : public p3d::UnstructuredBenchmarkBase {
  protected:
    size_t GetDataPointCount() const override {
        return 8000;  // 8,000 scattered points
    }

    std::string GetBenchmarkType() const override { return "_unstructured_in_domain"; }
};

int main() {
    BenchmarkUnstructuredInsideDomain8000 benchmark;
    benchmark.RunAllBenchmarks();
    return 0;
}
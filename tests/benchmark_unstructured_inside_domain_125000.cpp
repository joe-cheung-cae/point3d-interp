#include "benchmark_unstructured_base.h"

/**
 * @brief Performance benchmark program for unstructured data with 125,000 points (inside domain)
 */
class BenchmarkUnstructuredInsideDomain125000 : public p3d::UnstructuredBenchmarkBase {
  protected:
    size_t GetDataPointCount() const override {
        return 125000;  // 125,000 scattered points
    }

    std::string GetBenchmarkType() const override { return "_unstructured_in_domain"; }
};

int main() {
    BenchmarkUnstructuredInsideDomain125000 benchmark;
    benchmark.RunAllBenchmarks();
    return 0;
}
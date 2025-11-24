#include "benchmark_unstructured_base.h"

/**
 * @brief Performance benchmark program for unstructured data with 27,000 points (inside domain)
 */
class BenchmarkUnstructuredInsideDomain27000 : public p3d::UnstructuredBenchmarkBase {
  protected:
    size_t GetDataPointCount() const override {
        return 27000;  // 27,000 scattered points
    }

    std::string GetBenchmarkType() const override { return "_unstructured_in_domain"; }
};

int main() {
    BenchmarkUnstructuredInsideDomain27000 benchmark;
    benchmark.RunAllBenchmarks();
    return 0;
}
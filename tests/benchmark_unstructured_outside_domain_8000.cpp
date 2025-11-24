#include "benchmark_unstructured_outside_domain_base.h"

/**
 * @brief Performance benchmark program for unstructured data with 8,000 points (outside domain)
 */
class BenchmarkUnstructuredOutsideDomain8000 : public p3d::UnstructuredBenchmarkOutsideDomainBase {
  protected:
    size_t GetDataPointCount() const override {
        return 8000;  // 8,000 scattered points
    }
};

int main() {
    BenchmarkUnstructuredOutsideDomain8000 benchmark;
    benchmark.RunAllBenchmarks();
    return 0;
}
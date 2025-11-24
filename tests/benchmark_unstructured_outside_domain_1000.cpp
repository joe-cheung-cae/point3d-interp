#include "benchmark_unstructured_outside_domain_base.h"

/**
 * @brief Performance benchmark program for unstructured data with 1,000 points (outside domain)
 */
class BenchmarkUnstructuredOutsideDomain1000 : public p3d::UnstructuredBenchmarkOutsideDomainBase {
  protected:
    size_t GetDataPointCount() const override {
        return 1000;  // 1,000 scattered points
    }
};

int main() {
    BenchmarkUnstructuredOutsideDomain1000 benchmark;
    benchmark.RunAllBenchmarks();
    return 0;
}
#include "benchmark_unstructured_outside_domain_base.h"

/**
 * @brief Performance benchmark program for unstructured data with 125,000 points (outside domain)
 */
class BenchmarkUnstructuredOutsideDomain125000 : public p3d::UnstructuredBenchmarkOutsideDomainBase {
  protected:
    size_t GetDataPointCount() const override {
        return 125000;  // 125,000 scattered points
    }
};

int main() {
    BenchmarkUnstructuredOutsideDomain125000 benchmark;
    benchmark.RunAllBenchmarks();
    return 0;
}
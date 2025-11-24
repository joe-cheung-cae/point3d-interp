#include "benchmark_unstructured_outside_domain_base.h"

/**
 * @brief Performance benchmark program for unstructured data with 27,000 points (outside domain)
 */
class BenchmarkUnstructuredOutsideDomain27000 : public p3d::UnstructuredBenchmarkOutsideDomainBase {
  protected:
    size_t GetDataPointCount() const override {
        return 27000;  // 27,000 scattered points
    }
};

int main() {
    BenchmarkUnstructuredOutsideDomain27000 benchmark;
    benchmark.RunAllBenchmarks();
    return 0;
}
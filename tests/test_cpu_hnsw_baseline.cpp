#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "cuvs_silicon/baselines/cpu_hnsw.hpp"

namespace {

struct RecordingHnswIndex {
  std::size_t last_worker_count = 0;
  std::size_t last_ef = 0;
  const float* last_query = nullptr;
  std::size_t last_k = 0;

  void setNumThreads(std::size_t worker_count) {
    last_worker_count = worker_count;
  }

  void setEf(std::size_t ef) {
    last_ef = ef;
  }

  std::uint32_t searchKnn(const float* query, std::size_t k) {
    last_query = query;
    last_k = k;
    return 7;
  }
};

}  // namespace

int main() {
  using cuvs_silicon::baselines::CpuHnswBuildParams;
  using cuvs_silicon::baselines::CpuHnswBaselineBuilder;
  using cuvs_silicon::baselines::CpuHnswSearchParams;

  const CpuHnswBaselineBuilder builder;
  const auto& build_params = builder.params();
  const auto& search_params = builder.search_params();

  assert(build_params.m == 16);
  assert(build_params.ef_construction == 200);
  assert(build_params.worker_count == 1);
  assert(search_params.ef_search == 128);
  assert(search_params.worker_count == 1);

  bool rejected_parallel_indexing = false;
  try {
    CpuHnswBaselineBuilder(CpuHnswBuildParams{16, 200, 2});
  } catch (const std::invalid_argument&) {
    rejected_parallel_indexing = true;
  }
  assert(rejected_parallel_indexing);

  bool rejected_parallel_search = false;
  try {
    CpuHnswBaselineBuilder(CpuHnswBuildParams{}, CpuHnswSearchParams{128, 2});
  } catch (const std::invalid_argument&) {
    rejected_parallel_search = true;
  }
  assert(rejected_parallel_search);

  const std::vector<float> dataset{
      0.0F, 1.0F, 2.0F,
      3.0F, 4.0F, 5.0F,
  };

  const auto index = builder.build(dataset, 3);
  const auto& index_params = index.parameters();

  assert(index_params.m == 16);
  assert(index_params.ef_construction == 200);
  assert(index_params.worker_count == 1);
  assert(index_params.vector_count == 2);
  assert(index_params.dimension == 3);

  RecordingHnswIndex hnsw_index;
  const float query[] = {1.0F, 2.0F, 3.0F};
  const auto result = builder.search(hnsw_index, query, 10);

  assert(result == 7);
  assert(hnsw_index.last_worker_count == 1);
  assert(hnsw_index.last_ef == 128);
  assert(hnsw_index.last_query == query);
  assert(hnsw_index.last_k == 10);

  return 0;
}

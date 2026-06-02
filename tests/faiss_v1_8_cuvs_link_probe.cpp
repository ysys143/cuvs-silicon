#include <cstdint>

#include <faiss/Index.h>
#include <faiss/MetricType.h>

#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/resources.hpp>

namespace {

void reference_cuvs_cagra_symbols_from_faiss_consumer() {
  raft::resources resources;
  cuvs::neighbors::cagra::index_params index_params;
  cuvs::neighbors::cagra::search_params search_params;

  index_params.metric = cuvs::distance::DistanceType::L2Expanded;
  search_params.itopk_size = 2;
  search_params.algo = cuvs::neighbors::cagra::search_algo::AUTO;

  float dataset_storage[8] = {};
  const auto dataset =
      raft::make_device_matrix_view<const float, std::int64_t>(
          dataset_storage, 2, 4);

  auto index =
      cuvs::neighbors::cagra::build(resources, index_params, dataset);

  float query_storage[4] = {};
  std::uint32_t neighbor_storage[2] = {};
  float distance_storage[2] = {};

  const auto queries =
      raft::make_device_matrix_view<const float, std::int64_t>(
          query_storage, 1, 4);
  auto neighbors =
      raft::make_device_matrix_view<std::uint32_t, std::int64_t>(
          neighbor_storage, 1, 2);
  auto distances =
      raft::make_device_matrix_view<float, std::int64_t>(
          distance_storage, 1, 2);

  cuvs::neighbors::cagra::search(
      resources, search_params, index, queries, neighbors, distances);

  faiss::idx_t faiss_index_id = 0;
  (void)faiss_index_id;
}

}  // namespace

int main() {
  const auto probe = &reference_cuvs_cagra_symbols_from_faiss_consumer;
  return probe == nullptr ? 1 : 0;
}

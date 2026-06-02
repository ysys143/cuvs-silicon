#include <cstdint>
#include <limits>
#include <type_traits>

#include <faiss/Index.h>
#include <faiss/MetricType.h>

#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

static_assert(std::is_default_constructible_v<raft::resources>);
static_assert(std::is_default_constructible_v<cuvs::neighbors::cagra::index_params>);
static_assert(std::is_default_constructible_v<cuvs::neighbors::cagra::search_params>);
static_assert(std::is_default_constructible_v<
              cuvs::neighbors::cagra::index<float, std::uint32_t>>);
static_assert(std::is_same_v<
              cuvs::neighbors::cagra::index<float, std::uint32_t>::index_type,
              std::uint32_t>);
static_assert(std::is_signed_v<faiss::idx_t>);
static_assert(sizeof(faiss::idx_t) >= sizeof(std::uint32_t));
static_assert(std::is_same_v<float, float>);

namespace {

using FaissLabel = faiss::idx_t;
using CagraSearchLabel = std::uint32_t;
using FaissDistance = float;
using CagraSearchDistance = float;

static_assert(std::is_same_v<CagraSearchDistance, FaissDistance>);
static_assert(std::numeric_limits<CagraSearchLabel>::max() <=
              static_cast<unsigned long long>(
                  std::numeric_limits<FaissLabel>::max()));

void compile_cuvs_cagra_api_consumed_next_to_faiss_v1_8_headers() {
  raft::resources resources;
  cuvs::neighbors::cagra::index_params index_params;
  cuvs::neighbors::cagra::search_params search_params;

  index_params.intermediate_graph_degree = 128;
  index_params.graph_degree = 64;
  index_params.attach_dataset_on_build = true;
  index_params.guarantee_connectivity = false;
  index_params.metric = cuvs::distance::DistanceType::L2Expanded;

  search_params.itopk_size = 64;
  search_params.algo = cuvs::neighbors::cagra::search_algo::AUTO;
  search_params.hashmap_mode = cuvs::neighbors::cagra::hash_mode::AUTO;

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

  static_assert(std::is_same_v<decltype(neighbors)::element_type,
                               CagraSearchLabel>);
  static_assert(std::is_same_v<decltype(distances)::element_type,
                               CagraSearchDistance>);

  cuvs::neighbors::cagra::search(
      resources, search_params, index, queries, neighbors, distances);

  faiss::idx_t faiss_index_id = 0;
  FaissLabel* faiss_label_output = &faiss_index_id;
  FaissDistance* faiss_distance_output = distance_storage;
  *faiss_label_output = static_cast<FaissLabel>(neighbor_storage[0]);
  *faiss_distance_output = distances.data_handle()[0];
  (void)faiss_index_id;
}

}  // namespace

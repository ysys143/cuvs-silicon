#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <type_traits>
#include <vector>

#include <cuvs/distance/distance.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::neighbors::cagra {

enum class search_algo {
  SINGLE_CTA = 0,
  MULTI_CTA = 1,
  MULTI_KERNEL = 2,
  AUTO = 100,
};

enum class hash_mode {
  HASH = 0,
  SMALL = 1,
  AUTO = 100,
};

struct index_params {
  std::uint32_t intermediate_graph_degree = 128;
  std::uint32_t graph_degree = 64;
  bool attach_dataset_on_build = true;
  bool guarantee_connectivity = false;
  cuvs::distance::DistanceType metric =
      cuvs::distance::DistanceType::L2Expanded;
};

struct search_params {
  std::uint64_t max_queries = 0;
  std::uint64_t itopk_size = 64;
  std::uint64_t max_iterations = 0;
  search_algo algo = search_algo::AUTO;
  std::uint64_t team_size = 0;
  std::uint64_t search_width = 1;
  std::uint64_t min_iterations = 0;
  std::uint64_t thread_block_size = 0;
  hash_mode hashmap_mode = hash_mode::AUTO;
  std::uint64_t hashmap_min_bitlen = 0;
  float hashmap_max_fill_rate = 0.5F;
  std::uint32_t num_random_samplings = 1;
  std::uint64_t rand_xor_mask = 0x128394;
};

template <typename DataT, typename IndexT = std::uint32_t>
class index {
  static_assert(std::is_same_v<IndexT, std::uint32_t>,
                "cuvs-silicon CAGRA MVP exposes uint32_t graph indices");

 public:
  using value_type = DataT;
  using index_type = IndexT;

  index() = default;
  index(std::vector<DataT> dataset, std::int64_t rows, std::int64_t cols)
      : dataset_(std::move(dataset)), rows_(rows), cols_(cols) {}

  const std::vector<DataT>&    dataset()    const noexcept { return dataset_; }
  const std::vector<IndexT>&   knn_graph()  const noexcept { return knn_graph_; }
  std::int64_t rows()          const noexcept { return rows_; }
  std::int64_t cols()          const noexcept { return cols_; }
  std::uint32_t graph_degree() const noexcept { return graph_degree_; }

  void set_knn_graph(std::vector<IndexT> g, std::uint32_t degree) {
    knn_graph_    = std::move(g);
    graph_degree_ = degree;
  }

  // Navigation nodes: a subset of dataset indices used as entry points.
  // nav_vectors_ caches the actual vectors (N_NAV × D float32) so that
  // cblas_sgemm can compute all Q×N_NAV distances in one AMX-accelerated call.
  const std::vector<IndexT>&  nav_nodes()    const noexcept { return nav_nodes_; }
  const std::vector<DataT>&   nav_vectors()  const noexcept { return nav_vectors_; }
  void set_nav_nodes(std::vector<IndexT> n) { nav_nodes_ = std::move(n); }
  void set_nav_vectors(std::vector<DataT> v) { nav_vectors_ = std::move(v); }

 private:
  std::vector<DataT>    dataset_;
  std::vector<IndexT>   knn_graph_;   // N × graph_degree, row-major
  std::vector<IndexT>   nav_nodes_;   // sqrt(N) entry point indices
  std::vector<DataT>    nav_vectors_; // N_NAV × D, row-major (cached for sgemm)
  std::uint32_t         graph_degree_ = 0;
  std::int64_t          rows_ = 0;
  std::int64_t          cols_ = 0;
};

template <typename DataT, typename ExtentT>
index<DataT, std::uint32_t> build(
    const raft::resources& resources,
    const index_params& params,
    raft::device_matrix_view<const DataT, ExtentT> dataset);

template <typename DataT, typename IndexT, typename ExtentT>
void search(const raft::resources& resources,
            const search_params& params,
            const index<DataT, IndexT>& index,
            raft::device_matrix_view<const DataT, ExtentT> queries,
            raft::device_matrix_view<IndexT, ExtentT> neighbors,
            raft::device_matrix_view<float, ExtentT> distances);

// Index serialization — save/load a built CAGRA index to/from a binary file.
// Format: magic("CGRA") + version(1) + N + D + G + flags + knn_graph
//         [+ dataset if stored] [+ nav_nodes + nav_vectors if present]
void save_index(const index<float, std::uint32_t>& idx,
                const std::string& path);
index<float, std::uint32_t> load_index(const std::string& path);

// Metal dispatch instrumentation — test hook.
// metal_dispatch_count() returns how many times a Metal GPU kernel was
// actually dispatched via MTLComputeCommandEncoder since the last reset.
// A value of 0 after search() means CPU fallback was used (TDD Red condition).
std::uint64_t metal_dispatch_count() noexcept;
void reset_metal_dispatch_count() noexcept;
// Called from cagra.cpp Metal branch to record each GPU dispatch.
void increment_metal_dispatch_count() noexcept;

}  // namespace cuvs::neighbors::cagra

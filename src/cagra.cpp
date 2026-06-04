#include <cstdint>
#include <algorithm>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuvs/neighbors/cagra.hpp>

#if defined(__APPLE__) && defined(__aarch64__)
#include <cuvs_silicon/metal_context.hpp>
#include <Accelerate/Accelerate.h>
#endif

namespace cuvs::neighbors::cagra {

template <>
index<float, std::uint32_t> build<float, std::int64_t>(
    const raft::resources& resources,
    const index_params& params,
    raft::device_matrix_view<const float, std::int64_t> dataset) {
  (void)resources;
  if (params.metric != cuvs::distance::DistanceType::L2Expanded) {
    throw std::invalid_argument("cuvs-silicon CAGRA MVP supports L2Expanded");
  }
  if (dataset.data_handle() == nullptr || dataset.extent(0) <= 0 ||
      dataset.extent(1) <= 0) {
    throw std::invalid_argument("cuvs-silicon CAGRA build requires a non-empty dataset");
  }

  const std::int64_t N = dataset.extent(0);
  const std::int64_t D = dataset.extent(1);
  const float* data    = dataset.data_handle();

  std::vector<float> storage;
  if (params.attach_dataset_on_build)
    storage.assign(data, data + N * D);

  index<float, std::uint32_t> idx{std::move(storage), N, D};

#if defined(__APPLE__) && defined(__aarch64__)
  // GPU-accelerated KNN graph build via random-bucket seeding + nn-descent.
  // No N^2 cap — scales to millions of vectors.
  const std::uint32_t G = std::min<std::uint32_t>(
      params.graph_degree, static_cast<std::uint32_t>(N - 1));

  if (N > 1 && G > 0) {
    auto& ctx = cuvs_silicon::MetalContext::instance();
    if (ctx.is_available()) {
      const int64_t bucket_size = std::max<int64_t>(
          static_cast<int64_t>(G) * 8,
          std::min<int64_t>(4096LL,
              static_cast<int64_t>(std::sqrt(static_cast<double>(N)) * 2)));
      // IVF K-means seeding provides high-quality initial graph for large N,
      // so fewer nn-descent iterations are needed to reach good recall.
      const uint32_t nd_iters = (N <= 50000) ? 20 : (N <= 200000) ? 5 : 0;
      auto knn = ctx.build_knn_graph(data, N, D, G, nd_iters, bucket_size);
      idx.set_knn_graph(std::move(knn), G);

      // Navigation nodes: min(200, sqrt(N)) evenly-spaced indices.
      // Used at search time as brute-force entry point candidates.
      const int64_t n_nav = std::min<int64_t>(
          200, static_cast<int64_t>(std::sqrt(static_cast<double>(N))));
      std::vector<std::uint32_t> nav(static_cast<std::size_t>(n_nav));
      const double step = static_cast<double>(N) / n_nav;
      for (int64_t i = 0; i < n_nav; ++i)
        nav[static_cast<std::size_t>(i)] =
            static_cast<std::uint32_t>(i * step);
      // Build nav_vectors matrix (N_NAV × D) — cached for AMX-accelerated
      // cblas_sgemm lookup at search time. Avoids repeated gather per query.
      std::vector<float> nav_vecs(static_cast<std::size_t>(n_nav * D));
      for (int64_t i = 0; i < n_nav; ++i) {
        const std::uint32_t ni = nav[static_cast<std::size_t>(i)];
        std::copy_n(data + ni * D,
                    static_cast<std::size_t>(D),
                    nav_vecs.data() + i * D);
      }
      idx.set_nav_nodes(std::move(nav));
      idx.set_nav_vectors(std::move(nav_vecs));
    }
  }
#endif

  return idx;
}

template <>
void search<float, std::uint32_t, std::int64_t>(
    const raft::resources& resources,
    const search_params& params,
    const index<float, std::uint32_t>& index,
    raft::device_matrix_view<const float, std::int64_t> queries,
    raft::device_matrix_view<std::uint32_t, std::int64_t> neighbors,
    raft::device_matrix_view<float, std::int64_t> distances) {
  (void)params;
  (void)resources;
  if (queries.data_handle() == nullptr || neighbors.data_handle() == nullptr ||
      distances.data_handle() == nullptr) {
    throw std::invalid_argument("cuvs-silicon CAGRA search requires valid buffers");
  }
  if (queries.extent(1) != index.cols()) {
    throw std::invalid_argument("cuvs-silicon CAGRA query dimension mismatch");
  }
  if (neighbors.extent(0) != queries.extent(0) ||
      distances.extent(0) != queries.extent(0) ||
      neighbors.extent(1) != distances.extent(1)) {
    throw std::invalid_argument("cuvs-silicon CAGRA output shape mismatch");
  }

  const auto query_count = queries.extent(0);
  const auto dimension = queries.extent(1);
  const auto vector_count = index.rows();
  const auto k = neighbors.extent(1);
  const auto& dataset = index.dataset();

  // Metal GPU path — CAGRA beam search if graph built, brute-force otherwise.
#if defined(__APPLE__) && defined(__aarch64__)
  {
    auto& ctx = cuvs_silicon::MetalContext::instance();
    if (ctx.is_available()) {
      uint64_t dispatches = 0;
      const bool has_graph = (index.graph_degree() > 0 &&
                              !index.knn_graph().empty());
      if (has_graph) {
        // Compute per-query entry points via AMX-accelerated cblas_sgemm.
        // cross[Q×N_NAV] = queries[Q×D] @ nav_vectors[N_NAV×D]^T
        // dist[q,i] = ||q||^2 - 2*cross[q,i] + ||nav_i||^2 → argmin per row.
        const auto& nav     = index.nav_nodes();
        const auto& nav_vec = index.nav_vectors();
        std::vector<std::uint32_t> entry_nodes(
            static_cast<std::size_t>(query_count), 0u);
        if (!nav.empty() && !nav_vec.empty()) {
          const std::int64_t n_nav = static_cast<std::int64_t>(nav.size());
          const float* qptr        = queries.data_handle();

          // Query norms
          std::vector<float> q_norms(static_cast<std::size_t>(query_count));
          for (std::int64_t q = 0; q < query_count; ++q) {
            float s = 0.f;
            const float* qv = qptr + q * dimension;
            for (std::int64_t d = 0; d < dimension; ++d) s += qv[d] * qv[d];
            q_norms[static_cast<std::size_t>(q)] = s;
          }
          // Nav norms
          std::vector<float> n_norms(static_cast<std::size_t>(n_nav));
          for (std::int64_t i = 0; i < n_nav; ++i) {
            float s = 0.f;
            const float* nv = nav_vec.data() + i * dimension;
            for (std::int64_t d = 0; d < dimension; ++d) s += nv[d] * nv[d];
            n_norms[static_cast<std::size_t>(i)] = s;
          }
          // Cross products: cross[Q×N_NAV] = queries @ nav_vectors^T  (AMX)
          std::vector<float> cross(
              static_cast<std::size_t>(query_count * n_nav));
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                      (int)query_count, (int)n_nav, (int)dimension,
                      1.0f, qptr, (int)dimension,
                      nav_vec.data(), (int)dimension,
                      0.0f, cross.data(), (int)n_nav);
          // argmin per query
          for (std::int64_t q = 0; q < query_count; ++q) {
            float best_d = std::numeric_limits<float>::infinity();
            std::uint32_t best_n = nav[0];
            const float qn = q_norms[static_cast<std::size_t>(q)];
            for (std::int64_t i = 0; i < n_nav; ++i) {
              float d = qn - 2.f * cross[static_cast<std::size_t>(q * n_nav + i)]
                        + n_norms[static_cast<std::size_t>(i)];
              if (d < best_d) {
                best_d = d;
                best_n = nav[static_cast<std::size_t>(i)];
              }
            }
            entry_nodes[static_cast<std::size_t>(q)] = best_n;
          }
        }

        // Adaptive search params: small Q → smaller beam to reduce serial work
        // and improve single-query latency (beam_size scales O(beam²) per iter).
        uint32_t beam = static_cast<uint32_t>(
            params.itopk_size > 0 ? params.itopk_size : 128);
        uint32_t iters = static_cast<uint32_t>(
            params.max_iterations > 0 ? params.max_iterations : 200);
        if (query_count <= 4) {
            beam  = std::min(beam,  32u);
            iters = std::min(iters, 200u);
        }
        dispatches = ctx.search_cagra(
            dataset.data(),           vector_count, dimension,
            index.knn_graph().data(), index.graph_degree(),
            queries.data_handle(),    query_count,
            neighbors.data_handle(),  distances.data_handle(),
            k, beam, iters,
            nav.empty() ? nullptr : entry_nodes.data());
      } else {
        dispatches = ctx.search_brute_force(
            dataset.data(),          vector_count, dimension,
            queries.data_handle(),   query_count,
            neighbors.data_handle(), distances.data_handle(),
            k);
      }
      for (std::uint64_t i = 0; i < dispatches; ++i)
        increment_metal_dispatch_count();
      return;
    }
  }
#endif

  std::vector<std::pair<float, std::uint32_t>> candidates;
  candidates.reserve(static_cast<std::size_t>(vector_count));

  for (std::int64_t query_id = 0; query_id < query_count; ++query_id) {
    candidates.clear();
    const float* query = queries.data_handle() + (query_id * dimension);

    for (std::int64_t vector_id = 0; vector_id < vector_count; ++vector_id) {
      const float* vector = dataset.data() + (vector_id * dimension);
      float distance = 0.0F;
      for (std::int64_t dim = 0; dim < dimension; ++dim) {
        const float delta = query[dim] - vector[dim];
        distance += delta * delta;
      }
      candidates.emplace_back(distance, static_cast<std::uint32_t>(vector_id));
    }

    const auto result_count = std::min<std::int64_t>(k, vector_count);
    std::partial_sort(
        candidates.begin(),
        candidates.begin() + result_count,
        candidates.end(),
        [](const auto& lhs, const auto& rhs) {
          if (lhs.first == rhs.first) {
            return lhs.second < rhs.second;
          }
          return lhs.first < rhs.first;
        });

    for (std::int64_t result_id = 0; result_id < k; ++result_id) {
      const auto output_offset = (query_id * k) + result_id;
      if (result_id < result_count) {
        distances.data_handle()[output_offset] = candidates[result_id].first;
        neighbors.data_handle()[output_offset] = candidates[result_id].second;
      } else {
        distances.data_handle()[output_offset] =
            std::numeric_limits<float>::infinity();
        neighbors.data_handle()[output_offset] =
            std::numeric_limits<std::uint32_t>::max();
      }
    }
  }
}

// Metal dispatch counter — starts at 0, incremented only when a real
// MTLComputeCommandEncoder dispatch occurs. Currently always 0 because
// search() uses CPU brute-force. Green phase: metal_search.mm increments this.
static std::uint64_t g_metal_dispatch_count = 0;

std::uint64_t metal_dispatch_count() noexcept {
  return g_metal_dispatch_count;
}

void reset_metal_dispatch_count() noexcept {
  g_metal_dispatch_count = 0;
}

// ── Index serialization ────────────────────────────────────────────────────
// Binary format (little-endian):
//   [4]  magic "CGRA"
//   [4]  version = 1
//   [8]  N (rows), [8] D (cols), [4] G (graph_degree)
//   [4]  flags: bit0=has_dataset, bit1=has_nav
//   [N×G×4]  knn_graph (uint32)
//   if has_dataset: [N×D×4]  dataset (float32)
//   if has_nav: [4] n_nav, [n_nav×4] nav_nodes, [n_nav×D×4] nav_vectors

static void write_vec_u32(std::ofstream& f, const std::vector<std::uint32_t>& v) {
    f.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(std::uint32_t)));
}
static void write_vec_f32(std::ofstream& f, const std::vector<float>& v) {
    f.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(float)));
}
template<typename T>
static void write_val(std::ofstream& f, T v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template<typename T>
static T read_val(std::ifstream& f) {
    T v; f.read(reinterpret_cast<char*>(&v), sizeof(T)); return v;
}
static void read_vec_u32(std::ifstream& f, std::vector<std::uint32_t>& v, std::size_t n) {
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()),
           static_cast<std::streamsize>(n * sizeof(std::uint32_t)));
}
static void read_vec_f32(std::ifstream& f, std::vector<float>& v, std::size_t n) {
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()),
           static_cast<std::streamsize>(n * sizeof(float)));
}

void save_index(const index<float, std::uint32_t>& idx, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("save_index: cannot open " + path);

    f.write("CGRA", 4);
    write_val<std::uint32_t>(f, 1u);  // version
    write_val<std::int64_t>(f, idx.rows());
    write_val<std::int64_t>(f, idx.cols());
    write_val<std::uint32_t>(f, idx.graph_degree());

    const bool has_dataset = !idx.dataset().empty();
    const bool has_nav     = !idx.nav_nodes().empty();
    write_val<std::uint32_t>(f,
        (has_dataset ? 0x1u : 0u) | (has_nav ? 0x2u : 0u));

    if (has_dataset) write_vec_f32(f, idx.dataset());
    write_vec_u32(f, idx.knn_graph());
    if (has_nav) {
        write_val<std::uint32_t>(f, static_cast<std::uint32_t>(idx.nav_nodes().size()));
        write_vec_u32(f, idx.nav_nodes());
        write_vec_f32(f, idx.nav_vectors());
    }
    if (!f) throw std::runtime_error("save_index: write failed");
}

index<float, std::uint32_t> load_index(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load_index: cannot open " + path);

    char magic[4]; f.read(magic, 4);
    if (std::string(magic, 4) != "CGRA")
        throw std::runtime_error("load_index: invalid magic");
    const auto ver = read_val<std::uint32_t>(f);
    if (ver != 1) throw std::runtime_error("load_index: unsupported version");

    const auto N     = read_val<std::int64_t>(f);
    const auto D     = read_val<std::int64_t>(f);
    const auto G     = read_val<std::uint32_t>(f);
    const auto flags = read_val<std::uint32_t>(f);
    const bool has_dataset = (flags & 0x1u) != 0;
    const bool has_nav     = (flags & 0x2u) != 0;

    std::vector<float> dataset;
    if (has_dataset) read_vec_f32(f, dataset, static_cast<std::size_t>(N * D));

    index<float, std::uint32_t> idx(std::move(dataset), N, D);

    std::vector<std::uint32_t> knn;
    read_vec_u32(f, knn, static_cast<std::size_t>(N) * G);
    idx.set_knn_graph(std::move(knn), G);

    if (has_nav) {
        const auto n_nav = read_val<std::uint32_t>(f);
        std::vector<std::uint32_t> nav_n; read_vec_u32(f, nav_n, n_nav);
        std::vector<float>         nav_v; read_vec_f32(f, nav_v,
            static_cast<std::size_t>(n_nav) * static_cast<std::size_t>(D));
        idx.set_nav_nodes(std::move(nav_n));
        idx.set_nav_vectors(std::move(nav_v));
    }
    if (!f) throw std::runtime_error("load_index: read failed");
    return idx;
}

// Internal: called by metal_search.mm on every GPU kernel dispatch.
void increment_metal_dispatch_count() noexcept {
  ++g_metal_dispatch_count;
}

}  // namespace cuvs::neighbors::cagra

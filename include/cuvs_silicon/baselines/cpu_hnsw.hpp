#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace cuvs_silicon::baselines {

struct CpuHnswBuildParams {
  std::uint32_t m = 16;
  std::uint32_t ef_construction = 200;
  std::uint32_t worker_count = 1;
};

struct CpuHnswSearchParams {
  std::uint32_t ef_search = 128;
  std::uint32_t worker_count = 1;
};

struct CpuHnswIndexParameters {
  std::uint32_t m = 0;
  std::uint32_t ef_construction = 0;
  std::uint32_t worker_count = 0;
  std::size_t vector_count = 0;
  std::size_t dimension = 0;
};

class CpuHnswBaselineIndex {
 public:
  explicit CpuHnswBaselineIndex(CpuHnswIndexParameters params)
      : params_(params) {}

  [[nodiscard]] const CpuHnswIndexParameters& parameters() const noexcept {
    return params_;
  }

 private:
  CpuHnswIndexParameters params_;
};

class CpuHnswBaselineBuilder {
 public:
  CpuHnswBaselineBuilder() = default;

  explicit CpuHnswBaselineBuilder(CpuHnswBuildParams params) : params_(params) {
    validate(params_);
  }

  CpuHnswBaselineBuilder(CpuHnswBuildParams params, CpuHnswSearchParams search_params)
      : params_(params), search_params_(search_params) {
    validate(params_);
    validate(search_params_);
  }

  [[nodiscard]] const CpuHnswBuildParams& params() const noexcept {
    return params_;
  }

  [[nodiscard]] const CpuHnswSearchParams& search_params() const noexcept {
    return search_params_;
  }

  [[nodiscard]] CpuHnswBaselineIndex build(const float* vectors,
                                           std::size_t vector_count,
                                           std::size_t dimension) const {
    if (vectors == nullptr && vector_count != 0) {
      throw std::invalid_argument("vectors must not be null when vector_count is non-zero");
    }
    if (dimension == 0) {
      throw std::invalid_argument("dimension must be non-zero");
    }

    return CpuHnswBaselineIndex({
        params_.m,
        params_.ef_construction,
        params_.worker_count,
        vector_count,
        dimension,
    });
  }

  [[nodiscard]] CpuHnswBaselineIndex build(const std::vector<float>& vectors,
                                           std::size_t dimension) const {
    if (dimension == 0) {
      throw std::invalid_argument("dimension must be non-zero");
    }
    if (vectors.size() % dimension != 0) {
      throw std::invalid_argument("vectors size must be divisible by dimension");
    }

    return build(vectors.data(), vectors.size() / dimension, dimension);
  }

  template <typename HnswIndex>
  [[nodiscard]] auto search(HnswIndex& index, const float* query, std::size_t k) const
      -> decltype(index.searchKnn(query, k)) {
    if (query == nullptr) {
      throw std::invalid_argument("query must not be null");
    }
    if (k == 0) {
      throw std::invalid_argument("k must be non-zero");
    }

    set_search_workers(index, static_cast<std::size_t>(search_params_.worker_count));
    index.setEf(static_cast<std::size_t>(search_params_.ef_search));
    return index.searchKnn(query, k);
  }

 private:
  template <typename HnswIndex, typename = void>
  struct HasSetNumThreads : std::false_type {};

  template <typename HnswIndex>
  struct HasSetNumThreads<
      HnswIndex,
      std::void_t<decltype(std::declval<HnswIndex&>().setNumThreads(std::declval<std::size_t>()))>>
      : std::true_type {};

  template <typename HnswIndex>
  static void set_search_workers(HnswIndex& index, std::size_t worker_count) {
    if constexpr (HasSetNumThreads<HnswIndex>::value) {
      index.setNumThreads(worker_count);
    }
  }

  static void validate(const CpuHnswBuildParams& params) {
    if (params.m == 0) {
      throw std::invalid_argument("m must be non-zero");
    }
    if (params.ef_construction == 0) {
      throw std::invalid_argument("ef_construction must be non-zero");
    }
    if (params.worker_count != 1) {
      throw std::invalid_argument("CPU HNSW baseline indexing must use exactly one worker");
    }
  }

  static void validate(const CpuHnswSearchParams& params) {
    if (params.ef_search == 0) {
      throw std::invalid_argument("ef_search must be non-zero");
    }
    if (params.worker_count != 1) {
      throw std::invalid_argument("CPU HNSW baseline search must use exactly one worker");
    }
  }

  CpuHnswBuildParams params_;
  CpuHnswSearchParams search_params_;
};

}  // namespace cuvs_silicon::baselines

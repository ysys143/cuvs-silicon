#include <cassert>
#include <cstdint>
#include <algorithm>
#include <exception>
#include <limits>
#include <string>
#include <vector>

#include <faiss/Index.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/FaissException.h>

#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/resources.hpp>

namespace faiss {

FaissException::FaissException(const std::string& message) : msg(message) {}

FaissException::FaissException(const std::string& message,
                               const char* function_name,
                               const char* file,
                               int line)
    : msg(message + " in " + function_name + " at " + file + ":" +
          std::to_string(line)) {}

const char* FaissException::what() const noexcept { return msg.c_str(); }

Index::~Index() = default;

void Index::train(idx_t, const float*) {}

void Index::add_with_ids(idx_t, const float*, const idx_t*) { assert(false); }

void Index::range_search(
    idx_t, const float*, float, RangeSearchResult*, const SearchParameters*) const {
  assert(false);
}

void Index::assign(idx_t n, const float* x, idx_t* labels, idx_t k) const {
  std::vector<float> distances(static_cast<std::size_t>(n * k));
  search(n, x, k, distances.data(), labels);
}

size_t Index::remove_ids(const IDSelector&) {
  assert(false);
  return 0;
}

void Index::reconstruct(idx_t, float*) const { assert(false); }

void Index::reconstruct_batch(idx_t, const idx_t*, float*) const { assert(false); }

void Index::reconstruct_n(idx_t, idx_t, float*) const { assert(false); }

void Index::search_and_reconstruct(
    idx_t,
    const float*,
    idx_t,
    float*,
    idx_t*,
    float*,
    const SearchParameters*) const {
  assert(false);
}

void Index::compute_residual(const float*, float*, idx_t) const { assert(false); }

void Index::compute_residual_n(idx_t, const float*, float*, const idx_t*) const {
  assert(false);
}

DistanceComputer* Index::get_distance_computer() const {
  assert(false);
  return nullptr;
}

size_t Index::sa_code_size() const {
  assert(false);
  return 0;
}

void Index::sa_encode(idx_t, const float*, uint8_t*) const { assert(false); }

void Index::sa_decode(idx_t, const uint8_t*, float*) const { assert(false); }

void Index::merge_from(Index&, idx_t) { assert(false); }

void Index::check_compatible_for_merge(const Index&) const { assert(false); }

}  // namespace faiss

namespace {

template <typename Fn>
void assert_faiss_exception_contains(Fn&& fn, const std::string& expected) {
  try {
    fn();
  } catch (const faiss::FaissException& error) {
    assert(std::string(error.what()).find(expected) != std::string::npos);
    return;
  } catch (const std::exception&) {
    assert(false);
  }
  assert(false);
}

void assert_faiss_l2_results_are_sorted(const std::vector<float>& distances,
                                        const std::vector<faiss::idx_t>& labels,
                                        faiss::idx_t query_count,
                                        faiss::idx_t k) {
  assert(distances.size() == static_cast<std::size_t>(query_count * k));
  assert(labels.size() == static_cast<std::size_t>(query_count * k));

  for (faiss::idx_t query_id = 0; query_id < query_count; ++query_id) {
    const auto row_offset = query_id * k;
    for (faiss::idx_t result_id = 1; result_id < k; ++result_id) {
      const auto previous = row_offset + result_id - 1;
      const auto current = row_offset + result_id;
      assert(distances[previous] <= distances[current]);
      if (distances[previous] == distances[current] &&
          labels[previous] >= 0 && labels[current] >= 0) {
        assert(labels[previous] < labels[current]);
      }
    }
  }
}

class FaissFacingCagraIndex final : public faiss::Index {
 public:
  explicit FaissFacingCagraIndex(faiss::idx_t dimension,
                                 bool requires_training = false)
      : faiss::Index(dimension, faiss::METRIC_L2),
        requires_training_(requires_training) {
    is_trained = !requires_training_;
  }

  void train(faiss::idx_t vector_count, const float* vectors) override {
    assert(requires_training_);
    assert(!is_trained);
    assert(vector_count > 0);
    assert(vectors != nullptr);

    is_trained = true;
    ++train_call_count_;
  }

  void add(faiss::idx_t vector_count, const float* vectors) override {
    assert(vector_count > 0);
    assert(vectors != nullptr);
    if (requires_training_ && !is_trained) {
      train(vector_count, vectors);
    }
    assert(is_trained);

    cuvs::neighbors::cagra::index_params params;
    params.metric = cuvs::distance::DistanceType::L2Expanded;
    const auto old_size = added_vectors_.size();
    added_vectors_.resize(
        old_size + static_cast<std::size_t>(vector_count * d));
    std::copy(
        vectors,
        vectors + vector_count * d,
        added_vectors_.begin() + old_size);

    auto dataset = raft::make_device_matrix_view<const float, std::int64_t>(
        added_vectors_.data(), ntotal + vector_count, d);

    index_ = cuvs::neighbors::cagra::build(resources_, params, dataset);
    ntotal += vector_count;
  }

  void search(faiss::idx_t query_count,
              const float* queries,
              faiss::idx_t k,
              float* distances,
              faiss::idx_t* labels,
              const faiss::SearchParameters* params = nullptr) const override {
    assert(params == nullptr);
    assert(query_count > 0);
    assert(queries != nullptr);
    FAISS_THROW_IF_NOT(k > 0);
    assert(distances != nullptr);
    assert(labels != nullptr);

    cuvs::neighbors::cagra::search_params cagra_params;
    cagra_params.itopk_size = static_cast<std::uint64_t>(k);

    std::vector<std::uint32_t> cagra_labels(
        static_cast<std::size_t>(query_count * k));
    auto query_view = raft::make_device_matrix_view<const float, std::int64_t>(
        queries, query_count, d);
    auto label_view = raft::make_device_matrix_view<std::uint32_t, std::int64_t>(
        cagra_labels.data(), query_count, k);
    auto distance_view = raft::make_device_matrix_view<float, std::int64_t>(
        distances, query_count, k);

    cuvs::neighbors::cagra::search(
        resources_, cagra_params, index_, query_view, label_view, distance_view);

    for (faiss::idx_t offset = 0; offset < query_count * k; ++offset) {
      labels[offset] =
          cagra_labels[offset] == std::numeric_limits<std::uint32_t>::max()
              ? static_cast<faiss::idx_t>(-1)
              : static_cast<faiss::idx_t>(cagra_labels[offset]);
    }
  }

  void reset() override {
    ntotal = 0;
    added_vectors_.clear();
    index_ = {};
  }

  faiss::idx_t indexed_vector_count() const noexcept { return index_.rows(); }
  int train_call_count() const noexcept { return train_call_count_; }

 private:
  bool requires_training_ = false;
  int train_call_count_ = 0;
  raft::resources resources_;
  std::vector<float> added_vectors_;
  cuvs::neighbors::cagra::index<float, std::uint32_t> index_;
};

}  // namespace

int main() {
  constexpr faiss::idx_t dimension = 2;
  constexpr faiss::idx_t first_batch_count = 2;
  constexpr faiss::idx_t second_batch_count = 2;
  constexpr faiss::idx_t vector_count = first_batch_count + second_batch_count;
  constexpr faiss::idx_t query_count = 2;
  constexpr faiss::idx_t k = 2;

  const std::vector<float> database{
      0.0F, 0.0F,
      10.0F, 0.0F,
      0.0F, 10.0F,
      2.0F, 2.0F,
  };
  const std::vector<float> queries{
      0.0F, 0.0F,
      9.0F, 1.0F,
  };

  {
    raft::resources resources;
    cuvs::neighbors::cagra::index_params params;
    params.metric = cuvs::distance::DistanceType::L2Expanded;
    params.attach_dataset_on_build = false;

    const auto dataset = raft::make_device_matrix_view<const float, std::int64_t>(
        database.data(), vector_count, dimension);
    const auto lifecycle_index =
        cuvs::neighbors::cagra::build(resources, params, dataset);

    assert(lifecycle_index.rows() == vector_count);
    assert(lifecycle_index.cols() == dimension);
    assert(lifecycle_index.dataset().empty());
  }

  FaissFacingCagraIndex training_required_index(dimension, true);
  faiss::Index& training_required_faiss_index = training_required_index;
  assert(!training_required_faiss_index.is_trained);
  training_required_faiss_index.add(vector_count, database.data());
  assert(training_required_faiss_index.is_trained);
  assert(training_required_index.train_call_count() == 1);
  assert(training_required_faiss_index.ntotal == vector_count);
  assert(training_required_index.indexed_vector_count() == vector_count);

  FaissFacingCagraIndex incrementally_trained_index(dimension, true);
  faiss::Index& incremental_faiss_index = incrementally_trained_index;
  incremental_faiss_index.add(first_batch_count, database.data());
  incremental_faiss_index.add(
      second_batch_count, database.data() + first_batch_count * dimension);
  assert(incremental_faiss_index.is_trained);
  assert(incrementally_trained_index.train_call_count() == 1);
  assert(incremental_faiss_index.ntotal == vector_count);
  assert(incrementally_trained_index.indexed_vector_count() == vector_count);

  FaissFacingCagraIndex index(dimension);
  faiss::Index& faiss_index = index;
  faiss_index.add(first_batch_count, database.data());
  faiss_index.add(second_batch_count, database.data() + first_batch_count * dimension);
  assert(faiss_index.is_trained);
  assert(faiss_index.ntotal == vector_count);
  assert(index.indexed_vector_count() == vector_count);

  std::vector<float> distances(static_cast<std::size_t>(query_count * k));
  std::vector<faiss::idx_t> labels(static_cast<std::size_t>(query_count * k));
  faiss_index.search(query_count, queries.data(), k, distances.data(), labels.data());

  assert(faiss_index.d == dimension);
  assert(faiss_index.ntotal == vector_count);
  assert(labels.size() == static_cast<std::size_t>(query_count * k));
  assert(distances.size() == static_cast<std::size_t>(query_count * k));

  assert(labels[0] == 0);
  assert(distances[0] == 0.0F);
  assert(labels[1] == 3);
  assert(distances[1] == 8.0F);
  assert(labels[2] == 1);
  assert(distances[2] == 2.0F);
  assert(labels[3] == 3);
  assert(distances[3] == 50.0F);
  assert_faiss_l2_results_are_sorted(distances, labels, query_count, k);

  constexpr faiss::idx_t ordered_k = vector_count;
  const std::vector<float> ordering_query{
      0.0F, 0.0F,
  };
  std::vector<float> ordering_distances(
      static_cast<std::size_t>(ordered_k), -7.0F);
  std::vector<faiss::idx_t> ordering_labels(
      static_cast<std::size_t>(ordered_k), -7);
  faiss_index.search(
      1,
      ordering_query.data(),
      ordered_k,
      ordering_distances.data(),
      ordering_labels.data());

  assert_faiss_l2_results_are_sorted(
      ordering_distances, ordering_labels, 1, ordered_k);
  assert(ordering_labels[0] == 0);
  assert(ordering_distances[0] == 0.0F);
  assert(ordering_labels[1] == 3);
  assert(ordering_distances[1] == 8.0F);
  assert(ordering_labels[2] == 1);
  assert(ordering_distances[2] == 100.0F);
  assert(ordering_labels[3] == 2);
  assert(ordering_distances[3] == 100.0F);

  constexpr faiss::idx_t single_query_count = 1;
  constexpr faiss::idx_t oversized_k = vector_count + 1;
  std::vector<float> single_distances(
      static_cast<std::size_t>(single_query_count * oversized_k),
      -7.0F);
  std::vector<faiss::idx_t> single_labels(
      static_cast<std::size_t>(single_query_count * oversized_k),
      -7);
  faiss_index.search(
      single_query_count,
      queries.data(),
      oversized_k,
      single_distances.data(),
      single_labels.data());
  assert(single_distances.size() ==
         static_cast<std::size_t>(single_query_count * oversized_k));
  assert(single_labels.size() ==
         static_cast<std::size_t>(single_query_count * oversized_k));
  assert(single_labels[0] == 0);
  assert(single_distances[0] == 0.0F);
  assert(single_labels[oversized_k - 1] == -1);
  assert(single_distances[oversized_k - 1] ==
         std::numeric_limits<float>::infinity());

  constexpr faiss::idx_t batched_shape_query_count = 2;
  std::vector<float> batched_distances(
      static_cast<std::size_t>(batched_shape_query_count * oversized_k),
      -7.0F);
  std::vector<faiss::idx_t> batched_labels(
      static_cast<std::size_t>(batched_shape_query_count * oversized_k),
      -7);
  faiss_index.search(
      batched_shape_query_count,
      queries.data(),
      oversized_k,
      batched_distances.data(),
      batched_labels.data());
  assert(batched_distances.size() ==
         static_cast<std::size_t>(batched_shape_query_count * oversized_k));
  assert(batched_labels.size() ==
         static_cast<std::size_t>(batched_shape_query_count * oversized_k));
  assert(batched_labels[0] == 0);
  assert(batched_distances[0] == 0.0F);
  assert(batched_labels[oversized_k - 1] == -1);
  assert(batched_distances[oversized_k - 1] ==
         std::numeric_limits<float>::infinity());
  assert(batched_labels[(batched_shape_query_count * oversized_k) - 1] == -1);
  assert(batched_distances[(batched_shape_query_count * oversized_k) - 1] ==
         std::numeric_limits<float>::infinity());

  std::vector<float> invalid_k_distances(static_cast<std::size_t>(query_count));
  std::vector<faiss::idx_t> invalid_k_labels(static_cast<std::size_t>(query_count));
  assert_faiss_exception_contains(
      [&] {
        faiss_index.search(
            query_count,
            queries.data(),
            0,
            invalid_k_distances.data(),
            invalid_k_labels.data());
      },
      "k > 0");

  return 0;
}

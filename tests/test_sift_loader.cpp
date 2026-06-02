#include "cuvs_silicon/sift_loader.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::filesystem::path temp_path(const std::string& name) {
  return std::filesystem::temp_directory_path() / name;
}

void write_fvecs(const std::filesystem::path& path,
                 const std::vector<std::vector<float>>& rows,
                 std::int32_t dimension_override = -1) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to create test fvecs file");
  }

  for (const auto& row : rows) {
    const std::int32_t dimension =
        dimension_override >= 0 ? dimension_override
                                : static_cast<std::int32_t>(row.size());
    output.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
    output.write(reinterpret_cast<const char*>(row.data()),
                 static_cast<std::streamsize>(row.size() * sizeof(float)));
  }
}

void write_ivecs(const std::filesystem::path& path,
                 const std::vector<std::vector<std::int32_t>>& rows,
                 std::int32_t neighbor_count_override = -1) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to create test ivecs file");
  }

  for (const auto& row : rows) {
    const std::int32_t neighbor_count =
        neighbor_count_override >= 0
            ? neighbor_count_override
            : static_cast<std::int32_t>(row.size());
    output.write(reinterpret_cast<const char*>(&neighbor_count),
                 sizeof(neighbor_count));
    output.write(reinterpret_cast<const char*>(row.data()),
                 static_cast<std::streamsize>(row.size() *
                                              sizeof(std::int32_t)));
  }
}

void write_sift1m_query_fvecs(const std::filesystem::path& path) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to create test query fvecs file");
  }

  for (std::size_t row = 0; row < cuvs_silicon::kSift1MQueryVectorCount; ++row) {
    const std::int32_t dimension =
        static_cast<std::int32_t>(cuvs_silicon::kSift1MVectorDimension);
    output.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
    for (std::size_t col = 0; col < cuvs_silicon::kSift1MVectorDimension;
         ++col) {
      const float value =
          static_cast<float>((row * cuvs_silicon::kSift1MVectorDimension) + col);
      output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
  }
}

void write_sparse_sift1m_base_fvecs_shape_fixture(
    const std::filesystem::path& path) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to create test base fvecs file");
  }

  const std::int32_t dimension =
      static_cast<std::int32_t>(cuvs_silicon::kSift1MVectorDimension);
  const auto payload_bytes =
      static_cast<std::streamoff>(cuvs_silicon::kSift1MVectorDimension *
                                  sizeof(float));
  const char payload_tail = '\0';
  for (std::size_t row = 0; row < cuvs_silicon::kSift1MBaseVectorCount; ++row) {
    output.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
    output.seekp(payload_bytes - 1, std::ios::cur);
    output.write(&payload_tail, sizeof(payload_tail));
  }
}

void write_sift1m_groundtruth_ivecs(const std::filesystem::path& path) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to create test ground-truth ivecs file");
  }

  for (std::size_t row = 0; row < cuvs_silicon::kSift1MQueryVectorCount; ++row) {
    const std::int32_t neighbor_count =
        static_cast<std::int32_t>(
            cuvs_silicon::kSift1MGroundTruthNeighborCount);
    output.write(reinterpret_cast<const char*>(&neighbor_count),
                 sizeof(neighbor_count));
    for (std::size_t col = 0;
         col < cuvs_silicon::kSift1MGroundTruthNeighborCount; ++col) {
      const std::int32_t neighbor =
          static_cast<std::int32_t>(
              (row * cuvs_silicon::kSift1MGroundTruthNeighborCount) + col);
      output.write(reinterpret_cast<const char*>(&neighbor), sizeof(neighbor));
    }
  }
}

void expect_true(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Fn>
void expect_load_error(Fn&& fn, const std::string& message) {
  try {
    fn();
  } catch (const cuvs_silicon::SiftLoadError&) {
    return;
  }
  throw std::runtime_error(message);
}

template <typename Fn>
void expect_load_error_contains(Fn&& fn,
                                const std::string& expected_substring,
                                const std::string& message) {
  try {
    fn();
  } catch (const cuvs_silicon::SiftLoadError& error) {
    if (std::string(error.what()).find(expected_substring) !=
        std::string::npos) {
      return;
    }
    throw std::runtime_error(message + ": " + error.what());
  }
  throw std::runtime_error(message);
}

void test_loads_expected_shape_and_values() {
  const auto path = temp_path("cuvs_silicon_sift_loader_valid.fvecs");
  write_fvecs(path, {{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}});

  const auto dataset = cuvs_silicon::load_fvecs(path.string(), 2, 3);

  expect_true(dataset.rows == 2, "row count was not preserved");
  expect_true(dataset.cols == 3, "column count was not preserved");
  expect_true(dataset.values == std::vector<float>({1.0F, 2.0F, 3.0F, 4.0F,
                                                    5.0F, 6.0F}),
              "loaded vector payload did not match fvecs data");
  std::filesystem::remove(path);
}

void test_loads_128d_fvecs_contiguously() {
  const auto path = temp_path("cuvs_silicon_sift_loader_128d.fvecs");
  std::vector<float> first(cuvs_silicon::kSift1MVectorDimension);
  std::vector<float> second(cuvs_silicon::kSift1MVectorDimension);
  for (std::size_t col = 0; col < cuvs_silicon::kSift1MVectorDimension;
       ++col) {
    first[col] = static_cast<float>(col);
    second[col] =
        static_cast<float>(cuvs_silicon::kSift1MVectorDimension + col);
  }
  write_fvecs(path, {first, second});

  const auto dataset =
      cuvs_silicon::load_fvecs(path.string(), 2,
                              cuvs_silicon::kSift1MVectorDimension);

  expect_true(dataset.rows == 2, "128d fvecs row count was not preserved");
  expect_true(dataset.cols == cuvs_silicon::kSift1MVectorDimension,
              "128d fvecs dimension was not preserved");
  expect_true(dataset.values.size() ==
                  2 * cuvs_silicon::kSift1MVectorDimension,
              "128d fvecs payload size was wrong");
  for (std::size_t i = 1; i < dataset.values.size(); ++i) {
    expect_true(&dataset.values[i] == &dataset.values[0] + i,
                "128d fvecs payload is not contiguous");
  }
  expect_true(dataset.values.front() == 0.0F,
              "128d fvecs first payload value was wrong");
  expect_true(dataset.values.back() ==
                  static_cast<float>(dataset.values.size() - 1),
              "128d fvecs final payload value was wrong");
  std::filesystem::remove(path);
}

void test_rejects_wrong_dimension() {
  const auto path = temp_path("cuvs_silicon_sift_loader_wrong_dim.fvecs");
  write_fvecs(path, {{1.0F, 2.0F, 3.0F}}, 4);

  expect_load_error(
      [&] { (void)cuvs_silicon::load_fvecs(path.string(), 1, 3); },
      "loader accepted an fvecs row with the wrong dimension");
  std::filesystem::remove(path);
}

void test_rejects_wrong_vector_count() {
  const auto path = temp_path("cuvs_silicon_sift_loader_wrong_count.fvecs");
  write_fvecs(path, {{1.0F, 2.0F, 3.0F}});

  expect_load_error(
      [&] { (void)cuvs_silicon::load_fvecs(path.string(), 2, 3); },
      "loader accepted fewer vectors than the expected SIFT count");
  std::filesystem::remove(path);
}

void test_rejects_trailing_records() {
  const auto path = temp_path("cuvs_silicon_sift_loader_extra.fvecs");
  write_fvecs(path, {{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}});

  expect_load_error(
      [&] { (void)cuvs_silicon::load_fvecs(path.string(), 1, 3); },
      "loader accepted more vectors than the expected SIFT count");
  std::filesystem::remove(path);
}

void test_sift1m_constants_match_dataset_contract() {
  expect_true(cuvs_silicon::kSift1MBaseVectorCount == 1'000'000,
              "SIFT-1M base vector count constant changed");
  expect_true(cuvs_silicon::kSift1MQueryVectorCount == 10'000,
              "SIFT-1M query vector count constant changed");
  expect_true(cuvs_silicon::kSift1MVectorDimension == 128,
              "SIFT-1M vector dimension constant changed");
  expect_true(cuvs_silicon::kSift1MGroundTruthNeighborCount == 100,
              "SIFT-1M ground-truth neighbor count constant changed");
}

void test_loads_sift1m_queries_with_expected_shape() {
  const auto path = temp_path("cuvs_silicon_sift_loader_queries.fvecs");
  write_sift1m_query_fvecs(path);

  const auto dataset = cuvs_silicon::load_sift1m_queries(path.string());

  expect_true(dataset.rows == cuvs_silicon::kSift1MQueryVectorCount,
              "SIFT-1M query loader did not preserve query count");
  expect_true(dataset.cols == cuvs_silicon::kSift1MVectorDimension,
              "SIFT-1M query loader did not preserve vector dimension");
  expect_true(dataset.values.size() ==
                  cuvs_silicon::kSift1MQueryVectorCount *
                      cuvs_silicon::kSift1MVectorDimension,
              "SIFT-1M query loader returned the wrong payload size");
  expect_true(dataset.values.front() == 0.0F,
              "SIFT-1M query loader returned the wrong first value");
  expect_true(dataset.values.back() ==
                  static_cast<float>(dataset.values.size() - 1),
              "SIFT-1M query loader returned the wrong final value");
  std::filesystem::remove(path);
}

void test_sift1m_query_loader_rejects_wrong_query_count() {
  const auto path =
      temp_path("cuvs_silicon_sift_loader_queries_wrong_count.fvecs");
  write_fvecs(path, std::vector<std::vector<float>>(
                        1, std::vector<float>(
                               cuvs_silicon::kSift1MVectorDimension, 1.0F)));

  expect_load_error(
      [&] { (void)cuvs_silicon::load_sift1m_queries(path.string()); },
      "SIFT-1M query loader accepted the wrong query count");
  std::filesystem::remove(path);
}

void test_sift1m_query_loader_rejects_wrong_dimension() {
  const auto path = temp_path("cuvs_silicon_sift_loader_queries_wrong_dim.fvecs");
  write_fvecs(path, std::vector<std::vector<float>>(
                        1, std::vector<float>(
                               cuvs_silicon::kSift1MVectorDimension - 1,
                               1.0F)));

  expect_load_error(
      [&] { (void)cuvs_silicon::load_sift1m_queries(path.string()); },
      "SIFT-1M query loader accepted the wrong vector dimension");
  std::filesystem::remove(path);
}

void test_sift1m_base_loader_rejects_wrong_base_count() {
  const auto path = temp_path("cuvs_silicon_sift_loader_base_wrong_count.fvecs");
  write_fvecs(path, std::vector<std::vector<float>>(
                        1, std::vector<float>(
                               cuvs_silicon::kSift1MVectorDimension, 1.0F)));

  expect_load_error(
      [&] { (void)cuvs_silicon::load_sift1m_base(path.string()); },
      "SIFT-1M base loader accepted the wrong base vector count");
  std::filesystem::remove(path);
}

void test_sift1m_base_loader_rejects_wrong_dimension() {
  const auto path = temp_path("cuvs_silicon_sift_loader_base_wrong_dim.fvecs");
  write_fvecs(path, std::vector<std::vector<float>>(
                        1, std::vector<float>(
                               cuvs_silicon::kSift1MVectorDimension - 1,
                               1.0F)));

  expect_load_error(
      [&] { (void)cuvs_silicon::load_sift1m_base(path.string()); },
      "SIFT-1M base loader accepted the wrong vector dimension");
  std::filesystem::remove(path);
}

void test_sift1m_base_shape_validation_accepts_exact_base_count() {
  const auto path = temp_path("cuvs_silicon_sift_loader_base_exact_count.fvecs");
  write_sparse_sift1m_base_fvecs_shape_fixture(path);

  cuvs_silicon::validate_sift1m_base_shape(path.string());

  std::filesystem::remove(path);
}

void test_loads_ivecs_expected_shape_and_values() {
  const auto path = temp_path("cuvs_silicon_sift_loader_valid.ivecs");
  write_ivecs(path, {{7, 8, 9}, {10, 11, 12}});

  const auto dataset = cuvs_silicon::load_ivecs(path.string(), 2, 3);

  expect_true(dataset.query_count == 2,
              "ivecs query count was not preserved");
  expect_true(dataset.neighbor_count == 3,
              "ivecs neighbor count was not preserved");
  expect_true(dataset.neighbors ==
                  std::vector<std::uint32_t>({7, 8, 9, 10, 11, 12}),
              "loaded neighbor payload did not match ivecs data");
  std::filesystem::remove(path);
}

void test_ivecs_rejects_wrong_neighbor_count() {
  const auto path = temp_path("cuvs_silicon_sift_loader_wrong_neighbors.ivecs");
  write_ivecs(path, {{1, 2, 3}}, 4);

  expect_load_error(
      [&] { (void)cuvs_silicon::load_ivecs(path.string(), 1, 3); },
      "loader accepted an ivecs row with the wrong neighbor count");
  std::filesystem::remove(path);
}

void test_ivecs_rejects_wrong_query_count() {
  const auto path = temp_path("cuvs_silicon_sift_loader_wrong_queries.ivecs");
  write_ivecs(path, {{1, 2, 3}});

  expect_load_error(
      [&] { (void)cuvs_silicon::load_ivecs(path.string(), 2, 3); },
      "loader accepted fewer ivecs rows than the expected query count");
  std::filesystem::remove(path);
}

void test_ivecs_rejects_negative_neighbor_index() {
  const auto path = temp_path("cuvs_silicon_sift_loader_negative_neighbor.ivecs");
  write_ivecs(path, {{1, -2, 3}});

  expect_load_error(
      [&] { (void)cuvs_silicon::load_ivecs(path.string(), 1, 3); },
      "loader accepted a negative ground-truth neighbor index");
  std::filesystem::remove(path);
}

void test_loads_sift1m_groundtruth_with_expected_shape() {
  const auto path = temp_path("cuvs_silicon_sift_loader_groundtruth.ivecs");
  write_sift1m_groundtruth_ivecs(path);

  const auto dataset = cuvs_silicon::load_sift1m_groundtruth(path.string());

  expect_true(dataset.query_count == cuvs_silicon::kSift1MQueryVectorCount,
              "SIFT-1M ground-truth loader did not preserve query count");
  expect_true(dataset.neighbor_count ==
                  cuvs_silicon::kSift1MGroundTruthNeighborCount,
              "SIFT-1M ground-truth loader did not preserve neighbor count");
  expect_true(dataset.neighbors.size() ==
                  cuvs_silicon::kSift1MQueryVectorCount *
                      cuvs_silicon::kSift1MGroundTruthNeighborCount,
              "SIFT-1M ground-truth loader returned the wrong payload size");
  expect_true(dataset.neighbors.front() == 0,
              "SIFT-1M ground-truth loader returned the wrong first neighbor");
  expect_true(dataset.neighbors.back() ==
                  static_cast<std::uint32_t>(dataset.neighbors.size() - 1),
              "SIFT-1M ground-truth loader returned the wrong final neighbor");
  std::filesystem::remove(path);
}

void test_sift1m_groundtruth_loader_rejects_wrong_query_count() {
  const auto path =
      temp_path("cuvs_silicon_sift_loader_groundtruth_wrong_queries.ivecs");
  write_ivecs(path, std::vector<std::vector<std::int32_t>>(
                        1, std::vector<std::int32_t>(
                               cuvs_silicon::kSift1MGroundTruthNeighborCount,
                               1)));

  expect_load_error(
      [&] { (void)cuvs_silicon::load_sift1m_groundtruth(path.string()); },
      "SIFT-1M ground-truth loader accepted the wrong query count");
  std::filesystem::remove(path);
}

void test_sift1m_groundtruth_loader_rejects_wrong_neighbor_count() {
  const auto path =
      temp_path("cuvs_silicon_sift_loader_groundtruth_wrong_neighbors.ivecs");
  write_ivecs(path, std::vector<std::vector<std::int32_t>>(
                        1, std::vector<std::int32_t>(
                               cuvs_silicon::kSift1MGroundTruthNeighborCount - 1,
                               1)));

  expect_load_error(
      [&] { (void)cuvs_silicon::load_sift1m_groundtruth(path.string()); },
      "SIFT-1M ground-truth loader accepted the wrong neighbor count");
  std::filesystem::remove(path);
}

void test_combined_loader_exposes_compatible_dataset() {
  const auto base_path = temp_path("cuvs_silicon_sift_loader_combined_base.fvecs");
  const auto query_path =
      temp_path("cuvs_silicon_sift_loader_combined_queries.fvecs");
  const auto groundtruth_path =
      temp_path("cuvs_silicon_sift_loader_combined_groundtruth.ivecs");
  write_fvecs(base_path, {{1.0F, 2.0F}, {3.0F, 4.0F}, {5.0F, 6.0F}});
  write_fvecs(query_path, {{7.0F, 8.0F}, {9.0F, 10.0F}});
  write_ivecs(groundtruth_path, {{0, 2}, {1, 0}});

  const auto dataset = cuvs_silicon::load_sift_dataset(
      base_path.string(), query_path.string(), groundtruth_path.string(), 3, 2,
      2, 2);

  expect_true(dataset.base.rows == 3, "combined loader lost base row count");
  expect_true(dataset.base.cols == 2, "combined loader lost base dimension");
  expect_true(dataset.queries.rows == 2,
              "combined loader lost query row count");
  expect_true(dataset.queries.cols == 2,
              "combined loader lost query dimension");
  expect_true(dataset.groundtruth.query_count == 2,
              "combined loader lost ground-truth query count");
  expect_true(dataset.groundtruth.neighbor_count == 2,
              "combined loader lost ground-truth neighbor count");
  expect_true(dataset.base.values == std::vector<float>({1.0F, 2.0F, 3.0F,
                                                         4.0F, 5.0F, 6.0F}),
              "combined loader changed base payload");
  expect_true(dataset.queries.values ==
                  std::vector<float>({7.0F, 8.0F, 9.0F, 10.0F}),
              "combined loader changed query payload");
  expect_true(dataset.groundtruth.neighbors ==
                  std::vector<std::uint32_t>({0, 2, 1, 0}),
              "combined loader changed ground-truth payload");
  std::filesystem::remove(base_path);
  std::filesystem::remove(query_path);
  std::filesystem::remove(groundtruth_path);
}

void test_combined_loader_rejects_groundtruth_outside_base_rows() {
  const auto base_path =
      temp_path("cuvs_silicon_sift_loader_combined_bad_gt_base.fvecs");
  const auto query_path =
      temp_path("cuvs_silicon_sift_loader_combined_bad_gt_queries.fvecs");
  const auto groundtruth_path =
      temp_path("cuvs_silicon_sift_loader_combined_bad_gt.ivecs");
  write_fvecs(base_path, {{1.0F, 2.0F}, {3.0F, 4.0F}});
  write_fvecs(query_path, {{5.0F, 6.0F}});
  write_ivecs(groundtruth_path, {{0, 2}});

  expect_load_error_contains(
      [&] {
        (void)cuvs_silicon::load_sift_dataset(
            base_path.string(), query_path.string(), groundtruth_path.string(),
            2, 1, 2, 2);
      },
      "outside base",
      "combined loader accepted a ground-truth neighbor outside base rows");
  std::filesystem::remove(base_path);
  std::filesystem::remove(query_path);
  std::filesystem::remove(groundtruth_path);
}

}  // namespace

int main() {
  try {
    test_loads_expected_shape_and_values();
    test_loads_128d_fvecs_contiguously();
    test_rejects_wrong_dimension();
    test_rejects_wrong_vector_count();
    test_rejects_trailing_records();
    test_sift1m_constants_match_dataset_contract();
    test_loads_sift1m_queries_with_expected_shape();
    test_sift1m_query_loader_rejects_wrong_query_count();
    test_sift1m_query_loader_rejects_wrong_dimension();
    test_sift1m_base_loader_rejects_wrong_base_count();
    test_sift1m_base_loader_rejects_wrong_dimension();
    test_sift1m_base_shape_validation_accepts_exact_base_count();
    test_loads_ivecs_expected_shape_and_values();
    test_ivecs_rejects_wrong_neighbor_count();
    test_ivecs_rejects_wrong_query_count();
    test_ivecs_rejects_negative_neighbor_index();
    test_loads_sift1m_groundtruth_with_expected_shape();
    test_sift1m_groundtruth_loader_rejects_wrong_query_count();
    test_sift1m_groundtruth_loader_rejects_wrong_neighbor_count();
    test_combined_loader_exposes_compatible_dataset();
    test_combined_loader_rejects_groundtruth_outside_base_rows();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

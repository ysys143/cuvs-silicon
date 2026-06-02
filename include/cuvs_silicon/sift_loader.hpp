#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuvs_silicon {

constexpr std::size_t kSift1MBaseVectorCount = 1'000'000;
constexpr std::size_t kSift1MQueryVectorCount = 10'000;
constexpr std::size_t kSift1MVectorDimension = 128;
constexpr std::size_t kSift1MGroundTruthNeighborCount = 100;

class SiftLoadError : public std::runtime_error {
public:
  explicit SiftLoadError(const std::string& message);
};

struct FloatVectorDataset {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<float> values;
};

struct GroundTruthDataset {
  std::size_t query_count = 0;
  std::size_t neighbor_count = 0;
  std::vector<std::uint32_t> neighbors;
};

struct SiftDataset {
  FloatVectorDataset base;
  FloatVectorDataset queries;
  GroundTruthDataset groundtruth;
};

FloatVectorDataset load_fvecs(const std::string& path,
                              std::size_t expected_rows,
                              std::size_t expected_cols);

void validate_fvecs_shape(const std::string& path,
                          std::size_t expected_rows,
                          std::size_t expected_cols);

GroundTruthDataset load_ivecs(const std::string& path,
                              std::size_t expected_rows,
                              std::size_t expected_cols);

SiftDataset load_sift_dataset(const std::string& base_path,
                              const std::string& query_path,
                              const std::string& groundtruth_path,
                              std::size_t expected_base_rows,
                              std::size_t expected_query_rows,
                              std::size_t expected_cols,
                              std::size_t expected_groundtruth_cols);

FloatVectorDataset load_sift1m_base(const std::string& path);

void validate_sift1m_base_shape(const std::string& path);

FloatVectorDataset load_sift1m_queries(const std::string& path);

GroundTruthDataset load_sift1m_groundtruth(const std::string& path);

SiftDataset load_sift1m_dataset(const std::string& base_path,
                                const std::string& query_path,
                                const std::string& groundtruth_path);

}  // namespace cuvs_silicon

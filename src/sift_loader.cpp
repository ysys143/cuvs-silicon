#include "cuvs_silicon/sift_loader.hpp"

#include <fstream>
#include <limits>

namespace cuvs_silicon {
namespace {

std::size_t checked_product(std::size_t lhs, std::size_t rhs) {
  if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
    throw SiftLoadError("SIFT dataset shape overflows size_t");
  }
  return lhs * rhs;
}

}  // namespace

SiftLoadError::SiftLoadError(const std::string& message)
    : std::runtime_error(message) {}

FloatVectorDataset load_fvecs(const std::string& path,
                              std::size_t expected_rows,
                              std::size_t expected_cols) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw SiftLoadError("failed to open fvecs file: " + path);
  }

  const std::size_t value_count = checked_product(expected_rows, expected_cols);
  FloatVectorDataset dataset;
  dataset.rows = expected_rows;
  dataset.cols = expected_cols;
  dataset.values.reserve(value_count);

  for (std::size_t row = 0; row < expected_rows; ++row) {
    std::int32_t dimension = 0;
    input.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
    if (!input) {
      throw SiftLoadError("fvecs file ended before row dimension header");
    }
    if (dimension < 0 ||
        static_cast<std::size_t>(dimension) != expected_cols) {
      throw SiftLoadError("fvecs row dimension does not match expected width");
    }

    const auto old_size = dataset.values.size();
    dataset.values.resize(old_size + expected_cols);
    input.read(reinterpret_cast<char*>(dataset.values.data() + old_size),
               static_cast<std::streamsize>(expected_cols * sizeof(float)));
    if (!input) {
      throw SiftLoadError("fvecs file ended before row vector payload");
    }
  }

  char trailing = '\0';
  if (input.read(&trailing, 1)) {
    throw SiftLoadError("fvecs file has extra trailing bytes");
  }
  if (!input.eof()) {
    throw SiftLoadError("failed while checking fvecs file length");
  }

  return dataset;
}

GroundTruthDataset load_ivecs(const std::string& path,
                              std::size_t expected_rows,
                              std::size_t expected_cols) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw SiftLoadError("failed to open ivecs file: " + path);
  }

  const std::size_t value_count = checked_product(expected_rows, expected_cols);
  GroundTruthDataset dataset;
  dataset.query_count = expected_rows;
  dataset.neighbor_count = expected_cols;
  dataset.neighbors.reserve(value_count);

  for (std::size_t row = 0; row < expected_rows; ++row) {
    std::int32_t neighbor_count = 0;
    input.read(reinterpret_cast<char*>(&neighbor_count),
               sizeof(neighbor_count));
    if (!input) {
      throw SiftLoadError("ivecs file ended before row neighbor-count header");
    }
    if (neighbor_count < 0 ||
        static_cast<std::size_t>(neighbor_count) != expected_cols) {
      throw SiftLoadError(
          "ivecs row neighbor count does not match expected width");
    }

    for (std::size_t col = 0; col < expected_cols; ++col) {
      std::int32_t neighbor = 0;
      input.read(reinterpret_cast<char*>(&neighbor), sizeof(neighbor));
      if (!input) {
        throw SiftLoadError("ivecs file ended before row neighbor payload");
      }
      if (neighbor < 0) {
        throw SiftLoadError("ivecs neighbor index must be non-negative");
      }
      dataset.neighbors.push_back(static_cast<std::uint32_t>(neighbor));
    }
  }

  char trailing = '\0';
  if (input.read(&trailing, 1)) {
    throw SiftLoadError("ivecs file has extra trailing bytes");
  }
  if (!input.eof()) {
    throw SiftLoadError("failed while checking ivecs file length");
  }

  return dataset;
}

void validate_fvecs_shape(const std::string& path,
                          std::size_t expected_rows,
                          std::size_t expected_cols) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw SiftLoadError("failed to open fvecs file: " + path);
  }

  const std::size_t payload_bytes =
      checked_product(expected_cols, sizeof(float));
  if (payload_bytes >
      static_cast<std::size_t>(
          std::numeric_limits<std::streamoff>::max())) {
    throw SiftLoadError("SIFT fvecs row payload is too large to seek");
  }

  for (std::size_t row = 0; row < expected_rows; ++row) {
    std::int32_t dimension = 0;
    input.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
    if (!input) {
      throw SiftLoadError("fvecs file ended before row dimension header");
    }
    if (dimension < 0 ||
        static_cast<std::size_t>(dimension) != expected_cols) {
      throw SiftLoadError("fvecs row dimension does not match expected width");
    }
    input.seekg(static_cast<std::streamoff>(payload_bytes), std::ios::cur);
    if (!input) {
      throw SiftLoadError("fvecs file ended before row vector payload");
    }
  }

  char trailing = '\0';
  if (input.read(&trailing, 1)) {
    throw SiftLoadError("fvecs file has extra trailing bytes");
  }
  if (!input.eof()) {
    throw SiftLoadError("failed while checking fvecs file length");
  }
}

SiftDataset load_sift_dataset(const std::string& base_path,
                              const std::string& query_path,
                              const std::string& groundtruth_path,
                              std::size_t expected_base_rows,
                              std::size_t expected_query_rows,
                              std::size_t expected_cols,
                              std::size_t expected_groundtruth_cols) {
  SiftDataset dataset;
  dataset.base = load_fvecs(base_path, expected_base_rows, expected_cols);
  dataset.queries = load_fvecs(query_path, expected_query_rows, expected_cols);
  dataset.groundtruth =
      load_ivecs(groundtruth_path, expected_query_rows,
                 expected_groundtruth_cols);

  if (dataset.base.cols != dataset.queries.cols) {
    throw SiftLoadError("SIFT base and query dimensions are incompatible");
  }
  if (dataset.groundtruth.query_count != dataset.queries.rows) {
    throw SiftLoadError(
        "SIFT ground-truth query count is incompatible with queries");
  }
  for (const std::uint32_t neighbor : dataset.groundtruth.neighbors) {
    if (neighbor >= dataset.base.rows) {
      throw SiftLoadError("SIFT ground-truth neighbor is outside base rows");
    }
  }

  return dataset;
}

FloatVectorDataset load_sift1m_base(const std::string& path) {
  return load_fvecs(path, kSift1MBaseVectorCount, kSift1MVectorDimension);
}

void validate_sift1m_base_shape(const std::string& path) {
  validate_fvecs_shape(path, kSift1MBaseVectorCount,
                       kSift1MVectorDimension);
}

FloatVectorDataset load_sift1m_queries(const std::string& path) {
  return load_fvecs(path, kSift1MQueryVectorCount, kSift1MVectorDimension);
}

GroundTruthDataset load_sift1m_groundtruth(const std::string& path) {
  return load_ivecs(path, kSift1MQueryVectorCount,
                    kSift1MGroundTruthNeighborCount);
}

SiftDataset load_sift1m_dataset(const std::string& base_path,
                                const std::string& query_path,
                                const std::string& groundtruth_path) {
  return load_sift_dataset(base_path, query_path, groundtruth_path,
                           kSift1MBaseVectorCount, kSift1MQueryVectorCount,
                           kSift1MVectorDimension,
                           kSift1MGroundTruthNeighborCount);
}

}  // namespace cuvs_silicon

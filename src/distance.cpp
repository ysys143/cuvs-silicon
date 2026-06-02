#include "cuvs_silicon/distance.hpp"

#include <cstddef>
#include <stdexcept>

namespace cuvs_silicon {

float squared_l2_distance_128(const float* lhs, const float* rhs) {
  if (lhs == nullptr || rhs == nullptr) {
    throw std::invalid_argument(
        "squared_l2_distance_128 requires non-null vectors");
  }

  float distance = 0.0F;
  for (std::size_t dim = 0; dim < 128; ++dim) {
    const float delta = lhs[dim] - rhs[dim];
    distance += delta * delta;
  }
  return distance;
}

}  // namespace cuvs_silicon

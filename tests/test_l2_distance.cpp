#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

#include "cuvs_silicon/distance.hpp"

namespace {

void expect_near(float actual, float expected) {
  assert(std::fabs(actual - expected) < 1.0e-4F);
}

}  // namespace

int main() {
  constexpr std::size_t dimension = 128;

  std::vector<float> zeros(dimension, 0.0F);
  std::vector<float> ones(dimension, 1.0F);
  expect_near(cuvs_silicon::squared_l2_distance_128(zeros.data(), ones.data()),
              128.0F);

  std::vector<float> ascending(dimension);
  std::vector<float> shifted(dimension);
  float expected_shifted_distance = 0.0F;
  for (std::size_t dim = 0; dim < dimension; ++dim) {
    ascending[dim] = static_cast<float>(dim);
    shifted[dim] = static_cast<float>(dim) + 0.5F;
    expected_shifted_distance += 0.25F;
  }
  expect_near(cuvs_silicon::squared_l2_distance_128(ascending.data(),
                                                   shifted.data()),
              expected_shifted_distance);

  std::vector<float> mixed_a(dimension);
  std::vector<float> mixed_b(dimension);
  float expected_mixed_distance = 0.0F;
  for (std::size_t dim = 0; dim < dimension; ++dim) {
    const int lhs_value = static_cast<int>(dim % 7) - 3;
    const int rhs_value = static_cast<int>(dim % 5) - 2;
    mixed_a[dim] = static_cast<float>(lhs_value) * 0.25F;
    mixed_b[dim] = static_cast<float>(rhs_value) * -0.5F;
    const float delta = mixed_a[dim] - mixed_b[dim];
    expected_mixed_distance += delta * delta;
  }
  expect_near(
      cuvs_silicon::squared_l2_distance_128(mixed_a.data(), mixed_b.data()),
      expected_mixed_distance);

  return 0;
}

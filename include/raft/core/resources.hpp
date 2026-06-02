#pragma once

#include <cstddef>
#include <cstdint>

namespace raft {

class resources {
 public:
  resources() = default;

  void sync_stream() const {}
};

template <typename ElementType, typename IndexType = std::int64_t>
class device_matrix_view {
 public:
  using element_type = ElementType;
  using index_type = IndexType;

  constexpr device_matrix_view(ElementType* data,
                               IndexType rows,
                               IndexType cols) noexcept
      : data_(data), rows_(rows), cols_(cols) {}

  constexpr ElementType* data_handle() const noexcept { return data_; }
  constexpr ElementType* data() const noexcept { return data_; }
  constexpr IndexType extent(std::size_t axis) const noexcept {
    return axis == 0 ? rows_ : cols_;
  }
  constexpr IndexType size() const noexcept { return rows_ * cols_; }

 private:
  ElementType* data_;
  IndexType rows_;
  IndexType cols_;
};

template <typename ElementType, typename IndexType>
constexpr device_matrix_view<ElementType, IndexType> make_device_matrix_view(
    ElementType* data,
    IndexType rows,
    IndexType cols) noexcept {
  return device_matrix_view<ElementType, IndexType>(data, rows, cols);
}

}  // namespace raft

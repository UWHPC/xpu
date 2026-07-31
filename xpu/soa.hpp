#pragma once

#include <xpu/buffer.hpp>
#include <xpu/config.hpp>
#include <xpu/memory.hpp>
#include <cstddef>

namespace xpu {

template <typename T, std::size_t num_arrays>
class soa {
private:
  std::size_t count_;
  std::size_t stride_;
  xpu::buffer<T> buffer_;

public:
  soa() : count_{}, stride_{}, buffer_{} {}
  soa(soa&&) noexcept = default;
  soa& operator=(soa&&) noexcept = default;

  explicit soa(std::size_t count)
    : count_{count}
    , stride_{xpu::handle_pad<T>(count)}
    , buffer_{num_arrays * stride_}
  { }

  [[nodiscard]]
  std::size_t count() const {
    return count_;
  }

  [[nodiscard]]
  std::size_t storage_size() const {
    return num_arrays * stride_;
  }

  [[nodiscard]]
  std::size_t stride() const {
    return stride_;
  }

  [[nodiscard]]
  T* operator[](std::size_t arr_idx) {
    return buffer_.data() + arr_idx * stride();
  }

  [[nodiscard]]
  const T* operator[](std::size_t arr_idx) const {
    return buffer_.data() + arr_idx * stride();
  }

  // TODO: use an mdspan for view.
};

} // namespace xpu

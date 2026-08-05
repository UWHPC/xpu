#pragma once

#include <xpu/buffer.hpp>
#include <xpu/config.hpp>
#include <xpu/memory.hpp>
#include <cstddef>

namespace xpu {

template <typename T, std::size_t num_arrays>
class soa_view {
private:
  T* base_;
  std::size_t count_;

public:
  CUDA_CALLABLE
  explicit soa_view(T* base, std::size_t count)
    : base_{base}
    , count_{count}
  { }

  [[nodiscard]] CUDA_CALLABLE
  std::size_t count() const {
    return count_;
  }

  [[nodiscard]] CUDA_CALLABLE
  std::size_t stride() const {
    return xpu::handle_pad<T>(count_);
  }

  [[nodiscard]] CUDA_CALLABLE
  T* operator[](std::size_t arr_idx) {
    return base_ + arr_idx * stride();
  }

  [[nodiscard]] CUDA_CALLABLE
  const T* operator[](std::size_t arr_idx) const {
    return base_ + arr_idx * stride();
  }
};

template <typename T, std::size_t num_arrays>
class soa {
private:
  std::size_t count_;
  xpu::buffer<T> buffer_;

public:
  explicit soa(std::size_t count)
    : count_{count}
    , buffer_{num_arrays * xpu::handle_pad<T>(count)}
  { }

  [[nodiscard]]
  std::size_t count() const {
    return count_;
  }

  [[nodiscard]]
  std::size_t stride() const {
    return xpu::handle_pad<T>(count_);
  }

  [[nodiscard]]
  std::size_t storage_size() const {
    return num_arrays * stride();
  }

  [[nodiscard]]
  T* operator[](std::size_t arr_idx) {
    return buffer_.data() + arr_idx * stride();
  }

  [[nodiscard]]
  const T* operator[](std::size_t arr_idx) const {
    return buffer_.data() + arr_idx * stride();
  }

  [[nodiscard]]
  soa_view<T, num_arrays> view() {
    return soa_view<T, num_arrays>{buffer_.data(), count_};
  }

  [[nodiscard]]
  soa_view<const T, num_arrays> view() const {
    return soa_view<const T, num_arrays>{buffer_.data(), count_};
  }
};

} // namespace xpu

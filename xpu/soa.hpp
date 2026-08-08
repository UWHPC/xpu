#pragma once

#include <xpu/buffer.hpp>
#include <xpu/config.hpp>
#include <xpu/memory.hpp>

#if defined(XPU_CUDA)
  #include <cuda/std/array>
#else
  #include <array>
#endif

#include <cstddef>
#include <cassert>

namespace xpu {

using xstd::array;

template <typename T, std::size_t exposed_arrays>
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
    assert(arr_idx < exposed_arrays);
    return xpu::assume_aligned<T>(base_ + arr_idx * stride());
  }

  [[nodiscard]] CUDA_CALLABLE
  const T* operator[](std::size_t arr_idx) const {
    assert(arr_idx < exposed_arrays);
    return xpu::assume_aligned<T>(base_ + arr_idx * stride());
  }

  [[nodiscard]] CUDA_CALLABLE
  xpu::array<T*, exposed_arrays> pointers() {
    xpu::array<T*, exposed_arrays> ptrs{};

    for (auto i{0uz}; i < exposed_arrays; ++i) {
      ptrs[i] = xpu::assume_aligned<T>(base_ + i * stride());
    }

    return ptrs;
  }

  [[nodiscard]] CUDA_CALLABLE
  xpu::array<const T*, exposed_arrays> pointers() const {
    xpu::array<const T*, exposed_arrays> ptrs{};

    for (auto i{0uz}; i < exposed_arrays; ++i) {
      ptrs[i] = xpu::assume_aligned<T>(base_ + i * stride());
    }

    return ptrs;
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
    assert(arr_idx < num_arrays);
    return xpu::assume_aligned<T>(buffer_.data() + arr_idx * stride());
  }

  [[nodiscard]]
  const T* operator[](std::size_t arr_idx) const {
    assert(arr_idx < num_arrays);
    return xpu::assume_aligned<const T>(buffer_.data() + arr_idx * stride());
  }

  template <std::size_t exposed_arrays = num_arrays, std::size_t first_array = 0uz> [[nodiscard]]
  soa_view<T, exposed_arrays> view() {
    static_assert(first_array + exposed_arrays <= num_arrays);
    return soa_view<T, exposed_arrays>{buffer_.data() + first_array * stride(), count_};
  }

  template <std::size_t exposed_arrays = num_arrays, std::size_t first_array = 0uz> [[nodiscard]]
  soa_view<const T, exposed_arrays> view() const {
    static_assert(first_array + exposed_arrays <= num_arrays);
    return soa_view<const T, exposed_arrays>{buffer_.data() + first_array * stride(), count_};
  }
};

} // namespace xpu

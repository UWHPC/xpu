#pragma once

#include <cassert>
#include <cstddef>
#include <iostream>
#include <xpu/config.hpp>
#include <xpu/memory.hpp>

#include <type_traits>

namespace xpu {

template <typename T>
class buffer_view {
private:
  T* data_;
  std::size_t count_;

public:
  CUDA_CALLABLE
  explicit constexpr buffer_view(T* base, std::size_t count) noexcept
    : data_{base}
    , count_{count}
  { }

  [[nodiscard]] CUDA_CALLABLE
  constexpr auto count() const noexcept -> std::size_t {
    return count_;
  }

  template <typename Self> [[nodiscard]] CUDA_CALLABLE
  constexpr auto& operator[](this Self&& self, std::size_t idx) noexcept {
    assert(idx < self.count_);

    using element_t = std::conditional_t<
      std::is_const_v<std::remove_reference_t<Self>>,
      const T, T
    >;

    return static_cast<element_t*>(self.data_)[idx];
  }
};

template <typename T>
class buffer {
private:
  std::size_t count_;
  xpu::unique_ptr<T> data_;

public:
  explicit buffer(std::size_t count)
    : count_{count}
    , data_{xpu::make_unique<T>(xpu::handle_pad<T>(count))}
  { }

  [[nodiscard]]
  constexpr auto count() const noexcept -> std::size_t {
    return count_;
  }

  [[nodiscard]]
  constexpr auto capacity() const noexcept -> std::size_t {
    return xpu::handle_pad<T>(count_);
  }

  template <typename Self> [[nodiscard]]
  auto data(this Self&& self) noexcept {
    using element_t = std::conditional_t<
      std::is_const_v<std::remove_reference_t<Self>>,
      const T, T
    >;

    return xpu::assume_aligned<element_t>(self.data_.get());
  }

  template <typename Self> [[nodiscard]]
  auto view(this Self&& self) noexcept {
    using element_t = std::conditional_t<
      std::is_const_v<std::remove_reference_t<Self>>,
      const T, T
    >;

    return xpu::buffer_view<element_t>{
      self.data(), self.count_
    };
  }
};

} // namespace xpu

#pragma once

#include <xpu/array.hpp>
#include <xpu/config.hpp>
#include <xpu/launch.hpp>
#include <xpu/math.hpp>
#include <xpu/memory.hpp>

#include <cstddef>

namespace xpu {

namespace detail {

template <typename T>
struct load_element {
  const T* ptr;

  XPU_CUDA_CALLABLE
  auto operator()(const xpu::array<std::size_t, 1uz>& index) const -> T {
    return ptr[index[0]];
  }
};

template <typename T>
struct log_element {
  const T* ptr;

  XPU_CUDA_CALLABLE
  auto operator()(const xpu::array<std::size_t, 1uz>& index) const -> T {
    return xpu::log(ptr[index[0]]);
  }
};

} // namespace detail

template <arithmetic T>
inline auto accumulate(const T* ptr, std::size_t count) -> T {
  const auto range = xpu::range<1uz>{
    .begin = {0uz}, .end = {count}, .step = {1uz}
  };

  const auto contribution{xpu::detail::load_element<T>{ptr}};

  const auto sum_device{xpu::make_unique<T>(1uz)};

  xpu::parallel_reduce_sum(range, sum_device.get(), contribution);

  auto sum{T{0}};
  xpu::copy_n(&sum, sum_device.get(), 1uz);

  return sum;
}

template <arithmetic T>
inline auto average(const T* ptr, std::size_t count) -> T {
  return accumulate(ptr, count) / static_cast<T>(count);
}

template <std::floating_point T>
inline auto geometric_mean(const T* ptr, std::size_t count) -> T {
  const auto range = xpu::range<1uz>{
    .begin = {0uz}, .end = {count}, .step = {1uz}
  };

  const auto contribution{xpu::detail::log_element<T>{ptr}};

  const auto log_sum_device{xpu::make_unique<T>(1uz)};

  xpu::parallel_reduce_sum(range, log_sum_device.get(), contribution);

  auto log_sum{T{0}};
  xpu::copy_n(&log_sum, log_sum_device.get(), 1uz);

  return xpu::exp(log_sum / static_cast<T>(count));
}

} // namespace xpu

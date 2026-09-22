#pragma once

#include <xpu/array.hpp>
#include <xpu/config.hpp>
#include <xpu/launch.hpp>
#include <xpu/math.hpp>
#include <xpu/memory.hpp>

#include <cstddef>

namespace xpu {

template <arithmetic T>
inline auto accumulate(const T* ptr, std::size_t count) -> T {
  const auto range = xpu::range<1uz>{
    .begin = {0uz}, .end = {count}, .step = {1uz}
  };

  const auto contribution = [ptr] CUDA_CALLABLE (const xpu::array<std::size_t, 1uz>& index) -> T {
    return ptr[index[0]];
  };

  const auto scratch_bytes{xpu::parallel_reduce_sum_bytes<T>(range, contribution)};
  const auto scratch{xpu::make_unique<std::byte>(scratch_bytes)};
  const auto sum_device{xpu::make_unique<T>()};

  xpu::parallel_reduce_sum(range, sum_device.get(), contribution, scratch.get(), scratch_bytes);

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

  const auto contribution = [ptr] CUDA_CALLABLE (const xpu::array<std::size_t, 1uz>& index) -> T {
    return xpu::log(ptr[index[0]]);
  };

  const auto scratch_bytes{xpu::parallel_reduce_sum_bytes<T>(range, contribution)};
  const auto scratch{xpu::make_unique<std::byte>(scratch_bytes)};
  const auto log_sum_device{xpu::make_unique<T>()};

  xpu::parallel_reduce_sum(range, log_sum_device.get(), contribution, scratch.get(), scratch_bytes);

  auto log_sum{T{0}};
  xpu::copy_n(&log_sum, log_sum_device.get(), 1uz);

  return xpu::exp(log_sum / static_cast<T>(count));
}

} // namespace xpu

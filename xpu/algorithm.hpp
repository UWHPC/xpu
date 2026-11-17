#pragma once

#include <xpu/config.hpp>
#include <xpu/detail/checked.hpp>
#include <xpu/launch.hpp>

#if defined(XPU_CUDA)
  #include <cub/device/device_transform.cuh>
  #include <cuda/std/algorithm>
  #include <cuda/std/cstddef>
#else
  #include <algorithm>
  #include <cstddef>
  #include <cstring>
#endif


namespace xpu {

#if !defined(XPU_CUDA)
namespace detail {

// Threads pay off between these sizes: below, waking them costs more than the
// pass; above, one thread's non-temporal stores already fill the memory bus.
inline constexpr auto parallel_bytes_min{512uz * 1024uz};
inline constexpr auto parallel_bytes_max{16uz * 1024uz * 1024uz};
inline constexpr auto parallel_chunk_bytes{64uz * 1024uz};

[[nodiscard]]
inline constexpr auto parallel_bytes(std::size_t bytes) noexcept -> bool {
  return bytes >= parallel_bytes_min && bytes <= parallel_bytes_max;
}

// fcn(offset, length) over [0, count) in pieces of at most chunk
template <typename F>
inline auto for_each_chunk(
  std::size_t count,
  std::size_t chunk,
  bool parallel,
  F&& fcn
) -> void {
  if (!parallel) {
    fcn(0uz, count);

    return;
  }

  const auto chunks{xpu::ceiling_div(count, chunk)};

  #pragma omp parallel for schedule(static)
  for (auto index = 0uz; index < chunks; ++index) {
    const auto offset{index * chunk};

    fcn(offset, xpu::min(chunk, count - offset));
  }
}

template <typename T>
inline auto fill_chunks(
  T* XPU_RESTRICT ptr,
  std::size_t count,
  T value,
  bool parallel
) -> void {
  const auto chunk{xpu::max(parallel_chunk_bytes / sizeof(T), 1uz)};

  for_each_chunk(count, chunk, parallel, [&](std::size_t offset, std::size_t length) {
    std::fill_n(ptr + offset, length, value);
  });
}

} // namespace xpu::detail
#endif

template <typename T>
inline auto fill_n(
  T* XPU_RESTRICT ptr,
  std::size_t count,
  T value
) -> void {
#if defined(XPU_CUDA)
  xpu::cu_check(cub::DeviceTransform::Fill(ptr, count, value));
#else
  detail::fill_chunks(ptr, count, value, detail::parallel_bytes(count * sizeof(T)));
#endif
}

} // namespace xpu

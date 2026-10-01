#pragma once

#include <xpu/config.hpp>
#include <xpu/detail/checked.hpp>
#include <xpu/launch.hpp>

#if defined(XPU_CUDA)
  #include <cub/device/device_transform.cuh>
  #include <cuda/std/algorithm>
  #include <cuda/std/cstddef>
#elif defined(XPU_HIP)
  // hipCUB's C++23 path names std::extents without including <mdspan>
  #include <mdspan>
  #include <hipcub/device/device_for.hpp>
  #include <cstddef>
#else
  #include <algorithm>
  #include <cstddef>
  #include <cstring>
#endif


namespace xpu {

#if defined(XPU_HIP)
namespace detail {

template <typename T>
struct fill_function {
  T* ptr;
  T value;

  XPU_DEVICE_ONLY
  auto operator()(std::size_t linear) const -> void {
    ptr[linear] = value;
  }
};

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
#elif defined(XPU_HIP)
  if (count == 0uz) { return; }

  // hipCUB has no DeviceTransform::Fill
  xpu::cu_check(hipcub::DeviceFor::Bulk(count, detail::fill_function<T>{ptr, value}));
#else
  std::fill_n(ptr, count, value);
#endif
}

} // namespace xpu

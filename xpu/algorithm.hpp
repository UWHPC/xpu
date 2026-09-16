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

/** @brief Fill the first @p count elements of @p ptr with @p value. */
template <typename T>
inline auto fill_n(
  T* RESTRICT ptr, 
  std::size_t count,
  T value
) -> void {
#if defined(XPU_CUDA)
  xpu::cu_check(cub::DeviceTransform::Fill(ptr, count, value));
#else
  std::fill_n(ptr, count, value);
#endif
}

} // namespace xpu

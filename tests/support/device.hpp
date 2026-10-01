#pragma once

#include <xpu/config.hpp>

#include <source_location>

namespace test {

inline auto check_launch(
  std::source_location loc = std::source_location::current()
) -> void {
#if defined(XPU_CUDA)
  xpu::cu_check(cudaGetLastError(), cudaSuccess, loc);
#else
  xpu::cu_check(hipGetLastError(), hipSuccess, loc);
#endif
}

inline auto synchronize(
  std::source_location loc = std::source_location::current()
) -> void {
#if defined(XPU_CUDA)
  xpu::cu_check(cudaDeviceSynchronize(), cudaSuccess, loc);
#else
  xpu::cu_check(hipDeviceSynchronize(), hipSuccess, loc);
#endif
}

} // namespace test

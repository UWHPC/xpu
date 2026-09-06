#pragma once

#include <xpu/config.hpp>

#if defined (XPU_CUDA)
  #include <cuda/std/limits>
  #include <cuda/std/numbers>
#else
  #include <limits>
  #include <numbers>
#endif

namespace xpu {

namespace numbers = xstd::numbers;

using xstd::numeric_limits;

} // namespace xpu
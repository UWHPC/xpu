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

/** @brief Standard mathematical constants from the active backend. */
namespace numbers = xstd::numbers;

/** @brief std::numeric_limits or cuda::std::numeric_limits. */
using xstd::numeric_limits;

} // namespace xpu

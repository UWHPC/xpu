#pragma once

#include <xpu/config.hpp>

#if defined(XPU_CUDA)
  #include <cuda/std/array>
#else
  #include <array>
#endif

namespace xpu {

/** @brief std::array on CPU and cuda::std::array on CUDA. */
using xstd::array;

}

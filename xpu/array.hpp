#pragma once

#include <xpu/config.hpp>

#if defined(XPU_CUDA)
  #include <cuda/std/array>
#else
  #include <array>
#endif

namespace xpu {

using xstd::array;

}
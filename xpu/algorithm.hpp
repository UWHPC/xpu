#pragma once

#include <xpu/config.hpp>
#include <xpu/launch.hpp>

#if defined(XPU_CUDA)
  #include <cuda/std/algorithm>
  #include <cuda/std/cstddef>
#else
  #include <algorithm>
  #include <cstddef>
#endif


namespace xpu {

namespace kernel {

#if defined(XPU_CUDA)

template <typename T, typename V> __global__
void cudaBackendFill(
  T* RESTRICT ptr,
  std::size_t size,
  V value
) {
  const auto [i]{xpu::global_index<1>()};
  if (i >= size) { return; }

  ptr[i] = value;
}

#endif

} // namespace xpu::kernel

template <typename T, typename V>
inline void fill_n(
  T* RESTRICT ptr, 
  std::size_t size,
  V value
) {
#if defined(XPU_CUDA)
  dim3 backendFillThreads(256);
  dim3 backendFillBlocks(
    xpu::num_blocks(size, backendFillThreads.x)
  );

  kernel::cudaBackendFill<<<
    backendFillBlocks, backendFillThreads
  >>>(
    ptr, size, value
  );
  cuda_check(cudaGetLastError());
#else
  std::fill_n(ptr, size, value);
#endif
}

using xstd::min; using xstd::max;

}

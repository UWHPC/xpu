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

using xstd::min; using xstd::max;

namespace detail {

#if defined(XPU_CUDA)

template <typename T> __global__
void cudaBackendFillN(
  T* RESTRICT ptr,
  std::size_t size,
  T value
) {
  const auto [first]{xpu::global_index<1>()};
  const auto [stride]{xpu::global_stride<1>()};

  for (auto i{first}; i < size; i += stride) {
    ptr[i] = value;
  }
}

#endif

} // namespace xpu::detail

template <typename T>
inline void fill_n(
  T* RESTRICT ptr, 
  std::size_t size,
  T value
) {
#if defined(XPU_CUDA)
  if (size == 0) { return; }

  const dim3 threads{256};
  const dim3 blocks{
    xpu::min(
      xpu::num_blocks(size, threads.x),
      xpu::wave_blocks(detail::cudaBackendFillN<T>, threads.x)
    )
  };

  detail::cudaBackendFillN<T><<<
    blocks, threads
  >>>(
    ptr, size, value
  );
  cuda_check(cudaGetLastError());
#else
  std::fill_n(ptr, size, value);
#endif
}

} // namespace xpu

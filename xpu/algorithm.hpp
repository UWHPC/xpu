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

namespace detail {

#if defined(XPU_CUDA)

template <typename T> __global__
void cudaBackendFillN(
  T* RESTRICT ptr,
  std::size_t size,
  T value
) {
  const auto [i]{xpu::global_index<1>()};
  if (i >= size) { return; }

  ptr[i] = value;
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
  if (size == 0uz) { return; }

  const dim3 threads{256u};
  const dim3 blocks{
    xpu::block_per_dim(size, threads.x)
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

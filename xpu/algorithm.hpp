#pragma once

#include <xpu/config.hpp>
#include <xpu/launch.hpp>

#if defined(XPU_CUDA)
  #include <cuda/std/algorithm>
  #include <cuda/std/cstddef>
#else
  #include <algorithm>
  #include <cstddef>
  #include <cstring>
#endif


namespace xpu {

namespace detail {

#if defined(XPU_CUDA)

template <typename T> __global__
void cudaBackendFillN(
  T* RESTRICT ptr,
  std::size_t count,
  T value
) {
  const auto [i]{xpu::global_index<1>()};
  if (i >= count) { return; }

  ptr[i] = value;
}

#endif

} // namespace xpu::detail

template <typename T>
inline void fill_n(
  T* RESTRICT ptr, 
  std::size_t count,
  T value
) {
#if defined(XPU_CUDA)
  if (count == 0uz) { return; }

  const dim3 threads{256u};
  const dim3 blocks{
    xpu::block_per_dim(count, threads.x)
  };

  detail::cudaBackendFillN<T><<<
    blocks, threads
  >>>(
    ptr, count, value
  );
  xpu::cu_check(cudaGetLastError());
#else
  std::fill_n(ptr, count, value);
#endif
}

} // namespace xpu

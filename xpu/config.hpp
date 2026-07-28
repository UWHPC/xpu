#pragma once

#if defined(XPU_CUDA) && !defined(__CUDACC__)
  #error "ERROR: must use nvcc when compiling with the XPU_CUDA flag."
#endif

#if defined(XPU_CUDA)
  #include <cuda_runtime.h>
  #include <cuda/std/cstddef>
#endif

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <source_location>

#if defined(XPU_CUDA)
  namespace xstd = cuda::std;
#else
  namespace xstd = std;
#endif

namespace xpu {

#if defined(XPU_CUDA)

inline void cuda_check(
  cudaError_t result,
  std::source_location loc = std::source_location::current()
) {
  if (result == cudaSuccess) { return; }

  std::fprintf(
    stderr,
    "CUDA error at %s:%u in %s\n  %s\n",
    loc.file_name(),
    loc.line(),
      loc.function_name(),
    cudaGetErrorString(result)
  );

  std::abort();
}

#endif

#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
  #define RESTRICT __restrict
#else
  #define RESTRICT
#endif

#if defined(XPU_CUDA)
  #define CUDA_CALLABLE __host__ __device__
#else
  #define CUDA_CALLABLE
#endif

#ifndef XPU_SIMD_BYTES
  #if defined(__AVX512F__)
    #define XPU_SIMD_BYTES 64
  #elif defined(__AVX2__) || defined(__AVX__)
    #define XPU_SIMD_BYTES 32
  #else
    #define XPU_SIMD_BYTES 16
  #endif
#endif

inline constexpr std::size_t simd_bytes{XPU_SIMD_BYTES};
inline constexpr bool xpu_cuda{
#if defined(XPU_CUDA)
  true
#else
  false
#endif
};

} // namespace xpu

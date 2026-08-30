#pragma once

#include <type_traits>
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
#include <concepts>
#include <source_location>

#if defined(XPU_CUDA)
  namespace xstd = cuda::std;
#else
  namespace xstd = std;
#endif

namespace xpu {

#if defined(XPU_CUDA)

template <typename Status>
inline void cu_check(
  Status result,
  Status success = Status{},
  std::source_location loc = std::source_location::current()
) {
  if (result == success) { return; }

  std::fprintf(
    stderr,
    "CUDA error at %s:%u in %s\n status code: %d\n",
    loc.file_name(),
    loc.line(),
    loc.function_name(),
    static_cast<int>(result)
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

#if defined(XPU_CUDA)
  #define DEVICE_ONLY __device__
#else
  #define DEVICE_ONLY
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

template <typename T>
concept supported_float =
  std::same_as<T, float> ||
  std::same_as<T, double>;

template <typename T>
concept arithmetic = 
  (std::integral<T>        ||
   std::floating_point<T>) &&
  !std::same_as<T, bool>   &&
  !std::same_as<T, char>;

inline constexpr std::size_t simd_bytes{XPU_SIMD_BYTES};
inline constexpr bool xpu_cuda{
#if defined(XPU_CUDA)
  true
#else
  false
#endif
};

} // namespace xpu

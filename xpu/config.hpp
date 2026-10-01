#pragma once

#if defined(XPU_CUDA) && defined(XPU_HIP)
  #error "ERROR: XPU_CUDA and XPU_HIP cannot both be defined."
#endif

#if defined(XPU_CUDA) && !defined(__CUDACC__)
  #error "ERROR: must use nvcc when compiling with the XPU_CUDA flag."
#endif

#if defined(XPU_HIP) && !defined(__HIP__)
  #error "ERROR: must compile as HIP when compiling with the XPU_HIP flag."
#endif

#if defined(XPU_CUDA) || defined(XPU_HIP)
  #define XPU_GPU 1
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  #define XPU_DEVICE_COMPILE 1
#endif

#if defined(XPU_CUDA)
  #include <cuda_runtime.h>
  #include <cuda/std/cstddef>
#elif defined(XPU_HIP)
  #include <hip/hip_runtime.h>
#endif

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <concepts>
#include <source_location>
#include <type_traits>

#if defined(XPU_CUDA)
  namespace xstd = cuda::std;
#else
  namespace xstd = std;
#endif

namespace xpu {

#if defined(XPU_GPU)

template <typename Status>
inline auto cu_check(
  Status result,
  Status success = Status{},
  std::source_location loc = std::source_location::current()
) noexcept -> void {
  if (result == success) { return; }

#if defined(XPU_CUDA)
  constexpr auto backend{"CUDA"};
#else
  constexpr auto backend{"HIP"};
#endif

  std::fprintf(
    stderr,
    "%s error at %s:%u in %s\n status code: %d\n",
    backend,
    loc.file_name(),
    loc.line(),
    loc.function_name(),
    static_cast<int>(result)
  );

  std::abort();
}

#endif

#define XPU_FORCE_INLINE inline __attribute__((always_inline))

#if defined(__GNUC__) || defined(__clang__)
  #define XPU_RESTRICT __restrict
#else
  #define XPU_RESTRICT
#endif

#if defined(XPU_GPU)
  #define XPU_CUDA_CALLABLE __host__ __device__
#else
  #define XPU_CUDA_CALLABLE
#endif

#if defined(XPU_GPU)
  #define XPU_DEVICE_ONLY __device__
#else
  #define XPU_DEVICE_ONLY
#endif

#ifndef XPU_SIMD_BYTES
  #if defined(__AVX512F__)
    #define XPU_SIMD_BYTES 64uz
  #elif defined(__AVX2__) || defined(__AVX__)
    #define XPU_SIMD_BYTES 32uz
  #else
    #define XPU_SIMD_BYTES 16uz
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

inline constexpr auto simd_bytes{std::size_t{XPU_SIMD_BYTES}};
inline constexpr auto cuda_align_bytes{128uz};

inline constexpr auto xpu_cuda{
#if defined(XPU_CUDA)
  true
#else
  false
#endif
};

inline constexpr auto xpu_hip{
#if defined(XPU_HIP)
  true
#else
  false
#endif
};

inline constexpr auto xpu_gpu{xpu_cuda || xpu_hip};

inline constexpr auto alignment_bytes{
  xpu_gpu ? cuda_align_bytes : simd_bytes
};

} // namespace xpu

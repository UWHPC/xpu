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
#include <concepts>
#include <source_location>
#include <type_traits>

#if defined(XPU_CUDA)
  namespace xstd = cuda::std;
#else
  namespace xstd = std;
#endif

/** @brief Utilities shared by the CPU and CUDA backends. */
namespace xpu {

#if defined(XPU_CUDA)

/** @brief Check a CUDA status against @p success and abort on failure,
 *  reporting the status and @p loc.
 */
template <typename Status>
inline auto cu_check(
  Status result,
  Status success = Status{},
  std::source_location loc = std::source_location::current()
) noexcept -> void {
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

/** @brief Compiler spelling for a nonaliasing pointer. */
#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
  #define RESTRICT __restrict
#else
  #define RESTRICT
#endif

/** @brief Mark a function callable on both host and device. */
#if defined(XPU_CUDA)
  #define CUDA_CALLABLE __host__ __device__
#else
  #define CUDA_CALLABLE
#endif

/** @brief Mark a function callable on the CUDA device. */
#if defined(XPU_CUDA)
  #define DEVICE_ONLY __device__
#else
  #define DEVICE_ONLY
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

/** @brief Floating-point types supported by backend-specific operations. */
template <typename T>
concept supported_float =
  std::same_as<T, float> ||
  std::same_as<T, double>;

/** @brief Arithmetic types accepted by reductions and atomic operations. */
template <typename T>
concept arithmetic = 
  (std::integral<T>        ||
   std::floating_point<T>) &&
  !std::same_as<T, bool>   &&
  !std::same_as<T, char>;

/** @brief CPU SIMD alignment in bytes. */
inline constexpr auto simd_bytes{std::size_t{XPU_SIMD_BYTES}};
/** @brief Alignment used for CUDA allocations. */
inline constexpr auto cuda_align_bytes{128uz};

/** @brief Whether this translation unit uses the CUDA backend. */
inline constexpr auto xpu_cuda{
#if defined(XPU_CUDA)
  true
#else
  false
#endif
};

/** @brief Alignment selected for the active backend. */
inline constexpr auto alignment_bytes{
  xpu_cuda ? cuda_align_bytes : simd_bytes
};

} // namespace xpu

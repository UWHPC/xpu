#pragma once

#include <xpu/config.hpp>

#include <atomic>

#if defined(XPU_CUDA)
  #include <cuda/std/algorithm>
  #include <cuda/std/cmath>
  #include <cuda/std/cstdlib>
  #include <cuda/std/complex>
  #include <cuda/std/concepts>
  #include <cuda/std/type_traits>
#else
  #include <algorithm>
  #include <cmath>
  #include <cstdlib>
  #include <complex>
  #include <concepts>
  #include <type_traits>
#endif

namespace xpu {

// exp / log
using xstd::exp;   using xstd::expm1;
using xstd::log;   using xstd::log2;
using xstd::log10; using xstd::log1p;

// powers and roots
using xstd::sqrt;  using xstd::cbrt;
using xstd::pow;   using xstd::hypot;

// trig + inverse
using xstd::sin;   using xstd::cos;   using xstd::tan;
using xstd::asin;  using xstd::acos;  using xstd::atan;
using xstd::atan2;

// hyperbolic
using xstd::sinh;  using xstd::cosh;  using xstd::tanh;
using xstd::asinh; using xstd::acosh; using xstd::atanh;

// special
using xstd::erf;   using xstd::erfc;
using xstd::lgamma; using xstd::tgamma;

// magnitude / selection
using xstd::min; using xstd::max;
using xstd::abs;   using xstd::fabs;
using xstd::fmin;  using xstd::fmax;

// arithmetic
using xstd::fmod;  using xstd::remainder; using xstd::fma;

// rounding
using xstd::floor; using xstd::ceil;  using xstd::round;
using xstd::trunc; using xstd::nearbyint; using xstd::rint;

// classification
using xstd::copysign; using xstd::signbit;
using xstd::isnan;    using xstd::isinf;
using xstd::isfinite; using xstd::isnormal;

// decomposition
using xstd::ldexp; using xstd::frexp; using xstd::modf;

// Overflow safe version of (num + den - 1) / den
template <std::unsigned_integral T> [[nodiscard]] CUDA_CALLABLE
inline constexpr T ceiling_div(T num, T den) noexcept {
  return num / den + (num % den != 0);
}

template <std::floating_point T> CUDA_CALLABLE
inline void sincos(T arg, T* RESTRICT s, T* RESTRICT c) noexcept  {
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same_v<T, float>) { ::sincosf(arg, s, c); }
  else { ::sincos(arg, s, c); }
#else
  *s = xpu::sin(arg); *c = xpu::cos(arg);
#endif
}

template <std::floating_point T> CUDA_CALLABLE
inline T rsqrt(T x) noexcept {
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same_v<T, float>) { return ::rsqrtf(x); } else { return ::rsqrt(x); }
#else
  return T{1} / xpu::sqrt(x);
#endif
}

template <std::floating_point T> CUDA_CALLABLE
inline T norm3d(T x, T y, T z) noexcept {
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same_v<T, float>) { return ::norm3df(x, y, z); }
  else { return ::norm3d(x, y, z); }
#else
  return xpu::sqrt(x*x + y*y + z*z);
#endif
}

template <std::floating_point T> [[nodiscard]] CUDA_CALLABLE
inline T rnorm3d(T x, T y, T z) noexcept {
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same_v<T, float>) { return ::rnorm3df(x, y, z); }
  else { return ::rnorm3d(x, y, z); }
#else
  return T{1} / xpu::norm3d(x, y, z);
#endif
}

template <arithmetic T> CUDA_CALLABLE
inline void atomic_add(T* ptr, T value) noexcept {
#if defined(__CUDA_ARCH__)
  atomicAdd(ptr, value);
#else
  std::atomic_ref<T>(*ptr).fetch_add(value, std::memory_order_relaxed);
#endif
}

} // namespace xpu

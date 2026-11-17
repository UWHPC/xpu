#pragma once

#include <xpu/config.hpp>
#include <xpu/detail/checked.hpp>

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

#if defined(XPU_CUDA)
using xstd::fmin;  using xstd::fmax;
#else
// libm's fmin/fmax are calls that stop a loop from vectorizing unless the
// consumer builds with -ffinite-math-only; these keep the NaN rule (a NaN
// argument yields the other one) with two compares.
template <typename X, typename Y>
  requires std::is_arithmetic_v<X> && std::is_arithmetic_v<Y>
[[nodiscard]]
inline constexpr auto fmin(X x, Y y) noexcept -> decltype(std::fmin(x, y)) {
  using result_t = decltype(std::fmin(x, y));
  const auto a{static_cast<result_t>(x)};
  const auto b{static_cast<result_t>(y)};

  return (b != b || a < b) ? a : b;
}

template <typename X, typename Y>
  requires std::is_arithmetic_v<X> && std::is_arithmetic_v<Y>
[[nodiscard]]
inline constexpr auto fmax(X x, Y y) noexcept -> decltype(std::fmax(x, y)) {
  using result_t = decltype(std::fmax(x, y));
  const auto a{static_cast<result_t>(x)};
  const auto b{static_cast<result_t>(y)};

  return (b != b || a > b) ? a : b;
}
#endif

// arithmetic
using xstd::fmod;  using xstd::remainder; using xstd::fma;

// rounding
using xstd::floor; using xstd::ceil;
using xstd::trunc; using xstd::nearbyint; using xstd::rint;

#if defined(XPU_CUDA)
using xstd::round;
#else
// half-away-from-zero rounding has no SSE/AVX instruction, so libm's round is
// a call; trunc is one, and the half case is a compare.
template <typename X>
  requires std::is_arithmetic_v<X>
[[nodiscard]]
inline constexpr auto round(X x) noexcept -> decltype(std::round(x)) {
  using result_t = decltype(std::round(x));
  const auto value{static_cast<result_t>(x)};
  const auto whole{std::trunc(value)};
  const auto half{std::fabs(value - whole) >= result_t{0.5}};

  return whole + std::copysign(half ? result_t{1} : result_t{0}, value);
}
#endif

// classification
using xstd::copysign; using xstd::signbit;
using xstd::isnan;    using xstd::isinf;
using xstd::isfinite; using xstd::isnormal;

// decomposition
using xstd::ldexp; using xstd::frexp; using xstd::modf;

// Overflow safe version of (num + den - 1) / den
template <std::unsigned_integral T> [[nodiscard]] XPU_CUDA_CALLABLE
inline constexpr auto ceiling_div(T num, T den) noexcept -> T {
  const auto zero_denominator{den == 0};

  if (zero_denominator) {
    xpu::detail::checked_error("zero ceiling division denominator");
  }

  const auto quotient{num / den + (num % den != 0)};

  return quotient;
}

template <std::floating_point T> XPU_CUDA_CALLABLE
inline auto sincos(T arg, T* XPU_RESTRICT s, T* XPU_RESTRICT c) noexcept -> void {
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same_v<T, float>) { ::sincosf(arg, s, c); }
  else { ::sincos(arg, s, c); }
#else
  *s = xpu::sin(arg); *c = xpu::cos(arg);
#endif
}

template <std::floating_point T> [[nodiscard]] XPU_CUDA_CALLABLE
inline auto rsqrt(T x) noexcept -> T {
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same_v<T, float>) { return ::rsqrtf(x); } else { return ::rsqrt(x); }
#else
  return T{1} / xpu::sqrt(x);
#endif
}

template <std::floating_point T> [[nodiscard]] XPU_CUDA_CALLABLE
inline auto norm3d(T x, T y, T z) noexcept -> T {
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same_v<T, float>) { return ::norm3df(x, y, z); }
  else { return ::norm3d(x, y, z); }
#else
  return xpu::sqrt(x*x + y*y + z*z);
#endif
}

template <std::floating_point T> [[nodiscard]] XPU_CUDA_CALLABLE
inline auto rnorm3d(T x, T y, T z) noexcept -> T {
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same_v<T, float>) { return ::rnorm3df(x, y, z); }
  else { return ::rnorm3d(x, y, z); }
#else
  return T{1} / xpu::norm3d(x, y, z);
#endif
}

template <arithmetic T> XPU_CUDA_CALLABLE
inline auto atomic_add(T* ptr, T value) noexcept -> void {
#if defined(__CUDA_ARCH__)
  atomicAdd(ptr, value);
#else
  std::atomic_ref<T>(*ptr).fetch_add(value, std::memory_order_relaxed);
#endif
}

} // namespace xpu

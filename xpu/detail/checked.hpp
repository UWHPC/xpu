#pragma once

#include <xpu/config.hpp>

#include <concepts>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#if defined(XPU_CUDA)
  #include <cuda/std/limits>
#else
  #include <limits>
#endif
#include <type_traits>

namespace xpu::detail {

/** @brief Integral types other than bool. */
template <typename T>
concept integer =
  std::integral<T> &&
  !std::same_as<T, bool>;

/** @brief Unsigned integral types other than bool. */
template <typename T>
concept unsigned_integer =
  integer<T> &&
  std::unsigned_integral<T>;

/** @brief Types whose object representations can be copied as bytes. */
template <typename T>
concept trivially_copyable =
  std::is_trivially_copyable_v<T>;

/** @brief Print @p message and terminate on failed checked arithmetic.
 *  @details Traps in device code and aborts on the host.
 */
CUDA_CALLABLE
inline auto checked_error(const char* message) noexcept -> void {
#if defined(__CUDA_ARCH__)
  printf("xpu: %s\n", message);
  __trap();
#else
  std::fprintf(stderr, "xpu: %s\n", message);
  std::abort();
#endif
}

/** @brief Multiply unsigned integers, aborting on overflow. */
template <unsigned_integer T> [[nodiscard]] CUDA_CALLABLE
constexpr auto checked_mul(T a, T b) noexcept -> T {
  const auto overflow{a != 0 && b > xstd::numeric_limits<T>::max() / a};

  if (overflow) {
    checked_error("integer multiplication overflow");
  }

  const auto product{a * b};

  return product;
}

/** @brief Add unsigned integers, aborting on overflow. */
template <unsigned_integer T> [[nodiscard]] CUDA_CALLABLE
constexpr auto checked_add(T a, T b) noexcept -> T {
  const auto overflow{b > xstd::numeric_limits<T>::max() - a};

  if (overflow) {
    checked_error("integer addition overflow");
  }

  const auto sum{a + b};

  return sum;
}

/** @brief Round up to a nonzero multiple, aborting on overflow. */
template <unsigned_integer T> [[nodiscard]] CUDA_CALLABLE
constexpr auto checked_round_up(T count, T multiple) noexcept -> T {
  const auto zero_multiple{multiple == 0};

  if (zero_multiple) {
    checked_error("zero rounding multiple");
  }

  const auto remainder{count % multiple};
  const auto already_rounded{remainder == 0};

  const auto padding{static_cast<T>(already_rounded ? 0 : multiple - remainder)};
  const auto rounded_count{checked_add(count, padding)};

  return rounded_count;
}

/** @brief Convert an integer, aborting if the value is out of range. */
template <integer To, integer From> [[nodiscard]] CUDA_CALLABLE
constexpr auto checked_cast(From value) noexcept -> To {
  constexpr auto signed_to_unsigned{std::is_signed_v<From> && !std::is_signed_v<To>};
  constexpr auto both_signed{std::is_signed_v<From> && std::is_signed_v<To>};

  constexpr auto narrower_destination{
    xstd::numeric_limits<To>::digits < xstd::numeric_limits<From>::digits
  };

  auto out_of_range{false};

  if constexpr (signed_to_unsigned) {
    out_of_range = value < 0;
  }

  if constexpr (narrower_destination) {
    constexpr auto maximum{static_cast<From>(xstd::numeric_limits<To>::max())};
    const auto above_maximum{value > maximum};

    out_of_range = out_of_range || above_maximum;

    if constexpr (both_signed) {
      constexpr auto minimum{static_cast<From>(xstd::numeric_limits<To>::lowest())};
      const auto below_minimum{value < minimum};

      out_of_range = out_of_range || below_minimum;
    }
  }

  if (out_of_range) {
    checked_error("integer conversion out of range");
  }

  const auto converted_value{static_cast<To>(value)};

  return converted_value;
}

/** @brief Compute a byte count, aborting on multiplication overflow. */
template <trivially_copyable T> [[nodiscard]] CUDA_CALLABLE
constexpr auto checked_bytes(std::size_t count) noexcept -> std::size_t {
  const auto byte_count{checked_mul(count, sizeof(T))};

  return byte_count;
}

} // namespace xpu::detail

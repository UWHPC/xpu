#pragma once

#include "../support/aborts.hpp"
#include "../support/check.hpp"

#include <xpu/detail/checked.hpp>
#include <xpu/memory.hpp>
#include <xpu/soa.hpp>

#include <cstdint>
#include <limits>

struct unexpected_iteration {
  template <typename Index> CUDA_CALLABLE
  auto operator()(const Index&) const -> void {
    xpu::detail::checked_error("invalid or empty range invoked its callback");
  }
};

inline auto run_checked_cases() -> int {
  static_assert(xpu::detail::checked_mul(std::uint16_t{255}, std::uint16_t{257}) == 65535);
  static_assert(xpu::detail::checked_mul(0u, 42u) == 0u);
  static_assert(xpu::detail::checked_mul(42u, 0u) == 0u);
  static_assert(xpu::detail::checked_mul(std::uint8_t{15}, std::uint8_t{17}) == 255);
  static_assert(xpu::detail::checked_mul(std::numeric_limits<std::uint64_t>::max(), std::uint64_t{1}) ==
    std::numeric_limits<std::uint64_t>::max());

  static_assert(xpu::detail::checked_add(std::uint8_t{254}, std::uint8_t{1}) == 255);
  static_assert(xpu::detail::checked_add(std::uint8_t{255}, std::uint8_t{0}) == 255);

  static_assert(xpu::detail::checked_round_up(0u, 3u) == 0u);
  static_assert(xpu::detail::checked_round_up(9u, 3u) == 9u);
  static_assert(xpu::detail::checked_round_up(10u, 3u) == 12u);
  static_assert(xpu::detail::checked_round_up(std::uint8_t{254}, std::uint8_t{255}) == 255);

  static_assert(xpu::detail::checked_cast<std::int8_t>(-128) == -128);
  static_assert(xpu::detail::checked_cast<std::int8_t>(127u) == 127);
  static_assert(xpu::detail::checked_cast<std::uint8_t>(255) == 255);
  static_assert(xpu::detail::checked_cast<unsigned int>(0) == 0u);
  static_assert(xpu::detail::checked_cast<int>(std::int8_t{-128}) == -128);
  static_assert(xpu::detail::checked_cast<int>(std::uint8_t{255}) == 255);
  static_assert(xpu::detail::checked_cast<unsigned int>(std::uint8_t{255}) == 255u);

  static_assert(xpu::bytes<std::uint32_t>(3uz) == 12uz);
  static_assert(xpu::handle_pad<unsigned char>(xpu::simd_bytes) == xpu::simd_bytes);

  static_assert(xpu::detail::checked_bytes<std::uint32_t>(0uz) == 0uz);
  static_assert(xpu::detail::checked_bytes<std::uint32_t>(3uz) == 12uz);

  constexpr auto byte_count{xpu::simd_bytes + 1uz};
  constexpr auto padded_byte_count{xpu::checked_padding<unsigned char>(byte_count)};

  static_assert(xpu::checked_padding<unsigned char>(0uz) == 0uz);
  static_assert(xpu::checked_padding<unsigned char>(xpu::simd_bytes) == xpu::simd_bytes);
  static_assert(padded_byte_count >= byte_count);
  static_assert(padded_byte_count - byte_count < xpu::simd_bytes);
  static_assert(xpu::checked_padding<unsigned char>(padded_byte_count) == padded_byte_count);

  const auto empty_range = xpu::range<3uz>{
    {0uz, 0uz, 0uz},
    {std::numeric_limits<std::size_t>::max(), 2uz, 0uz},
    {1uz, 1uz, 1uz}
  };

  xpu::parallel_for(empty_range, unexpected_iteration{});

#if defined(__unix__)
  if (const auto failure{test::check_abort([] {
    static_cast<void>(xpu::ceiling_div(1uz, 0uz));
  })}; failure) {

    return failure;
  }

  if (const auto failure{test::check_abort([] {
    static_cast<void>(xpu::bytes<std::uint64_t>(std::numeric_limits<std::size_t>::max()));
  })}; failure) {

    return failure;
  }

  if (const auto failure{test::check_abort([] {
    auto view{xpu::soa_view<unsigned char, 2uz>{nullptr, std::numeric_limits<std::size_t>::max()}};
  })}; failure) {

    return failure;
  }

  if (const auto failure{test::check_abort([] {
    const auto range = xpu::range<2uz>{
      {0uz, 0uz},
      {std::numeric_limits<std::size_t>::max() / 2uz + 1uz, 2uz},
      {1uz, 1uz}
    };

    xpu::parallel_for(range, unexpected_iteration{});
  })}; failure) {

    return failure;
  }

  if (const auto failure{test::check_abort([] {
    static_cast<void>(xpu::detail::checked_mul(std::uint8_t{16}, std::uint8_t{16}));
  })}; failure) {

    return failure;
  }

  if (const auto failure{test::check_abort([] {
    static_cast<void>(xpu::detail::checked_add(std::uint8_t{255}, std::uint8_t{1}));
  })}; failure) {

    return failure;
  }

  if (const auto failure{test::check_abort([] {
    static_cast<void>(xpu::detail::checked_round_up(0u, 0u));
  })}; failure) {

    return failure;
  }

  if (const auto failure{test::check_abort([] {
    static_cast<void>(xpu::detail::checked_round_up(std::uint8_t{255}, std::uint8_t{2}));
  })}; failure) {

    return failure;
  }

  if (const auto failure{test::check_abort([] {
    static_cast<void>(xpu::detail::checked_cast<unsigned int>(-1));
  })}; failure) {

    return failure;
  }

  if (const auto failure{test::check_abort([] {
    static_cast<void>(xpu::detail::checked_cast<std::int8_t>(-129));
  })}; failure) {

    return failure;
  }

  if (const auto failure{test::check_abort([] {
    static_cast<void>(xpu::detail::checked_cast<std::int8_t>(128));
  })}; failure) {

    return failure;
  }

  if (const auto failure{test::check_abort([] {
    static_cast<void>(xpu::detail::checked_cast<std::int8_t>(128u));
  })}; failure) {

    return failure;
  }

  if (const auto failure{test::check_abort([] {
    static_cast<void>(xpu::detail::checked_cast<std::uint8_t>(256u));
  })}; failure) {

    return failure;
  }

  if (const auto failure{test::check_abort([] {
    const auto excessive_count{std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t) + 1uz};
    auto* allocation{xpu::alloc<std::uint64_t>(excessive_count)};
    xpu::free<std::uint64_t>(allocation);
  })}; failure) {

    return failure;
  }

  if (const auto failure{test::check_abort([] {
    auto values{xpu::buffer<std::uint64_t>{std::numeric_limits<std::size_t>::max()}};
  })}; failure) {

    return failure;
  }

  if (const auto failure{test::check_abort([] {
    const auto excessive_count{std::numeric_limits<std::size_t>::max() / 2uz + 1uz};
    auto values{xpu::soa<unsigned char, 2uz>{excessive_count}};
  })}; failure) {

    return failure;
  }

  if (const auto failure{test::check_abort([] {
    const auto excessive_count{std::numeric_limits<std::size_t>::max() / 2uz + 1uz};
    auto values{xpu::soa_batch<unsigned char, 1uz>{2uz, excessive_count}};
  })}; failure) {

    return failure;
  }

  if (const auto failure{test::check_abort([] {
    const auto excessive_count{std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t) + 1uz};
    auto source{std::uint64_t{}};
    auto destination{std::uint64_t{}};

    xpu::copy_n(&destination, &source, excessive_count);
  })}; failure) {

    return failure;
  }

  if (const auto failure{test::check_abort([] {
    const auto excessive_count{std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t) + 1uz};
    auto destination{std::uint64_t{}};

    xpu::zero_n(&destination, excessive_count);
  })}; failure) {

    return failure;
  }
#endif

  const auto success{0};

  return success;
}

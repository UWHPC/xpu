#pragma once

#include "../support/check.hpp"

#include <xpu/algorithm.hpp>
#include <xpu/memory.hpp>
#include <xpu/soa.hpp>

#include <memory>
#include <utility>

inline int run_soa_cases() {
  constexpr auto arrays{4uz};
  xpu::soa<int, arrays> values{test::count};

  if (values.count() != test::count) {
    return test::fail("SoA count is incorrect");
  }
  if (values.stride() != xpu::handle_pad<int>(test::count)) {
    return test::fail("SoA stride is incorrect");
  }
  if (values.storage_size() != arrays * values.stride()) {
    return test::fail("SoA storage size is incorrect");
  }

  for (auto array{0uz}; array < arrays; ++array) {
    xpu::fill_n(
      values[array], values.count(), static_cast<int>(array + 1uz)
    );
  }

  auto middle{values.view<2uz, 1uz>()};
  if (
    middle.count() != values.count() ||
    middle.stride() != values.stride() ||
    middle[1] != values[2]
  ) {
    return test::fail("mutable SoA view is incorrect");
  }

  const auto& const_values{std::as_const(values)};
  const auto const_middle{const_values.view<2uz, 1uz>()};
  if (const_middle[0] != const_values[1]) {
    return test::fail("const SoA view is incorrect");
  }

  auto result{std::make_unique<int[]>(values.count())};
  for (auto array{0uz}; array < arrays; ++array) {
    xpu::copy_n(result.get(), values[array], values.count());
    for (auto i{0uz}; i < values.count(); ++i) {
      if (result[i] != static_cast<int>(array + 1uz)) {
        return test::fail("SoA array contents are incorrect");
      }
    }
  }

  return 0;
}

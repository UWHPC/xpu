#pragma once

#include "../support/check.hpp"

#include <xpu/xpu.hpp>

#include <memory>

inline int run_integration_cases() {
  xpu::buffer<unsigned int> values{test::count};
  xpu::fill_n(values.data(), values.count(), 9u);

  auto result{std::make_unique<unsigned int[]>(test::count)};
  xpu::copy_n(result.get(), values.data(), test::count);
  for (auto i{0uz}; i < test::count; ++i) {
    if (result[i] != 9u) {
      return test::fail("xpu umbrella header test failed");
    }
  }

  return 0;
}

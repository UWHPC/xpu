#pragma once

#include "../support/check.hpp"

#include <xpu/algorithm.hpp>
#include <xpu/buffer.hpp>
#include <xpu/memory.hpp>

#include <memory>

inline int run_algorithm_cases() {
  xpu::buffer<int> values{test::count};
  xpu::fill_n(values.data(), values.count(), 42);

  auto result{std::make_unique<int[]>(test::count)};
  xpu::copy_n(result.get(), values.data(), test::count);

  for (auto i{0uz}; i < test::count; ++i) {
    if (result[i] != 42) {
      return test::fail("xpu::fill_n failed");
    }
  }

  return 0;
}

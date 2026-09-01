#pragma once

#include "../support/check.hpp"

#include <xpu/memory.hpp>

#include <memory>

inline int run_memory_cases() {
  static_assert(xpu::bytes<int>(3uz) == 3uz * sizeof(int));

  xpu::unique_ptr<int> values{test::count, 7};
  auto host{std::make_unique<int[]>(test::count)};
  xpu::copy_n(host.get(), values.get(), test::count);

  for (auto i{0uz}; i < test::count; ++i) {
    if (host[i] != 7) {
      return test::fail("xpu::unique_ptr initialization failed");
    }
  }

  xpu::zero_n(values.get(), test::count);
  xpu::copy_n(host.get(), values.get(), test::count);
  for (auto i{0uz}; i < test::count; ++i) {
    if (host[i] != 0) {
      return test::fail("xpu::zero_n failed");
    }
    host[i] = static_cast<int>(i);
  }

  xpu::copy_n(values.get(), host.get(), test::count);
  auto copied{std::make_unique<int[]>(test::count)};
  xpu::copy_n(copied.get(), values.get(), test::count);
  for (auto i{0uz}; i < test::count; ++i) {
    if (copied[i] != static_cast<int>(i)) {
      return test::fail("xpu::copy_n failed");
    }
  }

  const auto padded{xpu::handle_pad<int>(test::count)};
  if (padded < test::count) {
    return test::fail("xpu::handle_pad reduced the requested size");
  }
  if constexpr (xpu::is_padded<int>) {
    if (padded % (xpu::simd_bytes / sizeof(int)) != 0uz) {
      return test::fail("xpu::handle_pad produced a partial SIMD lane");
    }
  }

  return 0;
}

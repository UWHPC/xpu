#pragma once

#include "../support/check.hpp"

#include <xpu/buffer.hpp>
#include <xpu/memory.hpp>

#include <cstdint>
#include <memory>

inline int run_buffer_cases() {
  xpu::buffer<float> values{test::count};

  if (values.count() != test::count) {
    return test::fail("buffer count is incorrect");
  }
  if (values.capacity() != xpu::handle_pad<float>(test::count)) {
    return test::fail("buffer capacity is incorrect");
  }
  if (values.data() == nullptr) {
    return test::fail("non-empty buffer has a null data pointer");
  }

  if constexpr (!xpu::xpu_cuda) {
    const auto address{reinterpret_cast<std::uintptr_t>(values.data())};
    if (address % xpu::default_align<float> != 0uz) {
      return test::fail("buffer data is not correctly aligned");
    }
  }

  auto result{std::make_unique<float[]>(values.capacity())};
  xpu::copy_n(result.get(), values.data(), values.capacity());
  for (auto i{0uz}; i < values.capacity(); ++i) {
    if (result[i] != 0.0f) {
      return test::fail("buffer storage is not value-initialized");
    }
  }

  xpu::buffer<int> empty{0uz};
  if (empty.count() != 0uz || empty.capacity() != 0uz || empty.data() != nullptr) {
    return test::fail("empty buffer state is incorrect");
  }

  return 0;
}

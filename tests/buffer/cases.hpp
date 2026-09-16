#pragma once

#include "../support/check.hpp"

#include <xpu/buffer.hpp>
#include <xpu/memory.hpp>

#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

inline constexpr auto buffer_view_constant_case() -> bool {
  int values[]{1, 2, 3};
  auto view{xpu::buffer_view<int>{values, 3uz}};
  view[1uz] = 7;
  const auto readonly{xpu::buffer_view<const int>{values, 3uz}};
  return view.count() == 3uz && values[1] == 7 && readonly[1uz] == 7;
}

inline auto run_buffer_view_cases() -> int {
  static_assert(buffer_view_constant_case());
  static_assert(std::is_trivially_copyable_v<xpu::buffer_view<int>>);

  int values[]{1, 2, 3, 4};
  auto view{xpu::buffer_view<int>{values + 1, 2uz}};
  auto copy{view};
  auto readonly{xpu::buffer_view<const int>{values + 1, 2uz}};

  static_assert(std::is_same_v<decltype(view[0uz]), int&>);
  static_assert(std::is_same_v<decltype(std::as_const(view)[0uz]), const int&>);
  static_assert(std::is_same_v<decltype(readonly[0uz]), const int&>);
  static_assert(std::is_same_v<decltype(std::move(view)[0uz]), int&>);

  copy[1uz] = 9;
  if (view.count() != 2uz || &view[0uz] != values + 1 ||
      values[2] != 9 || readonly[1uz] != 9 || values[0] != 1 || values[3] != 4) {
    return test::fail("buffer view does not refer to the supplied storage");
  }

  const xpu::buffer_view<int> empty{nullptr, 0uz};
  if (empty.count() != 0uz) {
    return test::fail("empty buffer view count is incorrect");
  }

  return 0;
}

inline int run_buffer_cases() {
  if (const auto failure{run_buffer_view_cases()}; failure) {
    return failure;
  }

  auto values{xpu::buffer<float>{test::count}};
  auto view{values.view()};
  auto readonly{std::as_const(values).view()};
  static_assert(std::is_same_v<decltype(view), xpu::buffer_view<float>>);
  static_assert(std::is_same_v<decltype(readonly), xpu::buffer_view<const float>>);

  if (view.count() != values.count() || readonly.count() != values.count()) {
    return test::fail("buffer owner view count is incorrect");
  }

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
    view[0uz] = 7.0f;
    if (&view[0uz] != values.data() || readonly[0uz] != 7.0f) {
      return test::fail("buffer owner view does not refer to its storage");
    }
    view[0uz] = 0.0f;
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

  auto empty{xpu::buffer<int>{0uz}};
  if (empty.count() != 0uz || empty.capacity() != 0uz || empty.data() != nullptr) {
    return test::fail("empty buffer state is incorrect");
  }

  return 0;
}

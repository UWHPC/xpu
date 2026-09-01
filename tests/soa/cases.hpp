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

  constexpr auto batch_count{3uz};
  constexpr auto batch_elements{19uz};
  xpu::soa_batch<int, arrays> batches{batch_count, batch_elements};

  if (
    batches.batch_count() != batch_count ||
    batches.element_count() != batch_elements ||
    batches.logical_count() != batch_count * arrays * batch_elements
  ) {
    return test::fail("batched SoA dimensions are incorrect");
  }
  if (batches.array_stride() != xpu::handle_pad<int>(batch_elements)) {
    return test::fail("batched SoA array stride is incorrect");
  }
  if (batches.batch_stride() != arrays * batches.array_stride()) {
    return test::fail("batched SoA batch stride is incorrect");
  }
  if (batches.storage_size() != batch_count * batches.batch_stride()) {
    return test::fail("batched SoA storage size is incorrect");
  }

  for (auto batch{0uz}; batch < batch_count; ++batch) {
    auto batch_view{batches.view(batch)};
    for (auto array{0uz}; array < arrays; ++array) {
      const auto value{static_cast<int>(batch * arrays + array + 1uz)};
      xpu::fill_n(batch_view[array], batch_view.count(), value);
    }
  }

  auto second_batch{batches.view(1uz)};
  auto second_batch_middle{batches.view<2uz, 1uz>(1uz)};
  if (
    second_batch_middle.count() != batch_elements ||
    second_batch_middle.stride() != batches.array_stride() ||
    second_batch_middle[0] != second_batch[1] ||
    second_batch_middle[1] != second_batch[2]
  ) {
    return test::fail("mutable batched SoA subview is incorrect");
  }

  const auto& const_batches{std::as_const(batches)};
  const auto const_second_batch_middle{const_batches.view<2uz, 1uz>(1uz)};
  if (
    const_second_batch_middle[0] != second_batch[1] ||
    const_second_batch_middle[1] != second_batch[2]
  ) {
    return test::fail("const batched SoA subview is incorrect");
  }

  auto batch_result{std::make_unique<int[]>(batch_elements)};
  for (auto batch{0uz}; batch < batch_count; ++batch) {
    const auto batch_view{const_batches.view(batch)};
    for (auto array{0uz}; array < arrays; ++array) {
      xpu::copy_n(batch_result.get(), batch_view[array], batch_elements);
      const auto expected{static_cast<int>(batch * arrays + array + 1uz)};
      for (auto i{0uz}; i < batch_elements; ++i) {
        if (batch_result[i] != expected) {
          return test::fail("batched SoA array contents are incorrect");
        }
      }
    }
  }

  return 0;
}

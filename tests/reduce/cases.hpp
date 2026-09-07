#pragma once

#include "../support/check.hpp"

#include <xpu/buffer.hpp>
#include <xpu/launch.hpp>
#include <xpu/memory.hpp>

#include <array>

template <typename T>
struct sum_values {
  [[nodiscard]] DEVICE_ONLY
  auto operator()(const xpu::array<std::size_t, 2uz>&) -> T {
    const T value{-1};

    return value;
  }

  [[nodiscard]] DEVICE_ONLY
  auto operator()(const xpu::array<std::size_t, 2uz>& index) const -> T {
    const auto value{static_cast<T>(index[0] + index[1] % 7uz)};

    return value;
  }
};

struct nonconst_sum {
  DEVICE_ONLY
  auto operator()(const xpu::array<std::size_t, 2uz>&) -> int;
};

template <typename T, typename F>
concept supports_sum = requires(const xpu::range<2uz>& range, T* output, F contribution) {
  xpu::parallel_reduce_sum(range, output, contribution);
  xpu::parallel_reduce_sum_bytes<T>(range, contribution);
};

static_assert(supports_sum<int, sum_values<int>>);
static_assert(supports_sum<float, sum_values<float>>);
static_assert(supports_sum<double, sum_values<double>>);
static_assert(!supports_sum<int, nonconst_sum>);
static_assert(!supports_sum<int, sum_values<double>>);
static_assert(!supports_sum<double, sum_values<float>>);

template <typename T>
inline auto run_sum_type() -> int {
  constexpr std::array columns{0uz, 1uz, 3uz, 4097uz};
  sum_values<T> contribution{};
  xpu::buffer<T> output{1uz};

  for (const auto count : columns) {
    const xpu::range<2uz> range{
      {2uz, 1uz}, {6uz, 1uz + 3uz * count}, {2uz, 3uz}
    };
    const auto required_bytes{xpu::parallel_reduce_sum_bytes<T>(range, contribution)};
    const auto empty{count == 0uz};
    const auto invalid_empty_query{empty && required_bytes != 0uz};

    if (invalid_empty_query) {
      const auto failure{test::fail("empty reduction requested scratch storage")};

      return failure;
    }

    T expected{};

    for (auto row{2uz}; row < 6uz; row += 2uz) {
      for (auto column{1uz}; column < 1uz + 3uz * count; column += 3uz) {
        expected += static_cast<T>(row + column % 7uz);
      }
    }

    const auto capacity{required_bytes + 64uz};
    xpu::buffer<std::byte> scratch{capacity};
    const T initial{91};
    xpu::copy_n(output.data(), &initial, 1uz);

    for (auto repeat{0uz}; repeat < 2uz; ++repeat) {
      xpu::parallel_reduce_sum(
        range, output.data(), contribution,
        empty ? nullptr : scratch.data(), empty ? 0uz : capacity
      );

      T actual{};
      xpu::copy_n(&actual, output.data(), 1uz);
      const auto incorrect_sum{actual != expected};

      if (incorrect_sum) {
        const auto failure{test::fail("reduction did not overwrite output with the expected sum")};

        return failure;
      }
    }
  }

  const auto success{0};

  return success;
}

inline auto run_sum_cases() -> int {
  if (const auto failure{run_sum_type<int>()}; failure) {

    return failure;
  }

  if (const auto failure{run_sum_type<float>()}; failure) {

    return failure;
  }

  const auto result{run_sum_type<double>()};

  return result;
}

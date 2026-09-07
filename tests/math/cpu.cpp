#include "cases.hpp"

#include <xpu/algorithm.hpp>
#include <xpu/buffer.hpp>

#include <array>
#include <thread>

namespace {

int check_atomic_add() {
  constexpr auto thread_count{8uz};
  static_assert(atomic_add_count % thread_count == 0uz);

  auto result{atomic_add_initial};
  {
    auto workers = std::array<std::jthread, thread_count>{};
    for (auto& worker : workers) {
      worker = std::jthread{[&result] {
        for (auto i{0uz}; i < atomic_add_count / thread_count; ++i) {
          xpu::atomic_add(&result, 1);
        }
      }};
    }
  }

  return check_atomic_add_result(result);
}

} // namespace

int main() {
  if (const auto status{check_scalar_math_cases()}; status != 0) {
    return status;
  }
  if (const auto status{check_atomic_add()}; status != 0) {
    return status;
  }

  auto a{xpu::buffer<float>{test::count}};
  auto b{xpu::buffer<float>{test::count}};
  auto result{xpu::buffer<float>{test::count}};
  xpu::fill_n(a.data(), a.count(), 3.0f);
  xpu::fill_n(b.data(), b.count(), 4.0f);

  for (auto i{0uz}; i < result.count(); ++i) {
    result.data()[i] = xpu::rsqrt(
      a.data()[i] * a.data()[i] + b.data()[i] * b.data()[i]
    );
  }

  return check_inverse_norm_results(result.data());
}

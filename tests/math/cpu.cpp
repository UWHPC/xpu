#include "cases.hpp"

#include <xpu/algorithm.hpp>
#include <xpu/buffer.hpp>

int main() {
  if (const auto status{check_scalar_math_cases()}; status != 0) {
    return status;
  }

  xpu::buffer<float> a{test::count};
  xpu::buffer<float> b{test::count};
  xpu::buffer<float> result{test::count};
  xpu::fill_n(a.data(), a.count(), 3.0f);
  xpu::fill_n(b.data(), b.count(), 4.0f);

  for (auto i{0uz}; i < result.count(); ++i) {
    result.data()[i] = xpu::rsqrt(
      a.data()[i] * a.data()[i] + b.data()[i] * b.data()[i]
    );
  }

  return check_inverse_norm_results(result.data());
}

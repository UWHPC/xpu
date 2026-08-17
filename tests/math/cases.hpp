#pragma once

#include "../support/check.hpp"

#include <xpu/math.hpp>

inline int check_scalar_math_cases() {
  if (xpu::ceiling_div(10u, 3u) != 4u) {
    return test::fail("xpu::ceiling_div failed");
  }
  if (!test::near(xpu::norm3d(2.0, 3.0, 6.0), 7.0, 1e-12)) {
    return test::fail("xpu::norm3d failed");
  }
  if (!test::near(xpu::rnorm3d(2.0, 3.0, 6.0), 1.0 / 7.0, 1e-12)) {
    return test::fail("xpu::rnorm3d failed");
  }

  double sine{};
  double cosine{};
  xpu::sincos(0.5, &sine, &cosine);
  if (
    !test::near(sine, xpu::sin(0.5), 1e-12) ||
    !test::near(cosine, xpu::cos(0.5), 1e-12)
  ) {
    return test::fail("xpu::sincos failed");
  }

  return 0;
}

inline int check_inverse_norm_results(const float* result) {
  for (auto i{0uz}; i < test::count; ++i) {
    if (!test::near(result[i], 0.2f, 1e-6f)) {
      return test::fail("xpu::rsqrt failed");
    }
  }

  return 0;
}

#pragma once

#include "../support/check.hpp"

#include <xpu/config.hpp>

inline int check_simd_width() {
  if (
    xpu::simd_bytes != 16uz &&
    xpu::simd_bytes != 32uz &&
    xpu::simd_bytes != 64uz
  ) {
    return test::fail("unsupported SIMD width");
  }

  return 0;
}

#include "cases.hpp"

int main() {
  if (const auto status{check_simd_width()}; status != 0) {
    return status;
  }
  if (xpu::xpu_cuda || xpu::xpu_hip || xpu::xpu_gpu) {
    return test::fail("CPU configuration flags are incorrect");
  }

  return 0;
}

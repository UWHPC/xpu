#include "cases.hpp"

int main() {
  if (const auto status{check_simd_width()}; status != 0) {
    return status;
  }
  if (!xpu::xpu_cuda) {
    return test::fail("CUDA configuration flag is incorrect");
  }

  return 0;
}

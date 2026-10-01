#include "cases.hpp"
#include "../support/device.hpp"

namespace {

__global__ void write_soa_view(xpu::soa_view<int, 2> view) {
  const auto index{static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x};
  if (index < view.count()) {
    view[0][index] = static_cast<int>(index + 1uz);
    view[1][index] = static_cast<int>(2uz * (index + 1uz));
  }
}

__global__ void read_soa_view(const xpu::soa_view<int, 2> input, xpu::soa_view<int, 2> output) {
  const auto index{static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x};
  if (index < input.count()) {
    output[0][index] = (2 * input[0][index]);
    output[1][index] = input[1][index] + 1;
  }
}

}

auto run_device_soa_view_cases() -> int {
  constexpr auto count{8uz};
  auto values{xpu::soa<int, 2>{count}};
  auto output{xpu::soa<int, 2>{count}};
  write_soa_view<<<1, 32>>>(values.view());
  test::check_launch();
  read_soa_view<<<1, 32>>>(values.view(), output.view());
  test::check_launch();
  test::synchronize();

  int result[2][count]{};
  xpu::copy_n(result[0], output.view()[0], count);
  xpu::copy_n(result[1], output.view()[1], count);
  for (auto i{0uz}; i < count; ++i) {
    if (result[0][i] != static_cast<int>(2uz * (i + 1uz)) 
    ||  result[1][i] != static_cast<int>((2uz * (i + 1uz)) + 1uz)) {
      return test::fail("GPU SoA View Indexing is incorrect");
    }
  }
  return 0;
}

int main() {
  if (const auto failure{run_device_soa_view_cases()}; failure) {
    return failure;
  }

  return run_memory_cases();
}

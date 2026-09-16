#include "cases.hpp"

namespace {

__global__ void write_buffer_view(xpu::buffer_view<int> view) {
  const auto index{static_cast<std::size_t>(threadIdx.x)};
  if (index < view.count()) {
    view[index] = static_cast<int>(index + 1uz);
  }
}

__global__ void read_buffer_view(
  const xpu::buffer_view<const int> input,
  xpu::buffer_view<int> output
) {
  const auto index{static_cast<std::size_t>(threadIdx.x)};
  if (index < input.count()) {
    output[index] = input[index] * 2;
  }
}

auto run_device_buffer_view_cases() -> int {
  constexpr auto count{8uz};
  auto values{xpu::buffer<int>{count}};
  auto output{xpu::buffer<int>{count}};
  write_buffer_view<<<1, 32>>>(values.view());
  xpu::cu_check(cudaGetLastError());
  read_buffer_view<<<1, 32>>>(std::as_const(values).view(), output.view());
  xpu::cu_check(cudaGetLastError());
  xpu::cu_check(cudaDeviceSynchronize());

  int result[count]{};
  xpu::copy_n(result, output.data(), count);
  for (auto i{0uz}; i < count; ++i) {
    if (result[i] != static_cast<int>(2uz * (i + 1uz))) {
      return test::fail("CUDA buffer view indexing is incorrect");
    }
  }
  return 0;
}

} // namespace

int main() {
  if (const auto failure{run_device_buffer_view_cases()}; failure) {
    return failure;
  }
  return run_buffer_cases();
}

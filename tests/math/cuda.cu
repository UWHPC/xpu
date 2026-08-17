#include "cases.hpp"

#include <xpu/algorithm.hpp>
#include <xpu/buffer.hpp>
#include <xpu/memory.hpp>

#include <memory>

namespace {

__global__
void compute_inverse_norms(
  float* result,
  const float* a,
  const float* b,
  std::size_t count
) {
  const auto [index]{xpu::global_index<1>()};
  if (index >= count) { return; }

  result[index] = xpu::rsqrt(
    a[index] * a[index] + b[index] * b[index]
  );
}

} // namespace

int main() {
  if (const auto status{check_scalar_math_cases()}; status != 0) {
    return status;
  }

  xpu::buffer<float> a{test::count};
  xpu::buffer<float> b{test::count};
  xpu::buffer<float> result{test::count};
  xpu::fill_n(a.data(), a.count(), 3.0f);
  xpu::fill_n(b.data(), b.count(), 4.0f);

  constexpr auto threads{64u};
  const auto blocks{xpu::block_per_dim(test::count, threads)};
  compute_inverse_norms<<<blocks, threads>>>(
    result.data(), a.data(), b.data(), result.count()
  );
  xpu::cu_check(cudaGetLastError());
  xpu::cu_check(cudaDeviceSynchronize());

  auto host_result{std::make_unique<float[]>(test::count)};
  xpu::copy_n(host_result.get(), result.data(), test::count);
  return check_inverse_norm_results(host_result.get());
}

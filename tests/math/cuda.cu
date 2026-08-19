#include "cases.hpp"

#include <xpu/algorithm.hpp>
#include <xpu/buffer.hpp>
#include <xpu/memory.hpp>

#include <memory>

namespace {

__global__
void add_under_contention(int* result, std::size_t count) {
  const auto [index]{xpu::global_index<1>()};
  if (index >= count) { return; }

  xpu::atomic_add(result, 1);
}

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

  xpu::buffer<int> atomic_result{1uz};
  xpu::fill_n(atomic_result.data(), atomic_result.count(), atomic_add_initial);
  constexpr auto atomic_threads{64u};
  const auto atomic_blocks{
    xpu::block_per_dim(atomic_add_count, atomic_threads)
  };
  add_under_contention<<<atomic_blocks, atomic_threads>>>(
    atomic_result.data(), atomic_add_count
  );
  xpu::cu_check(cudaGetLastError());
  xpu::cu_check(cudaDeviceSynchronize());

  auto host_atomic_result{0};
  xpu::copy_n(&host_atomic_result, atomic_result.data(), 1uz);
  if (const auto status{check_atomic_add_result(host_atomic_result)}; status != 0) {
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

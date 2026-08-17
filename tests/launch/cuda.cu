#include "../support/check.hpp"

#include <xpu/buffer.hpp>
#include <xpu/launch.hpp>
#include <xpu/memory.hpp>

#include <memory>

namespace {

__global__
void write_launch_coordinates(
  std::size_t* indices,
  std::size_t* strides,
  std::size_t count
) {
  const auto [index]{xpu::global_index<1>()};
  const auto [stride]{xpu::global_stride<1>()};
  if (index >= count) { return; }

  indices[index] = index;
  strides[index] = stride;
}

} // namespace

int main() {
  constexpr auto threads{64u};
  const auto blocks{xpu::block_per_dim(test::count, threads)};
  const auto expected_stride{static_cast<std::size_t>(threads) * blocks};

  xpu::buffer<std::size_t> indices{test::count};
  xpu::buffer<std::size_t> strides{test::count};
  write_launch_coordinates<<<blocks, threads>>>(
    indices.data(), strides.data(), test::count
  );
  xpu::cu_check(cudaGetLastError());
  xpu::cu_check(cudaDeviceSynchronize());

  auto host_indices{std::make_unique<std::size_t[]>(test::count)};
  auto host_strides{std::make_unique<std::size_t[]>(test::count)};
  xpu::copy_n(host_indices.get(), indices.data(), test::count);
  xpu::copy_n(host_strides.get(), strides.data(), test::count);

  for (auto i{0uz}; i < test::count; ++i) {
    if (host_indices[i] != i || host_strides[i] != expected_stride) {
      return test::fail("CUDA launch coordinates are incorrect");
    }
  }

  return 0;
}

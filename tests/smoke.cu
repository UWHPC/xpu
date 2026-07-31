#include <xpu/xpu.hpp>

namespace {

__global__
void rnorm_arrays(float* rnorm, float* a, float* b, std::size_t N) {
  const auto [first]{xpu::global_index<1>()};
  const auto [stride]{xpu::global_stride<1>()};

  for (auto i{first}; i < N; i += stride) {
    rnorm[i] = xpu::rsqrt(a[i] * a[i] + b[i] * b[i]);
  }
} 

enum Axis : std::size_t { X, Y, Z, NUM };

} // namespace

int main() {
  constexpr auto N{16'384uz};
  constexpr auto EPS{1e-5f};

  xpu::soa<float, Axis::NUM> soa{N};

  xpu::buffer<float> buffer_x{N};
  xpu::buffer<float> buffer_y{N};
  xpu::buffer<float> buffer_z{N};

  const auto val_x{2.0f};
  const auto val_y{val_x + 1.0f};
  const auto val_z{val_y + 1.0f};

  xpu::fill_n(buffer_x.data(), N, val_x);
  xpu::fill_n(buffer_y.data(), N, val_y);
  xpu::fill_n(buffer_z.data(), N, val_z);

  dim3 threads{256u};
  dim3 blocks{xpu::blocks<1>(rnorm_arrays, threads, N)};

  rnorm_arrays<<<blocks, threads>>>(soa[Axis::X], buffer_y.data(), buffer_z.data(), N);
  rnorm_arrays<<<blocks, threads>>>(soa[Axis::Y], buffer_z.data(), buffer_x.data(), N);
  rnorm_arrays<<<blocks, threads>>>(soa[Axis::Z], buffer_x.data(), buffer_y.data(), N);

  xpu::cuda_check(cudaDeviceSynchronize());

  auto* test_soa{new float[soa.storage_size()]{0.0f}};
  xpu::cuda_check(cudaMemcpy(test_soa, soa[Axis::X], soa.storage_size() * sizeof(float), cudaMemcpyDeviceToHost));

  for (auto i{0uz}; i < soa.count(); ++i) {
    const auto d1{test_soa[i] - xpu::rsqrt(val_y * val_y + val_z * val_z)};
    const auto d2{test_soa[i + soa.stride()] - xpu::rsqrt(val_x * val_x + val_z * val_z)};
    const auto d3{test_soa[i + 2uz * soa.stride()] - xpu::rsqrt(val_y * val_y + val_x * val_x)};

    const auto pass{
      xpu::abs(d1) <= EPS &&
      xpu::abs(d2) <= EPS &&
      xpu::abs(d3) <= EPS
    };

    if (!pass) {
      delete[] test_soa;
      return -1;
    }
  }

  delete[] test_soa;
  return 0;
}

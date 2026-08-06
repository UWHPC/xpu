#include "test.hpp"

namespace {

void rnorm_arrays(float* rnorm, float* a, float* b, std::size_t N) {
  for (auto i{0uz}; i < N; ++i) {
    rnorm[i] = xpu::rsqrt(a[i] * a[i] + b[i] * b[i]);
  }
} 

enum Axis : std::size_t { X, Y, Z, NUM };

} // namespace

int main() {
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

  rnorm_arrays(soa[Axis::X], buffer_y.data(), buffer_z.data(), N);
  rnorm_arrays(soa[Axis::Y], buffer_z.data(), buffer_x.data(), N);
  rnorm_arrays(soa[Axis::Z], buffer_x.data(), buffer_y.data(), N);

  for (auto i{0uz}; i < soa.count(); ++i) {
    const auto d1{soa[Axis::X][i] - xpu::rsqrt(val_y * val_y + val_z * val_z)};
    const auto d2{soa[Axis::Y][i] - xpu::rsqrt(val_x * val_x + val_z * val_z)};
    const auto d3{soa[Axis::Z][i] - xpu::rsqrt(val_y * val_y + val_x * val_x)};

    const auto pass{
      xpu::abs(d1) <= EPS &&
      xpu::abs(d2) <= EPS &&
      xpu::abs(d3) <= EPS
    };

    if (!pass) {
      return -1;
    }
  }

  return 0;
}

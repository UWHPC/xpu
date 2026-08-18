#include "cases.hpp"

#include <xpu/buffer.hpp>
#include <xpu/config.hpp>
#include <xpu/memory.hpp>
#include <xpu/random.hpp>

#include <array>

__global__
void generate_random_samples(random_sample* samples) {
  xpu::random::generator generator;
  generator.seed(42uz, 3uz, 1uz);

  for (auto i{0uz}; i < random_sample_count; ++i) {
    samples[i] = {
      generator.uniform<float>(),
      generator.uniform<double>(),
      generator.uniform(-2.0, 3.0),
      generator.uniform_index(7uz)
    };
  }
}

int main() {
  xpu::buffer<random_sample> samples{random_sample_count};
  generate_random_samples<<<1u, 1u>>>(samples.data());
  xpu::cu_check(cudaGetLastError());

  std::array<random_sample, random_sample_count> host_samples{};
  xpu::copy_n(
    host_samples.data(), samples.data(), random_sample_count
  );

  return check_random_samples(host_samples.data());
}

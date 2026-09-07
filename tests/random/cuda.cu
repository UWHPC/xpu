#include "cases.hpp"

#include <xpu/buffer.hpp>
#include <xpu/config.hpp>
#include <xpu/memory.hpp>
#include <xpu/random.hpp>

#include <array>

__global__
void generate_random_samples(random_sample* samples) {
  auto generator{xpu::random::generator{}};
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
  if (const auto failure{run_batched_seed_cases()}; failure) {

    return failure;
  }

  auto samples{xpu::buffer<random_sample>{random_sample_count}};
  generate_random_samples<<<1u, 1u>>>(samples.data());
  xpu::cu_check(cudaGetLastError());

  auto host_samples = std::array<random_sample, random_sample_count>{};
  xpu::copy_n(
    host_samples.data(), samples.data(), random_sample_count
  );

  return check_random_samples(host_samples.data());
}

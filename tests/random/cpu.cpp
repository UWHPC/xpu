#include "cases.hpp"

#include <xpu/random.hpp>

#include <array>

int main() {
  xpu::random::generator generator;
  generator.seed(42uz, 3uz, 1uz);

  std::array<random_sample, random_sample_count> samples{};
  for (auto& sample : samples) {
    sample = {
      generator.uniform<float>(),
      generator.uniform<double>(),
      generator.uniform(-2.0, 3.0),
      generator.uniform_index(7uz)
    };
  }

  return check_random_samples(samples.data());
}

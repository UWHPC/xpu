#pragma once

#include "../support/check.hpp"

#include <cstddef>

struct random_sample {
  float uniform_float;
  double uniform_double;
  double bounded;
  std::size_t index;
};

inline constexpr auto random_sample_count{256uz};

inline int check_random_samples(const random_sample* samples) {
  for (auto i{0uz}; i < random_sample_count; ++i) {
    const auto& sample{samples[i]};

    if (sample.uniform_float < 0.0f || sample.uniform_float >= 1.0f) {
      return test::fail("float uniform sample is outside [0, 1)");
    }
    if (sample.uniform_double < 0.0 || sample.uniform_double >= 1.0) {
      return test::fail("double uniform sample is outside [0, 1)");
    }
    if (sample.bounded < -2.0 || sample.bounded >= 3.0) {
      return test::fail("bounded uniform sample is outside its interval");
    }
    if (sample.index >= 7uz) {
      return test::fail("uniform_index sample is outside its interval");
    }
  }

  return 0;
}

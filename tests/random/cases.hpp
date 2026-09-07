#pragma once

#include "../support/check.hpp"

#include <xpu/buffer.hpp>
#include <xpu/random.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

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

struct compare_seeded_generators {
  xpu::random::generator* generators;
  const std::uint64_t* stream_ids;
  int* results;
  std::uint64_t master_seed;
  std::uint64_t offset;

  DEVICE_ONLY
  auto operator()(const xpu::array<std::size_t, 1uz>& index) const -> void {
    const auto i{index[0]};
    auto reference{xpu::random::generator{}};
    reference.seed(master_seed, stream_ids[i], offset);

    auto matches{true};

    for (auto sample{0uz}; sample < 3uz; ++sample) {
      const auto actual{generators[i].uniform<double>()};
      const auto expected{reference.uniform<double>()};

      matches = matches && actual == expected;
    }

    results[i] = matches;
  }
};

inline auto run_batched_seed_cases() -> int {
  constexpr auto count{4uz};
  constexpr auto master_seed{std::uint64_t{0x123456789abcdef0ULL}};
  constexpr auto stream_ids = std::array<std::array<std::uint64_t, count>, 3uz>{{
    {19u, 3u, 19u, 0x100000003ULL},
    {0x100000003ULL, 19u, 3u, 19u},
    {19u, 3u, 19u, 0x100000003ULL}
  }};
  constexpr auto offsets = std::array<std::uint64_t, 3uz>{0u, 5u, 0u};

  xpu::random::seed_n(nullptr, nullptr, 0uz, master_seed);

  auto generators{xpu::buffer<xpu::random::generator>{count}};
  auto streams{xpu::buffer<std::uint64_t>{count}};
  auto results{xpu::buffer<int>{count}};
  auto host_results = std::array<int, count>{};

  const auto range = xpu::range<1uz>{{0uz}, {count}, {1uz}};

  for (auto round{0uz}; round < offsets.size(); ++round) {
    xpu::copy_n(streams.data(), stream_ids[round].data(), count);
    xpu::random::seed_n(
      generators.data(), streams.data(), count, master_seed, offsets[round]
    );
    xpu::random::seed_n(generators.data(), streams.data(), 0uz, master_seed + 1u);

    const auto compare = compare_seeded_generators{
      generators.data(), streams.data(), results.data(), master_seed, offsets[round]
    };

    xpu::parallel_for(range, compare);
    xpu::copy_n(host_results.data(), results.data(), count);

    for (const auto matches : host_results) {
      if (const auto failed{!matches}; failed) {
        const auto failure{test::fail("batched seeding differs from individual seeding")};

        return failure;
      }
    }
  }

  const auto success{0};

  return success;
}

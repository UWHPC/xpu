#pragma once

#include <xpu/config.hpp>
#include <xpu/launch.hpp>

#if defined(XPU_CUDA)
  #include <curand_kernel.h>
#else
  #include <array>
#endif

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace xpu {

namespace random {

#if !defined(XPU_CUDA)
namespace detail {

// Philox4x32-10, the generator the CUDA backend gets from cuRAND
[[nodiscard]]
inline auto philox4x32(
  std::uint64_t counter_low,
  std::uint64_t counter_high,
  std::array<std::uint32_t, 2uz> key
) noexcept -> std::array<std::uint32_t, 4uz> {
  constexpr auto rounds{10};
  constexpr auto multiplier_0{std::uint64_t{0xD2511F53u}};
  constexpr auto multiplier_1{std::uint64_t{0xCD9E8D57u}};
  constexpr auto weyl_0{std::uint32_t{0x9E3779B9u}};
  constexpr auto weyl_1{std::uint32_t{0xBB67AE85u}};

  auto counter = std::array<std::uint32_t, 4uz>{
    static_cast<std::uint32_t>(counter_low),
    static_cast<std::uint32_t>(counter_low >> 32u),
    static_cast<std::uint32_t>(counter_high),
    static_cast<std::uint32_t>(counter_high >> 32u)
  };

  for (auto round{0}; round < rounds; ++round) {
    const auto product_0{multiplier_0 * counter[0]};
    const auto product_1{multiplier_1 * counter[2]};

    counter = {
      static_cast<std::uint32_t>(product_1 >> 32u) ^ counter[1] ^ key[0],
      static_cast<std::uint32_t>(product_1),
      static_cast<std::uint32_t>(product_0 >> 32u) ^ counter[3] ^ key[1],
      static_cast<std::uint32_t>(product_0)
    };

    key[0] += weyl_0;
    key[1] += weyl_1;
  }

  return counter;
}

} // namespace xpu::random::detail
#endif

class generator {
private:
#if defined(XPU_CUDA)
  curandStatePhilox4_32_10_t engine_;

  [[nodiscard]] XPU_DEVICE_ONLY
  auto next_u32() -> std::uint32_t {
    return curand(&engine_);
  }
#else
  std::uint64_t counter_low_;
  std::uint64_t counter_high_;
  std::array<std::uint32_t, 2uz> key_;
  std::array<std::uint32_t, 4uz> output_;
  std::uint32_t position_;

  [[nodiscard]]
  auto next_u32() noexcept -> std::uint32_t {
    const auto value{output_[position_++]};
    const auto exhausted{position_ == 4u};

    if (exhausted) {
      counter_high_ += ++counter_low_ == 0u;
      output_ = detail::philox4x32(counter_low_, counter_high_, key_);
      position_ = 0u;
    }

    return value;
  }
#endif

  [[nodiscard]] XPU_DEVICE_ONLY
  auto next_u64() -> std::uint64_t {
    const auto high{static_cast<std::uint64_t>(next_u32())};
    const auto low{static_cast<std::uint64_t>(next_u32())};
    return (high << 32u) | low;
  }

public:
  XPU_DEVICE_ONLY
  auto seed(
    std::uint64_t master_seed,
    std::uint64_t stream_id = 0,
    std::uint64_t offset = 0
  ) -> void {
#if defined(XPU_CUDA)
    curand_init(master_seed, stream_id, offset, &engine_);
#else
    // same layout as curand_init: the stream is the high counter half, the
    // offset counts 32-bit outputs, four per counter value
    counter_low_ = offset >> 2u;
    counter_high_ = stream_id;
    key_ = {
      static_cast<std::uint32_t>(master_seed),
      static_cast<std::uint32_t>(master_seed >> 32u)
    };
    output_ = detail::philox4x32(counter_low_, counter_high_, key_);
    position_ = static_cast<std::uint32_t>(offset & 3u);
#endif
  }

  template <supported_float T> [[nodiscard]] XPU_DEVICE_ONLY
  auto uniform() -> T {
#if defined(XPU_CUDA)
    if constexpr (std::same_as<T, float>) {
      return T{1} - curand_uniform(&engine_);
    } else {
      return T{1} - curand_uniform_double(&engine_);
    }
#else
    // curand_uniform and curand_uniform_double, bit for bit
    if constexpr (std::same_as<T, float>) {
      const auto value{static_cast<float>(next_u32())};
      return T{1} - (value * 0x1p-32f + 0x1p-33f);
    } else {
      const auto value{static_cast<double>(next_u32())};
      return T{1} - (value * 0x1p-32 + 0x1p-32);
    }
#endif
  }

  template <supported_float T> [[nodiscard]] XPU_DEVICE_ONLY
  auto uniform(T minimum, T maximum) -> T {
    assert(minimum < maximum);
    const auto value{minimum + (maximum - minimum) * uniform<T>()};
    if (value < maximum) { return value; }

#if defined(XPU_CUDA)
    if constexpr (std::same_as<T, float>) {
      return ::nextafterf(maximum, minimum);
    } else {
      return ::nextafter(maximum, minimum);
    }
#else
    return std::nextafter(maximum, minimum);
#endif
  }

  [[nodiscard]] XPU_DEVICE_ONLY
  auto uniform_index(std::size_t upper_bound) -> std::size_t {
    assert(upper_bound != 0uz);

    // Lemire's multiply-shift rejection: no modulo unless a draw lands in
    // the biased sliver, and bounds below 2^32 use one 32-bit output
    const auto bound{static_cast<std::uint64_t>(upper_bound)};
    const auto narrow{(bound >> 32u) == 0u};

    if (narrow) {
      const auto bound_32{static_cast<std::uint32_t>(bound)};
      auto product{static_cast<std::uint64_t>(next_u32()) * bound_32};
      auto low{static_cast<std::uint32_t>(product)};

      if (low < bound_32) {
        const auto threshold{(std::uint32_t{0} - bound_32) % bound_32};

        while (low < threshold) {
          product = static_cast<std::uint64_t>(next_u32()) * bound_32;
          low = static_cast<std::uint32_t>(product);
        }
      }

      return static_cast<std::size_t>(product >> 32u);
    }

    auto value{next_u64()};
    auto low{value * bound};

    if (low < bound) {
      const auto threshold{(std::uint64_t{0} - bound) % bound};

      while (low < threshold) {
        value = next_u64();
        low = value * bound;
      }
    }

    return static_cast<std::size_t>(xpu::detail::mul_high(value, bound));
  }
};

namespace detail {

struct seed_generators {
  generator* generators;
  const std::uint64_t* stream_ids;
  std::uint64_t master_seed;
  std::uint64_t offset;

  XPU_DEVICE_ONLY
  auto operator()(const xpu::array<std::size_t, 1uz>& index) const -> void {
    const auto i{index[0]};

    generators[i].seed(master_seed, stream_ids[i], offset);
  }
};

} // namespace xpu::random::detail

inline auto seed_n(
  generator* generators,
  const std::uint64_t* stream_ids,
  std::size_t count,
  std::uint64_t master_seed,
  std::uint64_t offset = 0
) -> void {
  const auto range = xpu::range<1uz>{
    {0uz},
    {count},
    {1uz}
  };

  const auto seed = detail::seed_generators{
    generators, stream_ids, master_seed, offset
  };

  xpu::parallel_for(range, seed);
}

} // namespace xpu::random

} // namespace xpu

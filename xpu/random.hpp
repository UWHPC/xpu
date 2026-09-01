#pragma once

#include <xpu/config.hpp>

#if defined(XPU_CUDA)
  #include <curand_kernel.h>
#else
  #include <random>
#endif

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace xpu {

namespace random {

class generator {
private:
#if defined(XPU_CUDA)
  curandStatePhilox4_32_10_t engine_;

  [[nodiscard]] DEVICE_ONLY
  auto next_u32() -> std::uint32_t {
    return curand(&engine_);
  }

  [[nodiscard]] DEVICE_ONLY
  auto next_u64() -> std::uint64_t {
    const auto high{static_cast<std::uint64_t>(next_u32())};
    const auto low{static_cast<std::uint64_t>(next_u32())};
    return (high << 32u) | low;
  }
#else
  std::mt19937_64 engine_;
#endif

public:
  DEVICE_ONLY
  auto seed(
    std::uint64_t master_seed,
    std::uint64_t stream_id = 0,
    std::uint64_t offset = 0
  ) -> void {
#if defined(XPU_CUDA)
    curand_init(master_seed, stream_id, offset, &engine_);
#else
    std::seed_seq seed{
      static_cast<std::uint32_t>(master_seed),
      static_cast<std::uint32_t>(master_seed >> 32u),
      static_cast<std::uint32_t>(stream_id),
      static_cast<std::uint32_t>(stream_id >> 32u)
    };
    engine_.seed(seed);
    engine_.discard(offset);
#endif
  }

  template <supported_float T> [[nodiscard]] DEVICE_ONLY
  auto uniform() -> T {
#if defined(XPU_CUDA)
    if constexpr (std::same_as<T, float>) {
      return T{1} - curand_uniform(&engine_);
    } else {
      return T{1} - curand_uniform_double(&engine_);
    }
#else
    return std::generate_canonical<T, std::numeric_limits<T>::digits>(engine_);
#endif
  }

  template <supported_float T> [[nodiscard]] DEVICE_ONLY
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

  [[nodiscard]] DEVICE_ONLY
  auto uniform_index(std::size_t upper_bound) -> std::size_t {
    assert(upper_bound != 0uz);

#if defined(XPU_CUDA)
    const auto bound{static_cast<std::uint64_t>(upper_bound)};
    const auto threshold{(std::uint64_t{0} - bound) % bound};

    auto value{next_u64()};
    while (value < threshold) {
      value = next_u64();
    }

    return static_cast<std::size_t>(value % bound);
#else
    std::uniform_int_distribution<std::size_t> distribution{
      0uz, upper_bound - 1uz
    };
    return distribution(engine_);
#endif
  }
};

} // namespace xpu::random

} // namespace xpu

#pragma once

#include <xpu/config.hpp>
#include <xpu/math.hpp>

namespace xpu {

#if defined(XPU_CUDA)

[[nodiscard]]
inline unsigned int num_blocks(std::size_t n, unsigned int threads) {
  return static_cast<unsigned int>(ceiling_div<std::size_t>(n, threads));
}

template <int Dims> struct GlobalId;
template <> struct GlobalId<1> { std::size_t x{}; };
template <> struct GlobalId<2> { std::size_t x{}, y{}; };
template <> struct GlobalId<3> { std::size_t x{}, y{}, z{}; };

template <int Dims> __device__ [[nodiscard]]
inline GlobalId<Dims> global_index() noexcept {
  GlobalId<Dims> id{};
  
  if constexpr (Dims >= 1) {
    id.x = static_cast<std::size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  }
  if constexpr (Dims >= 2) {
    id.y = static_cast<std::size_t>(blockDim.y) * blockIdx.y + threadIdx.y;
  }
  if constexpr (Dims >= 3) {
    id.z = static_cast<std::size_t>(blockDim.z) * blockIdx.z + threadIdx.z;
  }

  return id;
}

#endif

}
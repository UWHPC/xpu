#pragma once

#include <xpu/config.hpp>
#include <xpu/math.hpp>

namespace xpu {

#if defined(XPU_CUDA)

namespace detail {

[[nodiscard]]
inline unsigned int device_SMs() {
  static const unsigned int cached{[] {
      auto device{0}, sms{0};
      xpu::cuda_check(cudaGetDevice(&device));
      xpu::cuda_check(cudaDeviceGetAttribute(
        &sms, cudaDevAttrMultiProcessorCount, device
      ));
      return static_cast<unsigned int>(sms);
    }();
  }
  return cached;
}

template <typename Kernel> [[nodiscard]]
inline unsigned int wave_blocks(Kernel kernel, unsigned int threads, std::size_t smem = 0uz) {
  auto blocks_per_SM{0};
  cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_SM, kernel, static_cast<int>(threads), smem
  ));
  return device_SMs() * static_cast<unsigned int>(blocks_per_SM);
}

[[nodiscard]]
inline unsigned int num_blocks(std::size_t n, unsigned int threads) {
  return static_cast<unsigned int>(ceiling_div<std::size_t>(n, threads));
}

}

template <int Dims> struct Coord;
template <> struct Coord<1> { std::size_t x{}; };
template <> struct Coord<2> { std::size_t x{}, y{}; };
template <> struct Coord<3> { std::size_t x{}, y{}, z{}; };

template <int Dims> __device__ [[nodiscard]]
inline Coord<Dims> global_index() noexcept {
  Coord<Dims> id{};
  
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

template <int Dims> __device__ [[nodiscard]]
inline Coord<Dims> global_stride() noexcept {
  Coord<Dims> stride{};
  
  if constexpr (Dims >= 1) {
    stride.x = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  }
  if constexpr (Dims >= 2) {
    stride.y = static_cast<std::size_t>(blockDim.y) * gridDim.y;
  }
  if constexpr (Dims >= 3) {
    stride.z = static_cast<std::size_t>(blockDim.z) * gridDim.z;
  }

  return stride;
}

// TODO: fix + make this actually do what I want and apply changes to call sites.
template <int Dims, typename Kernel> [[nodiscard]]
inline dim3 blocks(std::size_t size, dim3 threads, Kernel kernel) {
  dim3 d{};

  if constexpr (Dims >= 1) {
    d.x = xpu::min(xpu::detail::num_blocks(size, threads.x), xpu::detail::wave_blocks(kernel, threads.x));
  }
  if constexpr (Dims >= 2) {
    d.y = xpu::min(xpu::detail::num_blocks(size, threads.y), xpu::detail::wave_blocks(kernel, threads.y));
  }
  if constexpr (Dims >= 3) {
    d.z = xpu::min(xpu::detail::num_blocks(size, threads.z), xpu::detail::wave_blocks(kernel, threads.z));
  }
  
  return d;
}

#endif

} // namespace xpu

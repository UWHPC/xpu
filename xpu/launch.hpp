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
    }()
  };
  return cached;
}

template <typename Kernel> [[nodiscard]]
inline unsigned int wave_blocks(Kernel kernel, dim3 threads, std::size_t smem = 0uz) {
  auto const thread_budget{threads.x * threads.y * threads.z};
  auto blocks_per_SM{0};

  cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_SM, kernel, static_cast<int>(thread_budget), smem
  ));

  return xpu::detail::device_SMs() * static_cast<unsigned int>(blocks_per_SM);
}

[[nodiscard]]
inline unsigned int num_blocks(std::size_t size, unsigned int threads) {
  return static_cast<unsigned int>(xpu::ceiling_div<std::size_t>(size, threads));
}

template <typename Kernel> [[nodiscard]]
inline unsigned int blocks_for(Kernel kernel, unsigned int threads, std::size_t size) {
  return xpu::min(
    xpu::detail::num_blocks(size, threads),
    xpu::detail::wave_blocks(kernel, threads)
  );
}

} // namespace xpu::detail

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

[[nodiscard]]
inline unsigned int block_per_dim(std::size_t size, unsigned int dim_threads) {
  return xpu::detail::num_blocks(size, dim_threads);
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
inline dim3 blocks(Kernel kernel, dim3 threads, std::size_t size) {
  dim3 d{};

  if constexpr (Dims >= 1) {
    d.x = xpu::detail::blocks_for(kernel, threads.x, size);
  }
  if constexpr (Dims >= 2) {
    d.y = xpu::detail::blocks_for(kernel, threads.y, size);
  }
  if constexpr (Dims >= 3) {
    d.z = xpu::detail::blocks_for(kernel, threads.z, size);
  }
  
  return d;
}

#endif

} // namespace xpu

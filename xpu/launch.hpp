#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>
#include <xpu/config.hpp>
#include <xpu/math.hpp>

#if defined(XPU_CUDA)
  #include <cuda/std/array>
#else
  #include <array>
#endif

namespace xpu {

#if defined(XPU_CUDA) && defined(__CUDACC__)

namespace detail {

[[nodiscard]]
inline auto device_SMs() -> unsigned int {
  static const unsigned int cached{[] {
      auto device{0}, sms{0};
      xpu::cu_check(cudaGetDevice(&device));
      xpu::cu_check(cudaDeviceGetAttribute(
        &sms, cudaDevAttrMultiProcessorCount, device
      ));
      return static_cast<unsigned int>(sms);
    }()
  };
  return cached;
}

template <typename Kernel> [[nodiscard]]
inline auto wave_blocks(Kernel kernel, dim3 threads, std::size_t smem = 0uz) -> unsigned int {
  auto const thread_budget{threads.x * threads.y * threads.z};
  auto blocks_per_SM{0};

  cu_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_SM, kernel, static_cast<int>(thread_budget), smem
  ));

  return xpu::detail::device_SMs() * static_cast<unsigned int>(blocks_per_SM);
}

[[nodiscard]]
inline constexpr auto num_blocks(std::size_t size, unsigned int threads) noexcept -> unsigned int {
  return static_cast<unsigned int>(xpu::ceiling_div<std::size_t>(size, threads));
}

template <typename Kernel> [[nodiscard]]
inline auto blocks_for(Kernel kernel, unsigned int threads, std::size_t size) -> unsigned int {
  return xpu::min(
    xpu::detail::num_blocks(size, threads),
    xpu::detail::wave_blocks(kernel, threads)
  );
}

} // namespace xpu::detail

[[nodiscard]] DEVICE_ONLY
inline auto linear_index() noexcept -> std::size_t {
  return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

[[nodiscard]] DEVICE_ONLY
inline auto linear_stride() noexcept -> std::size_t {
  return static_cast<std::size_t>(gridDim.x) * blockDim.x;
}

template <int Dims> struct Coord;
template <> struct Coord<1> { std::size_t x{}; };
template <> struct Coord<2> { std::size_t x{}, y{}; };
template <> struct Coord<3> { std::size_t x{}, y{}, z{}; };

template <int Dims> __device__ [[nodiscard]]
inline auto global_index() noexcept -> Coord<Dims> {
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

inline constexpr auto block_per_dim(std::size_t size, unsigned int dim_threads) noexcept -> unsigned int {
  return xpu::ceiling_div<unsigned int>(static_cast<int>(size), static_cast<int>(dim_threads));
}

template <int Dims> __device__ [[nodiscard]]
inline auto global_stride() noexcept -> Coord<Dims> {
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

#endif

using xstd::array;

template <std::size_t dims>
struct [[nodiscard]] range {
  xpu::array<std::size_t, dims> begin;
  xpu::array<std::size_t, dims> end;
  xpu::array<std::size_t, dims> step;
};

namespace detail {

template <std::size_t dims>
inline constexpr auto num_itrs(
  const xpu::range<dims>& range
) noexcept -> std::size_t {
  static_assert(dims > 0uz, "ERROR: Dimension must be greater than 0.");
  auto total{1uz};

  for (auto d{0uz}; d < dims; ++d) {
    if (range.step[d] == 0uz || range.begin[d] >= range.end[d]) {
      return 0uz;
    }

    const auto delta{range.end[d] - range.begin[d]};
    const auto count{xpu::ceiling_div(delta, range.step[d])};
    total *= count;
  }

  return total;
}

template <std::size_t dims> CUDA_CALLABLE
inline constexpr auto itr_index(
  const xpu::range<dims>& range,
  std::size_t linear
) noexcept -> xpu::array<std::size_t, dims> {
  static_assert(dims > 0uz, "ERROR: Dimension must be greater than 0.");
  xpu::array<std::size_t, dims> index{};

  for (auto d{dims}; d-- > 0uz;) {
    const auto delta{range.end[d] - range.begin[d]};
    const auto count{xpu::ceiling_div(delta, range.step[d])};

    index[d] = range.begin[d] + (linear % count) * range.step[d];
    linear /= count;
  }

  return index;
}

#if defined(XPU_CUDA)
template <std::size_t dims, typename F> __global__
auto parallelForLaunchImpl(
  xpu::range<dims> range,
  std::size_t total,
  F fcn
) -> void {
  for (
    auto linear{xpu::linear_index()};
    linear < total;
    linear += xpu::linear_stride()
  ) {
    fcn(xpu::detail::itr_index(range, linear));
  }
}
#endif

} // namespace xpu::detail

template <std::size_t dims, typename F>
inline auto parallel_for(
  const xpu::range<dims>& range,
  F&& fcn
) -> void {
  static_assert(dims > 0uz, "ERROR: Dimension must be greater than 0.");

  const auto total{xpu::detail::num_itrs(range)};
  if (total == 0uz) { return; }

#if defined(XPU_CUDA)
  using function_t = std::decay_t<F>;

  constexpr auto gpuThreads{256u};
  const auto gpuBlocks{xpu::detail::blocks_for(
    xpu::detail::parallelForLaunchImpl<dims, function_t>,
    gpuThreads,
    total
  )};

  xpu::detail::parallelForLaunchImpl<dims, function_t><<<
    gpuBlocks, gpuThreads
  >>>(
    range, total, function_t{std::forward<F>(fcn)}
  );
  xpu::cu_check(cudaGetLastError());
#else
  #pragma omp parallel for
  for (auto linear = 0uz; linear < total; ++linear) {
    fcn(xpu::detail::itr_index(range, linear));
  }
#endif
}

} // namespace xpu

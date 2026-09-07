#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>
#include <xpu/config.hpp>
#include <xpu/detail/checked.hpp>
#include <xpu/math.hpp>

#if defined(XPU_CUDA)
  #include <cub/device/device_reduce.cuh>
  #include <cuda/std/array>
  #include <thrust/iterator/counting_iterator.h>
  #include <thrust/iterator/transform_iterator.h>
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
      const auto multiprocessors{xpu::detail::checked_cast<unsigned int>(sms)};

      return multiprocessors;
    }()
  };
  return cached;
}

template <typename Kernel> [[nodiscard]]
inline auto wave_blocks(Kernel kernel, dim3 threads, std::size_t smem = 0uz) -> unsigned int {
  const auto thread_budget{xpu::detail::checked_mul(
    xpu::detail::checked_mul(threads.x, threads.y), threads.z
  )};
  auto blocks_per_SM{0};

  cu_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_SM, kernel, xpu::detail::checked_cast<int>(thread_budget), smem
  ));

  const auto blocks{xpu::detail::checked_mul(
    xpu::detail::device_SMs(),
    xpu::detail::checked_cast<unsigned int>(blocks_per_SM)
  )};

  return blocks;
}

[[nodiscard]]
inline constexpr auto num_blocks(std::size_t size, unsigned int threads) noexcept -> unsigned int {
  const auto zero_threads{threads == 0u};

  if (zero_threads) {
    xpu::detail::checked_error("zero launch threads");
  }

  const auto blocks{xpu::detail::checked_cast<unsigned int>(
    xpu::ceiling_div<std::size_t>(size, threads)
  )};

  return blocks;
}

template <typename Kernel> [[nodiscard]]
inline auto blocks_for(Kernel kernel, unsigned int threads, std::size_t size) -> unsigned int {
  const auto zero_threads{threads == 0u};

  if (zero_threads) {
    xpu::detail::checked_error("zero launch threads");
  }

  const auto needed_blocks{xpu::ceiling_div<std::size_t>(size, threads)};
  const auto available_blocks{xpu::detail::wave_blocks(kernel, threads)};
  const auto blocks{xpu::detail::checked_cast<unsigned int>(
    xpu::min(needed_blocks, static_cast<std::size_t>(available_blocks))
  )};

  return blocks;
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
  const auto blocks{xpu::detail::num_blocks(size, dim_threads)};

  return blocks;
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
    const auto empty_dimension{range.step[d] == 0uz || range.begin[d] >= range.end[d]};

    if (empty_dimension) {
      const auto empty_count{0uz};

      return empty_count;
    }
  }

  for (auto d{0uz}; d < dims; ++d) {
    const auto delta{range.end[d] - range.begin[d]};
    const auto count{xpu::ceiling_div(delta, range.step[d])};
    total = xpu::detail::checked_mul(total, count);
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

template <typename F, typename T, std::size_t dims>
concept sum_contribution = requires(
  const std::remove_reference_t<F>& contribution,
  const xpu::array<std::size_t, dims>& index
) {
  { contribution(index) } -> std::same_as<T>;
};

template <std::size_t dims, arithmetic T, typename F>
struct reduction_contribution {
  xpu::range<dims> range;
  F contribution;

  [[nodiscard]] CUDA_CALLABLE
  auto operator()(std::size_t linear) const -> T {
    const auto index{xpu::detail::itr_index(range, linear)};
    const T value{contribution(index)};

    return value;
  }
};

#if defined(XPU_CUDA)
template <arithmetic T, std::size_t dims, typename F>
inline auto reduction_input(
  const xpu::range<dims>& range,
  F&& contribution
) {
  using function_t = std::decay_t<F>;

  const reduction_contribution<dims, T, function_t> transform{
    range, function_t{std::forward<F>(contribution)}
  };

  const thrust::counting_iterator<std::size_t> indices{0uz};
  const auto input{thrust::make_transform_iterator(indices, transform)};

  return input;
}

template <arithmetic T, typename Input> [[nodiscard]]
inline auto reduction_bytes(Input input, std::ptrdiff_t count) -> std::size_t {
  std::size_t required_bytes{};

  xpu::cu_check(cub::DeviceReduce::Sum(
    nullptr, required_bytes, input, static_cast<T*>(nullptr), count
  ));

  return required_bytes;
}

template <std::size_t dims, typename F> __global__
auto cudaParallelForKernel(
  xpu::range<dims> range,
  std::size_t total,
  F fcn
) -> void {
  for (
    auto linear{xpu::linear_index()};
    linear < total;
  ) {
    fcn(xpu::detail::itr_index(range, linear));

    const auto stride{xpu::linear_stride()};
    const auto last_iteration{total - linear <= stride};

    if (last_iteration) {
      break;
    }

    linear += stride;
  }
}
#endif

template <std::size_t dims, typename F>
inline auto parallel_for_impl(
  const xpu::range<dims>& range,
  std::size_t total,
  F&& fcn
) -> void {
#if defined(XPU_CUDA)
  using function_t = std::decay_t<F>;

  constexpr auto gpuThreads{256u};
  const auto gpuBlocks{xpu::detail::blocks_for(
    xpu::detail::cudaParallelForKernel<dims, function_t>,
    gpuThreads,
    total
  )};

  xpu::detail::cudaParallelForKernel<dims, function_t><<<
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

template <std::size_t dims, arithmetic T, typename F>
inline auto parallel_reduce_sum_impl(
  const xpu::range<dims>& range,
  std::size_t total,
  T* output_ptr,
  F&& contribution,
  [[maybe_unused]] void* scratch = nullptr,
  [[maybe_unused]] std::size_t scratch_bytes = 0uz
) -> void {
  using function_t = std::decay_t<F>;

  const reduction_contribution<dims, T, function_t> transform{
    range, function_t{std::forward<F>(contribution)}
  };

#if defined(XPU_CUDA)
  const thrust::counting_iterator<std::size_t> indices{0uz};
  const auto input{thrust::make_transform_iterator(indices, transform)};

  const auto count{xpu::detail::checked_cast<std::ptrdiff_t>(total)};
  const auto required_bytes{reduction_bytes<T>(input, count)};

  const auto insufficient_scratch{
    scratch == nullptr || scratch_bytes < required_bytes
  };

  if (insufficient_scratch) {
    xpu::detail::checked_error("insufficient reduction scratch storage");
  }

  xpu::cu_check(cub::DeviceReduce::Sum(
    scratch, scratch_bytes, input, output_ptr, count
  ));
#else
  T total_sum{};

  #pragma omp parallel for reduction(+ : total_sum)
  for (auto linear = 0uz; linear < total; ++linear) {
    total_sum += transform(linear);
  }

  *output_ptr = total_sum;
#endif
}

} // namespace xpu::detail

template <std::size_t dims, typename F>
inline auto parallel_for(
  const xpu::range<dims>& range,
  F&& fcn
) -> void {
  static_assert(dims > 0uz, "ERROR: Dimension must be greater than 0.");

  const auto total{xpu::detail::num_itrs(range)};
  if (total == 0uz) { return; }

  detail::parallel_for_impl(range, total, std::forward<F>(fcn));
}

template <arithmetic T, std::size_t dims, typename F>
  requires detail::sum_contribution<F, T, dims>
[[nodiscard]]
inline auto parallel_reduce_sum_bytes(
  [[maybe_unused]] const xpu::range<dims>& range,
  [[maybe_unused]] F&& contribution
) -> std::size_t {
  static_assert(dims > 0uz, "ERROR: Dimension must be greater than 0.");

  std::size_t required_bytes{};

#if defined(XPU_CUDA)
  const auto total{xpu::detail::num_itrs(range)};
  const auto empty{total == 0uz};

  if (empty) {

    return required_bytes;
  }

  const auto input{detail::reduction_input<T>(range, std::forward<F>(contribution))};
  const auto count{xpu::detail::checked_cast<std::ptrdiff_t>(total)};

  required_bytes = detail::reduction_bytes<T>(input, count);
#endif

  return required_bytes;
}

template <std::size_t dims, arithmetic T, typename F>
  requires detail::sum_contribution<F, T, dims>
inline auto parallel_reduce_sum(
  const xpu::range<dims>& range,
  T* output_ptr,
  F&& contribution,
  void* scratch = nullptr,
  std::size_t scratch_bytes = 0uz
) -> void {
  static_assert(dims > 0uz, "ERROR: Dimension must be greater than 0.");

  const auto total{xpu::detail::num_itrs(range)};
  const auto empty{total == 0uz};

  if (empty) {
#if defined(XPU_CUDA)
    xpu::cu_check(cudaMemsetAsync(output_ptr, 0, sizeof(T)));
#else
    *output_ptr = T{};
#endif

    return;
  }

  detail::parallel_reduce_sum_impl(
    range, total, output_ptr, std::forward<F>(contribution), scratch, scratch_bytes
  );
}

} // namespace xpu

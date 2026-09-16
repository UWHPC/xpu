#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>
#include <xpu/config.hpp>
#include <xpu/detail/checked.hpp>
#include <xpu/math.hpp>

#if defined(XPU_CUDA)
  #include <cub/device/device_for.cuh>
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

/** @brief Return the number of streaming multiprocessors on the current CUDA device.
 *  @details The first result is cached for later calls.
 */
[[nodiscard]]
inline auto device_SMs() -> unsigned int {
  static const auto cached{[] {
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

/** @brief Return the active blocks per multiprocessor times the device's
 *  multiprocessor count for @p kernel, @p threads, and @p smem bytes of shared memory.
 */
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

/** @brief Return ceil(@p size / @p threads), aborting if @p threads is zero
 *  or the result cannot fit in an unsigned int.
 */
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

/** @brief Return the smaller of the blocks needed for @p size items and
 *  the blocks that fill one occupancy wave for @p kernel.
 *  @pre @p threads is nonzero.
 */
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

/** @brief Return blockIdx.x * blockDim.x + threadIdx.x. */
[[nodiscard]] DEVICE_ONLY
inline auto linear_index() noexcept -> std::size_t {
  return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

/** @brief Return gridDim.x * blockDim.x for a grid-stride loop. */
[[nodiscard]] DEVICE_ONLY
inline auto linear_stride() noexcept -> std::size_t {
  return static_cast<std::size_t>(gridDim.x) * blockDim.x;
}

/** @brief Unsigned global CUDA coordinates in one, two, or three dimensions. */
template <int Dims> struct Coord;
template <> struct Coord<1> { std::size_t x{}; };
template <> struct Coord<2> { std::size_t x{}, y{}; };
template <> struct Coord<3> { std::size_t x{}, y{}, z{}; };

/** @brief Return block index times block size plus thread index in each
 *  requested dimension.
 */
template <int Dims> __device__ [[nodiscard]]
inline auto global_index() noexcept -> Coord<Dims> {
  auto id = Coord<Dims>{};
  
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

/** @brief Return ceil(@p size / @p dim_threads) for one grid dimension. */
inline constexpr auto block_per_dim(std::size_t size, unsigned int dim_threads) noexcept -> unsigned int {
  const auto blocks{xpu::detail::num_blocks(size, dim_threads)};

  return blocks;
}

/** @brief Return grid size times block size in each requested dimension. */
template <int Dims> __device__ [[nodiscard]]
inline auto global_stride() noexcept -> Coord<Dims> {
  auto stride = Coord<Dims>{};
  
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

/** @brief Iterate each dimension from begin to end, excluding end, by step. */
template <std::size_t dims>
struct [[nodiscard]] range {
  /** @brief Inclusive starting index in each dimension. */
  xpu::array<std::size_t, dims> begin;
  /** @brief Exclusive ending index in each dimension. */
  xpu::array<std::size_t, dims> end;
  /** @brief Step size in each dimension; zero makes the range empty. */
  xpu::array<std::size_t, dims> step;
};

namespace detail {

/** @brief Multiply the iteration counts of all dimensions.
 *  @details Returns zero if any dimension has zero step or begin is at or past
 *  end. Aborts if the product overflows std::size_t.
 */
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

/** @brief Map @p linear to an index in @p range with the last dimension
 *  varying fastest.
 *  @pre @p linear is less than num_itrs(@p range).
 */
template <std::size_t dims> CUDA_CALLABLE
inline constexpr auto itr_index(
  const xpu::range<dims>& range,
  std::size_t linear
) noexcept -> xpu::array<std::size_t, dims> {
  static_assert(dims > 0uz, "ERROR: Dimension must be greater than 0.");
  auto index = xpu::array<std::size_t, dims>{};

  for (auto d{dims}; d-- > 1uz;) {
    const auto delta{range.end[d] - range.begin[d]};
    const auto count{xpu::ceiling_div(delta, range.step[d])};
    const auto coordinate{linear % count};

    index[d] = range.begin[d] + coordinate * range.step[d];
    linear /= count;
  }

  index[0] = range.begin[0] + linear * range.step[0];

  return index;
}

/** @brief Accept a callable that takes a range index and returns exactly T. */
template <typename F, typename T, std::size_t dims>
concept sum_contribution = requires(
  const std::remove_reference_t<F>& contribution,
  const xpu::array<std::size_t, dims>& index
) {
  { contribution(index) } -> std::same_as<T>;
};

/** @brief Expose multidimensional contributions as a one-dimensional
 *  sequence of T values for the reduction backend.
 */
template <std::size_t dims, arithmetic T, typename F>
struct reduction_contribution {
  xpu::range<dims> range;
  F contribution;

  /** @brief Convert @p linear to a range index and evaluate contribution there. */
  [[nodiscard]] CUDA_CALLABLE
  auto operator()(std::size_t linear) const -> T {
    const auto index{xpu::detail::itr_index(range, linear)};
    const auto value{T{contribution(index)}};

    return value;
  }
};

#if defined(XPU_CUDA)
/** @brief Return a transform iterator that evaluates @p contribution when
 *  successive indices of @p range are dereferenced, without storing values.
 */
template <arithmetic T, std::size_t dims, typename F>
inline auto reduction_input(
  const xpu::range<dims>& range,
  F&& contribution
) {
  using function_t = std::decay_t<F>;

  const auto transform = reduction_contribution<dims, T, function_t>{
    range, function_t{std::forward<F>(contribution)}
  };

  const auto indices{thrust::counting_iterator<std::size_t>{0uz}};
  const auto input{thrust::make_transform_iterator(indices, transform)};

  return input;
}

/** @brief Return the temporary storage size in bytes that CUB needs to sum
 *  @p count values from @p input.
 *  @details Calls CUB with null temporary storage to query its size. It does
 *  not run the reduction or read the input values.
 */
template <arithmetic T, typename Input> [[nodiscard]]
inline auto reduction_bytes(Input input, std::ptrdiff_t count) -> std::size_t {
  auto required_bytes{0uz};

  xpu::cu_check(cub::DeviceReduce::Sum(
    nullptr, required_bytes, input, static_cast<T*>(nullptr), count
  ));

  return required_bytes;
}

/** @brief Let CUB's one-dimensional bulk loop invoke a callable with
 *  multidimensional range indices.
 */
template <std::size_t dims, typename F>
struct parallel_for_function {
  xpu::range<dims> range;
  F fcn;

  /** @brief Convert @p linear to a range index and invoke fcn with it. */
  DEVICE_ONLY
  auto operator()(std::size_t linear) -> void {
    const auto index{xpu::detail::itr_index(range, linear)};

    fcn(index);
  }
};
#endif

/** @brief Execute @p fcn for the first @p total indices in @p range.
 *  @details Uses CUB on CUDA and an OpenMP loop on CPU.
 */
template <std::size_t dims, typename F>
inline auto parallel_for_impl(
  const xpu::range<dims>& range,
  std::size_t total,
  F&& fcn
) -> void {
#if defined(XPU_CUDA)
  using function_t = std::decay_t<F>;

  const auto operation = parallel_for_function<dims, function_t>{
    range,
    function_t{std::forward<F>(fcn)}
  };

  xpu::cu_check(cub::DeviceFor::Bulk(total, operation));
#else
  #pragma omp parallel for
  for (auto linear = 0uz; linear < total; ++linear) {
    fcn(xpu::detail::itr_index(range, linear));
  }
#endif
}

/** @brief Sum @p total contributions from @p range into @p output_ptr.
 *  @details CUDA uses caller-provided scratch storage and aborts if it is
 *  missing or too small. CPU uses an OpenMP reduction.
 */
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

  const auto transform = reduction_contribution<dims, T, function_t>{
    range, function_t{std::forward<F>(contribution)}
  };

#if defined(XPU_CUDA)
  const auto indices{thrust::counting_iterator<std::size_t>{0uz}};
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
  auto total_sum{T{}};

  #pragma omp parallel for reduction(+ : total_sum)
  for (auto linear = 0uz; linear < total; ++linear) {
    total_sum += transform(linear);
  }

  *output_ptr = total_sum;
#endif
}

} // namespace xpu::detail

/** @brief Invoke @p fcn once per index of @p range.
 *  @details An empty range invokes nothing. Calls may run concurrently and
 *  have no defined order.
 */
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

/** @brief Return the temporary storage size in bytes needed to sum
 *  @p contribution over @p range with parallel_reduce_sum().
 *  @details Returns zero for an empty range and on the CPU backend. This
 *  queries storage requirements without evaluating @p contribution.
 */
template <arithmetic T, std::size_t dims, typename F>
  requires detail::sum_contribution<F, T, dims>
[[nodiscard]]
inline auto parallel_reduce_sum_bytes(
  [[maybe_unused]] const xpu::range<dims>& range,
  [[maybe_unused]] F&& contribution
) -> std::size_t {
  static_assert(dims > 0uz, "ERROR: Dimension must be greater than 0.");

  auto required_bytes{0uz};

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

/** @brief Evaluate @p contribution at each index of @p range and write their
 *  sum to @p output_ptr.
 *  @details An empty range writes zero. On CUDA, pass scratch storage and its
 *  byte count from parallel_reduce_sum_bytes() for a nonempty range.
 *  Summation order is unspecified.
 */
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

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>
#include <xpu/config.hpp>
#include <xpu/detail/checked.hpp>
#include <xpu/math.hpp>

#if defined(_OPENMP)
  #include <omp.h>
#endif

// Loops with fewer iterations than this run on the calling thread (0: always parallel)
#ifndef XPU_PARALLEL_FOR_SERIAL_BELOW
  #define XPU_PARALLEL_FOR_SERIAL_BELOW 0uz
#endif

// OpenMP schedule for the CPU loops; dynamic balances hybrid P/E cores better
#ifndef XPU_PARALLEL_SCHEDULE
  #define XPU_PARALLEL_SCHEDULE static
#endif

#define XPU_STRINGIFY_VALUE(value) #value
#define XPU_STRINGIFY(value) XPU_STRINGIFY_VALUE(value)

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

inline auto retain_default_pool() -> void {
  static const auto once{[] {
      auto device{0};
      auto pool{cudaMemPool_t{}};
      auto threshold{std::numeric_limits<std::uint64_t>::max()};
      xpu::cu_check(cudaGetDevice(&device));
      xpu::cu_check(cudaDeviceGetDefaultMemPool(&pool, device));
      xpu::cu_check(cudaMemPoolSetAttribute(
        pool, cudaMemPoolAttrReleaseThreshold, &threshold
      ));

      return true;
    }()
  };

  static_cast<void>(once);
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

[[nodiscard]] XPU_DEVICE_ONLY
inline auto linear_index() noexcept -> std::size_t {
  return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

[[nodiscard]] XPU_DEVICE_ONLY
inline auto linear_stride() noexcept -> std::size_t {
  return static_cast<std::size_t>(gridDim.x) * blockDim.x;
}

template <int Dims> struct Coord;
template <> struct Coord<1> { std::size_t x{}; };
template <> struct Coord<2> { std::size_t x{}, y{}; };
template <> struct Coord<3> { std::size_t x{}, y{}, z{}; };

template <int Dims> __device__ [[nodiscard]]
inline auto global_index() noexcept -> Coord<Dims> {
  auto id{Coord<Dims>{}};
  
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
  auto stride{Coord<Dims>{}};
  
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
constexpr auto itr_counts(
  const xpu::range<dims>& range
) noexcept -> xpu::array<size_t, dims> {
  static_assert(dims > 0uz, "ERROR: Dimension must be greater than 0.");
  auto counts{xpu::array<std::size_t, dims>{}};

  for (auto d{0uz}; d < dims; ++d) {
    const auto empty_dimension{range.step[d] == 0uz || range.begin[d] >= range.end[d]};
    if (empty_dimension) {return xpu::array<size_t, dims>{};}
    const auto delta{range.end[d] - range.begin[d]};
    counts[d] = xpu::ceiling_div(delta, range.step[d]);
  }
  return counts;
}

template <std::size_t dims>
constexpr auto num_itrs(
  const xpu::range<dims>& range
) noexcept -> std::size_t {
  static_assert(dims > 0uz, "ERROR: Dimension must be greater than 0.");
  auto total{1uz};
  const auto counts{itr_counts(range)};

  for (auto d{0uz}; d < dims; ++d){
    total = checked_mul(total, counts[d]);
  }


  return total;
}

template <std::size_t dims> XPU_CUDA_CALLABLE
constexpr auto itr_index(
  const xpu::range<dims>& range,
  std::size_t linear,
  xpu::array<std::size_t, dims> counts
) noexcept -> xpu::array<std::size_t, dims> {
  static_assert(dims > 0uz, "ERROR: Dimension must be greater than 0.");
  auto index{xpu::array<std::size_t, dims>{}};

  for (auto d{dims}; d-- > 1uz;) {
    const auto linear_quotient{linear/counts[d]};
    const auto coordinate{linear - (linear_quotient * counts[d])};
    linear = linear_quotient;

    index[d] = range.begin[d] + coordinate * range.step[d];
  }

  index[0] = range.begin[0] + linear * range.step[0];

  return index;
}

template <typename F, typename T, std::size_t dims>
concept sum_contribution = requires(
  const std::remove_reference_t<F>& contribution,
  const xpu::array<std::size_t, dims>& index
) {
  { contribution(index) } -> std::same_as<T>;
};

inline constexpr auto narrow_itrs{std::size_t{1} << 32u};

// ceil(2^64 / count): the high word of linear * reciprocal is linear / count
// for linear and count below 2^32. A count of 1 has no such word and stores 0.
template <std::size_t dims>
constexpr auto itr_reciprocals(
  const xpu::array<std::size_t, dims>& counts
) noexcept -> xpu::array<std::uint64_t, dims> {
  auto reciprocals{xpu::array<std::uint64_t, dims>{}};

  for (auto d{0uz}; d < dims; ++d) {
    const auto has_reciprocal{counts[d] > 1uz};
    if (has_reciprocal) {
      reciprocals[d] = std::numeric_limits<std::uint64_t>::max() / counts[d] + 1u;
    }
  }

  return reciprocals;
}

[[nodiscard]] XPU_CUDA_CALLABLE
inline auto mul_high(std::uint64_t a, std::uint64_t b) noexcept -> std::uint64_t {
#if defined(__CUDA_ARCH__)
  return __umul64hi(a, b);
#else
  return static_cast<std::uint64_t>((static_cast<unsigned __int128>(a) * b) >> 64u);
#endif
}

template <std::size_t dims> XPU_CUDA_CALLABLE
inline auto itr_index_narrow(
  const xpu::range<dims>& range,
  std::size_t linear,
  const xpu::array<std::size_t, dims>& counts,
  const xpu::array<std::uint64_t, dims>& reciprocals
) noexcept -> xpu::array<std::size_t, dims> {
  auto index{xpu::array<std::size_t, dims>{}};

  for (auto d{dims}; d-- > 1uz;) {
    const auto has_reciprocal{reciprocals[d] != 0u};
    const auto linear_quotient{has_reciprocal ? mul_high(linear, reciprocals[d]) : linear};
    const auto coordinate{linear - (linear_quotient * counts[d])};
    linear = linear_quotient;

    index[d] = range.begin[d] + coordinate * range.step[d];
  }

  index[0] = range.begin[0] + linear * range.step[0];

  return index;
}

template <std::size_t dims, arithmetic T, typename F>
struct reduction_contribution {
  xpu::range<dims> range;
  F contribution;
  xpu::array<std::size_t, dims> counts;
  xpu::array<std::uint64_t, dims> reciprocals;
  bool narrow;

  [[nodiscard]] XPU_CUDA_CALLABLE
  auto operator()(std::size_t linear) const -> T {
    const auto index{
      narrow
        ? xpu::detail::itr_index_narrow(range, linear, counts, reciprocals)
        : xpu::detail::itr_index(range, linear, counts)
    };
    const auto value{T{contribution(index)}};

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

  const auto counts{itr_counts(range)};
  const auto transform{reduction_contribution<dims, T, function_t>{
    range, function_t{std::forward<F>(contribution)},
    counts, itr_reciprocals(counts), num_itrs(range) <= narrow_itrs
  }};

  const auto indices{thrust::counting_iterator<std::size_t>{0uz}};
  const auto input{thrust::make_transform_iterator(indices, transform)};

  return input;
}

template <std::size_t dims, typename F>
struct parallel_for_function {
  xpu::range<dims> range;
  F fcn;
  xpu::array<std::size_t, dims> counts;
  bool narrow;

  template <typename Digit> [[nodiscard]] XPU_DEVICE_ONLY
  auto digits(std::size_t linear) const -> xpu::array<Digit, dims> {
    auto position{xpu::array<Digit, dims>{}};
    auto rest{static_cast<Digit>(linear)};

    for (auto d{dims}; d-- > 1uz;) {
      const auto count{static_cast<Digit>(counts[d])};
      const auto quotient{rest / count};

      position[d] = rest - quotient * count;
      rest = quotient;
    }

    position[0] = rest;

    return position;
  }

  // The only divisions are the ones that split first and stride into
  // per-dimension digits; every later index is an add with carry.
  template <typename Digit> XPU_DEVICE_ONLY
  auto walk(std::size_t first, std::size_t stride, std::size_t total) const -> void {
    auto position{digits<Digit>(first)};
    const auto single{total - first <= stride};
    const auto advance{single ? xpu::array<Digit, dims>{} : digits<Digit>(stride)};

    for (auto linear{first};;) {
      auto index{xpu::array<std::size_t, dims>{}};

      for (auto d{0uz}; d < dims; ++d) {
        index[d] = range.begin[d] + position[d] * range.step[d];
      }

      fcn(index);

      const auto last{total - linear <= stride};
      if (last) { return; }

      linear += stride;
      auto carry{Digit{0}};

      for (auto d{dims}; d-- > 1uz;) {
        const auto room{static_cast<Digit>(counts[d]) - position[d]};
        const auto step{static_cast<Digit>(advance[d] + carry)};
        const auto wraps{step >= room};

        position[d] = wraps ? step - room : position[d] + step;
        carry = wraps ? Digit{1} : Digit{0};
      }

      position[0] += advance[0] + carry;
    }
  }

  XPU_DEVICE_ONLY
  auto operator()(std::size_t first, std::size_t stride, std::size_t total) const -> void {
    const auto idle{first >= total};
    if (idle) { return; }

    if constexpr (dims == 1uz) {
      for (auto linear{first};; linear += stride) {
        fcn(xpu::array<std::size_t, 1uz>{range.begin[0] + linear * range.step[0]});

        const auto last{total - linear <= stride};
        if (last) { return; }
      }
    } else if (narrow) {
      walk<std::uint32_t>(first, stride, total);
    } else {
      walk<std::size_t>(first, stride, total);
    }
  }
};

template <typename Op>
__global__ void parallel_for_kernel(const Op op, std::size_t total) {
  op(xpu::linear_index(), xpu::linear_stride(), total);
}


#endif

#if !defined(XPU_CUDA)
template <std::size_t dims>
struct box_split {
  std::size_t split;
  std::size_t block;
  std::size_t blocks;
  std::size_t boxes;
};

// boxes of ~grain iterations: dims before split fixed, split blocked, dims after split whole
template <std::size_t dims>
constexpr auto split_boxes(
  const xpu::array<std::size_t, dims>& counts,
  std::size_t total,
  std::size_t grain
) noexcept -> box_split<dims> {
  auto split{0uz}, outer{1uz}, inner{total / counts[0]};
  for (; inner > grain; inner /= counts[++split]) {
    outer *= counts[split];
  }

  const auto block{xpu::min(counts[split], xpu::ceiling_div(grain, inner))};
  const auto blocks{xpu::ceiling_div(counts[split], block)};

  return {split, block, blocks, outer * blocks};
}

template <std::size_t dims>
XPU_FORCE_INLINE auto select_box(
  const xpu::range<dims>& range,
  const xpu::array<std::size_t, dims>& counts,
  const box_split<dims>& boxes,
  std::size_t box,
  xpu::range<dims>& box_range,
  xpu::array<std::size_t, dims>& box_counts
) noexcept -> void {
  const auto split{boxes.split};
  const auto first{box % boxes.blocks * boxes.block};

  box_range.begin[split] += first * range.step[split];
  box_counts[split] = xpu::min(boxes.block, counts[split] - first);

  for (auto dimension{split}, rest{box / boxes.blocks}; dimension-- > 0uz;) {
    box_range.begin[dimension] += rest % counts[dimension] * range.step[dimension];
    box_counts[dimension] = 1uz;
    rest /= counts[dimension];
  }
}

inline constexpr auto static_schedule{
  std::string_view{XPU_STRINGIFY(XPU_PARALLEL_SCHEDULE)} == "static"
};

// several boxes per thread keep a static schedule balanced, but boxes stay
// long enough for the innermost simd loop to pay off
[[nodiscard]]
inline auto parallel_for_grain(std::size_t total) noexcept -> std::size_t {
  constexpr auto grain{8192uz};
  constexpr auto short_grain{256uz};
  constexpr auto boxes_per_thread{8uz};
#if defined(_OPENMP)
  const auto threads{static_cast<std::size_t>(omp_get_max_threads())};
#else
  const auto threads{1uz};
#endif

  const auto share{xpu::ceiling_div(total, threads)};

  return xpu::min(xpu::max(share / boxes_per_thread, xpu::min(share, short_grain)), grain);
}

template <bool unit_step, std::size_t dimension, std::size_t dims>
inline auto parallel_for_coordinate(
  const xpu::range<dims>& range,
  std::size_t iteration
) -> std::size_t {
  if constexpr (unit_step) {
    return range.begin[dimension] + iteration;
  } else {
    return range.begin[dimension] + iteration * range.step[dimension];
  }
}

template <bool unit_step, std::size_t dims, typename F, typename... Coordinates>
XPU_FORCE_INLINE auto parallel_for_nested(
  const xpu::range<dims>& range,
  const xpu::array<std::size_t, dims>& counts,
  F& fcn,
  Coordinates... outer
) -> void {
  constexpr auto dimension{sizeof...(Coordinates)};

  if constexpr (dimension == dims) {
    fcn(xpu::array<std::size_t, dims>{outer...});
  } else if constexpr (dimension + 1uz == dims) {
    #pragma omp simd
    for (auto iteration = 0uz; iteration < counts[dimension]; ++iteration) {
      parallel_for_nested<unit_step>(
        range, counts, fcn, outer..., parallel_for_coordinate<unit_step, dimension>(range, iteration)
      );
    }
  } else {
    for (auto iteration{0uz}; iteration < counts[dimension]; ++iteration) {
      parallel_for_nested<unit_step>(
        range, counts, fcn, outer..., parallel_for_coordinate<unit_step, dimension>(range, iteration)
      );
    }
  }
}

template <bool unit_step, std::size_t dims, typename F>
inline auto parallel_for_cpu(
  const xpu::range<dims>& range,
  const xpu::array<std::size_t, dims>& counts,
  std::size_t total,
  F& fcn
) -> void {
  if constexpr (!std::is_copy_constructible_v<F>) {
    auto shared{[&fcn](auto&& index) {
      fcn(std::forward<decltype(index)>(index));
    }};

    parallel_for_cpu<unit_step>(range, counts, total, shared);
  } else if constexpr (dims == 1uz && static_schedule) {
    // one long simd loop per thread beats boxes when nothing needs balancing
    const auto parallel{total >= XPU_PARALLEL_FOR_SERIAL_BELOW};

    #pragma omp parallel for simd schedule(static) firstprivate(fcn) if(parallel : parallel)
    for (auto iteration = 0uz; iteration < counts[0]; ++iteration) {
      fcn(xpu::array<std::size_t, 1uz>{parallel_for_coordinate<unit_step, 0uz>(range, iteration)});
    }
  } else {
    // parallelizing over boxes instead of the first dimension keeps every
    // thread busy when the first dimension is shorter than the thread count
    const auto boxes{split_boxes(counts, total, parallel_for_grain(total))};
    const auto parallel{total >= XPU_PARALLEL_FOR_SERIAL_BELOW};

    #pragma omp parallel for schedule(XPU_PARALLEL_SCHEDULE) firstprivate(fcn) if(parallel)
    for (auto box = 0uz; box < boxes.boxes; ++box) {
      auto box_range{range};
      auto box_counts{counts};

      select_box(range, counts, boxes, box, box_range, box_counts);
      parallel_for_nested<unit_step>(box_range, box_counts, fcn);
    }
  }
}
#endif

template <std::size_t dims, typename F>
inline auto parallel_for_impl(
  const xpu::range<dims>& range,
  F&& fcn
) -> void {
#if defined(XPU_CUDA)
  const auto total{xpu::detail::num_itrs(range)};
  if (total == 0uz) { return; }

  using function_t = std::decay_t<F>;
  using operation_t = parallel_for_function<dims, function_t>;

  const auto operation{operation_t{
    range,
    function_t{std::forward<F>(fcn)},
    itr_counts(range),
    total < narrow_itrs
  }};

  constexpr auto threads{256u};
  static const auto wave{xpu::detail::wave_blocks(parallel_for_kernel<operation_t>, threads)};

  const auto needed{xpu::ceiling_div<std::size_t>(total, threads)};
  const auto blocks{static_cast<unsigned int>(xpu::min(needed, static_cast<std::size_t>(wave)))};

  parallel_for_kernel<<<blocks, threads>>>(operation, total);
  xpu::cu_check(cudaGetLastError());
#else
  auto unit_step{true};
  auto counts{range.end};
  auto total{1uz};

  for (auto dimension{0uz}; dimension < dims; ++dimension) {
    const auto empty_dimension{range.step[dimension] == 0uz || range.begin[dimension] >= range.end[dimension]};
    if (empty_dimension) { return; }
  }

  for (auto dimension{0uz}; dimension < dims; ++dimension) {
    const auto delta{range.end[dimension] - range.begin[dimension]};

    unit_step = unit_step && range.step[dimension] == 1uz;
    counts[dimension] = xpu::ceiling_div(delta, range.step[dimension]);
    total = xpu::detail::checked_mul(total, counts[dimension]);
  }

  if (unit_step) {
    parallel_for_cpu<true>(range, counts, total, fcn);
  } else {
    parallel_for_cpu<false>(range, counts, total, fcn);
  }
#endif
}

#if !defined(XPU_CUDA)
template <bool unit_step, std::size_t dims, arithmetic T, typename F, typename... Coordinates>
XPU_FORCE_INLINE auto parallel_reduce_nested(
  const xpu::range<dims>& range,
  const xpu::array<std::size_t, dims>& counts,
  const F& contribution,
  T& total_sum,
  Coordinates... outer
) -> void {
  constexpr auto dimension{sizeof...(Coordinates)};

  if constexpr (dimension + 1uz == dims) {
    // GCC won't vectorize an omp simd body that constructs the index array itself
    const auto step{unit_step ? 1uz : range.step[dimension]};
    const auto contribution_at{[&](std::size_t iteration) {
      return contribution(xpu::array<std::size_t, dims>{
        outer..., range.begin[dimension] + iteration * step
      });
    }};

    #pragma omp simd reduction(+ : total_sum)
    for (auto iteration = 0uz; iteration < counts[dimension]; ++iteration) {
      total_sum += contribution_at(iteration);
    }
  } else {
    for (auto iteration{0uz}; iteration < counts[dimension]; ++iteration) {
      parallel_reduce_nested<unit_step>(
        range, counts, contribution, total_sum, outer...,
        parallel_for_coordinate<false, dimension>(range, iteration)
      );
    }
  }
}

template <bool unit_step, arithmetic T, std::size_t dims, typename F> [[nodiscard]]
inline auto parallel_reduce_cpu(
  const xpu::range<dims>& range,
  std::size_t total,
  const F& contribution
) -> T {
  constexpr auto grain{8192uz};
  auto counts{range.end};

  for (auto dimension{0uz}; dimension < dims; ++dimension) {
    const auto delta{range.end[dimension] - range.begin[dimension]};
    counts[dimension] = xpu::ceiling_div(delta, range.step[dimension]);
  }

  const auto boxes{split_boxes(counts, total, grain)};
  const auto parallel{boxes.boxes > 1uz};
  auto total_sum{T{}};

  #pragma omp parallel for schedule(XPU_PARALLEL_SCHEDULE) reduction(+ : total_sum) firstprivate(contribution) if(parallel)
  for (auto box = 0uz; box < boxes.boxes; ++box) {
    auto box_range{range};
    auto box_counts{counts};

    select_box(range, counts, boxes, box, box_range, box_counts);
    parallel_reduce_nested<unit_step>(box_range, box_counts, contribution, total_sum);
  }

  return total_sum;
}
#endif

template <std::size_t dims, arithmetic T, typename F>
inline auto parallel_reduce_sum_impl(
  const xpu::range<dims>& range,
  std::size_t total,
  T* output_ptr,
  F&& contribution
) -> void {
#if defined(XPU_CUDA)
  const auto input{reduction_input<T>(range, std::forward<F>(contribution))};
  const auto count{xpu::detail::checked_cast<std::ptrdiff_t>(total)};

  xpu::detail::retain_default_pool();
  xpu::cu_check(cub::DeviceReduce::Sum(input, output_ptr, count));
#else
  if (range.step[dims - 1uz] == 1uz) {
    *output_ptr = parallel_reduce_cpu<true, T>(range, total, contribution);
  } else {
    *output_ptr = parallel_reduce_cpu<false, T>(range, total, contribution);
  }
#endif
}

} // namespace xpu::detail

template <std::size_t dims, typename F>
inline auto parallel_for(
  const xpu::range<dims>& range,
  F&& fcn
) -> void {
  static_assert(dims > 0uz, "ERROR: Dimension must be greater than 0.");

  detail::parallel_for_impl(range, std::forward<F>(fcn));
}

template <std::size_t dims, arithmetic T, typename F>
  requires detail::sum_contribution<F, T, dims>
inline auto parallel_reduce_sum(
  const xpu::range<dims>& range,
  T* output_ptr,
  F&& contribution
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
    range, total, output_ptr, std::forward<F>(contribution)
  );
}

} // namespace xpu

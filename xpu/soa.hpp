#pragma once

#include <xpu/buffer.hpp>
#include <xpu/config.hpp>
#include <xpu/detail/checked.hpp>
#include <xpu/memory.hpp>

#if defined(XPU_CUDA)
  #include <cuda/std/array>
#else
  #include <array>
#endif

#include <cstddef>
#include <cassert>

namespace xpu {

using xstd::array;

template <typename T, std::size_t exposed_arrays>
class soa_view {
  static_assert(exposed_arrays > 0, "ERROR: number of arrays must be greater than zero");

private:
  T* base_;
  std::size_t count_;

public:
  CUDA_CALLABLE
  explicit constexpr soa_view(T* base, std::size_t count) noexcept
    : base_{base}
    , count_{count}
  {
    const auto storage_count{xpu::detail::checked_mul(exposed_arrays, stride())};

    static_cast<void>(xpu::detail::checked_bytes<T>(storage_count));
  }

  [[nodiscard]] CUDA_CALLABLE
  constexpr auto count() const noexcept -> std::size_t {
    return count_;
  }

  [[nodiscard]] CUDA_CALLABLE
  constexpr auto stride() const noexcept -> std::size_t {
    return xpu::handle_pad<T>(count_);
  }

  [[nodiscard]] CUDA_CALLABLE
  auto operator[](std::size_t arr_idx) noexcept -> T* {
    assert(arr_idx < exposed_arrays);
    return xpu::assume_aligned<T>(base_ + arr_idx * stride());
  }

  [[nodiscard]] CUDA_CALLABLE
  auto operator[](std::size_t arr_idx) const noexcept -> const T* {
    assert(arr_idx < exposed_arrays);
    return xpu::assume_aligned<T>(base_ + arr_idx * stride());
  }

  [[nodiscard]] CUDA_CALLABLE
  auto pointers() noexcept -> xpu::array<T*, exposed_arrays> {
    auto ptrs = xpu::array<T*, exposed_arrays>{};

    for (auto i{0uz}; i < exposed_arrays; ++i) {
      ptrs[i] = xpu::assume_aligned<T>(base_ + i * stride());
    }

    return ptrs;
  }

  [[nodiscard]] CUDA_CALLABLE
  auto pointers() const noexcept -> xpu::array<const T*, exposed_arrays> {
    auto ptrs = xpu::array<const T*, exposed_arrays>{};

    for (auto i{0uz}; i < exposed_arrays; ++i) {
      ptrs[i] = xpu::assume_aligned<T>(base_ + i * stride());
    }

    return ptrs;
  }
};

template <typename T, std::size_t exposed_arrays>
class soa_batch_view {
  static_assert(exposed_arrays > 0, "ERROR: number of arrays must be greater than zero");

private:
  T* base_;
  std::size_t batches_;
  std::size_t count_;
  std::size_t batch_stride_;

public:
  CUDA_CALLABLE
  explicit constexpr soa_batch_view(
    T* base,
    std::size_t batches,
    std::size_t count,
    std::size_t batch_stride
  ) noexcept
    : base_{base}
    , batches_{batches}
    , count_{count}
    , batch_stride_{batch_stride}
  { }

  [[nodiscard]] CUDA_CALLABLE
  constexpr auto batch_count() const noexcept -> std::size_t {

    return batches_;
  }

  [[nodiscard]] CUDA_CALLABLE
  constexpr auto element_count() const noexcept -> std::size_t {

    return count_;
  }

  [[nodiscard]] CUDA_CALLABLE
  constexpr auto array_stride() const noexcept -> std::size_t {
    const auto stride{xpu::handle_pad<T>(count_)};

    return stride;
  }

  [[nodiscard]] CUDA_CALLABLE
  constexpr auto batch_stride() const noexcept -> std::size_t {

    return batch_stride_;
  }

  [[nodiscard]] CUDA_CALLABLE
  auto view(std::size_t batch) noexcept -> xpu::soa_view<T, exposed_arrays> {
    assert(batch < batches_);

    auto* base{base_ + batch * batch_stride_};
    const auto arrays{xpu::soa_view<T, exposed_arrays>{base, count_}};

    return arrays;
  }

  [[nodiscard]] CUDA_CALLABLE
  auto view(std::size_t batch) const noexcept -> xpu::soa_view<const T, exposed_arrays> {
    assert(batch < batches_);

    const auto* base{base_ + batch * batch_stride_};
    const auto arrays{xpu::soa_view<const T, exposed_arrays>{base, count_}};

    return arrays;
  }
};

template <typename T, std::size_t num_arrays>
class soa {
  static_assert(num_arrays > 0, "ERROR: number of arrays must be greater than zero");

private:
  std::size_t count_;
  xpu::buffer<T> buffer_;

public:
  explicit soa(std::size_t count) noexcept
    : count_{count}
    , buffer_{
        xpu::detail::checked_mul(
          num_arrays,
          xpu::checked_padding<T>(count)
        )
      }
  { }

  [[nodiscard]]
  constexpr auto count() const noexcept -> std::size_t {
    return count_;
  }

  [[nodiscard]]
  constexpr auto stride() const noexcept -> std::size_t {
    return xpu::handle_pad<T>(count_);
  }

  [[nodiscard]]
  constexpr auto storage_size() const noexcept -> std::size_t {
    return num_arrays * stride();
  }

  [[nodiscard]]
  auto operator[](std::size_t arr_idx) noexcept -> T* {
    assert(arr_idx < num_arrays);
    return xpu::assume_aligned<T>(buffer_.data() + arr_idx * stride());
  }

  [[nodiscard]]
  auto operator[](std::size_t arr_idx) const noexcept -> const T* {
    assert(arr_idx < num_arrays);
    return xpu::assume_aligned<const T>(buffer_.data() + arr_idx * stride());
  }

  template <std::size_t exposed_arrays = num_arrays, std::size_t first_array = 0uz> [[nodiscard]]
  auto view() noexcept -> soa_view<T, exposed_arrays> {
    static_assert(xpu::detail::checked_add(first_array, exposed_arrays) <= num_arrays);
    return soa_view<T, exposed_arrays>{buffer_.data() + first_array * stride(), count_};
  }

  template <std::size_t exposed_arrays = num_arrays, std::size_t first_array = 0uz> [[nodiscard]]
  auto view() const noexcept -> soa_view<const T, exposed_arrays> {
    static_assert(xpu::detail::checked_add(first_array, exposed_arrays) <= num_arrays);
    return soa_view<const T, exposed_arrays>{buffer_.data() + first_array * stride(), count_};
  }
};

template <typename T, std::size_t num_arrays>
class soa_batch {
  static_assert(num_arrays > 0, "ERROR: number of arrays must be greater than zero");

private:
  std::size_t batches_;
  std::size_t count_;
  buffer<T> storage_;

public:
  explicit soa_batch(std::size_t batches, std::size_t count) noexcept
    : batches_{batches}
    , count_{count}
    , storage_{
        xpu::detail::checked_mul(
          batches,
          xpu::detail::checked_mul(
            num_arrays,
            xpu::checked_padding<T>(count)
          )
        )
      }
  { }

  [[nodiscard]] CUDA_CALLABLE
  constexpr auto batch_count() const noexcept -> std::size_t {
    return batches_;
  }

  [[nodiscard]] CUDA_CALLABLE
  constexpr auto element_count() const noexcept -> std::size_t {
    return count_;
  }

  [[nodiscard]] CUDA_CALLABLE
  constexpr auto logical_count() const noexcept -> std::size_t {
    return num_arrays * element_count() * batch_count();
  }

  [[nodiscard]] CUDA_CALLABLE
  constexpr auto array_stride() const noexcept -> std::size_t {
    return xpu::handle_pad<T>(count_);
  }

  [[nodiscard]] CUDA_CALLABLE
  constexpr auto batch_stride() const noexcept -> std::size_t {
    return num_arrays * array_stride();
  }

  [[nodiscard]] CUDA_CALLABLE
  constexpr auto storage_size() const noexcept -> std::size_t {
    return batch_stride() * batch_count();
  }

  template <std::size_t exposed_arrays = num_arrays, std::size_t first_array = 0uz> [[nodiscard]]
  auto view() noexcept -> xpu::soa_batch_view<T, exposed_arrays> {
    constexpr auto viewed_end{xpu::detail::checked_add(first_array, exposed_arrays)};
    constexpr auto valid_arrays{viewed_end <= num_arrays};

    static_assert(valid_arrays, "ERROR: number of viewed arrays is too large");

    auto* base{storage_.data() + first_array * array_stride()};
    const auto batches{xpu::soa_batch_view<T, exposed_arrays>{
      base, batches_, count_, batch_stride()
    }};

    return batches;
  }

  template <std::size_t exposed_arrays = num_arrays, std::size_t first_array = 0uz> [[nodiscard]]
  auto view() const noexcept -> xpu::soa_batch_view<const T, exposed_arrays> {
    constexpr auto viewed_end{xpu::detail::checked_add(first_array, exposed_arrays)};
    constexpr auto valid_arrays{viewed_end <= num_arrays};

    static_assert(valid_arrays, "ERROR: number of viewed arrays is too large");

    const auto* base{storage_.data() + first_array * array_stride()};
    const auto batches{xpu::soa_batch_view<const T, exposed_arrays>{
      base, batches_, count_, batch_stride()
    }};

    return batches;
  }

  template <std::size_t exposed_arrays = num_arrays, std::size_t first_array = 0uz> [[nodiscard]]
  auto view(std::size_t batch) noexcept -> xpu::soa_view<T, exposed_arrays> {
    static_assert(exposed_arrays <= num_arrays, "ERROR: exposed arrays is greater than number of arrays");
    static_assert(xpu::detail::checked_add(first_array, exposed_arrays) <= num_arrays, "ERROR: number of viewed arrays is too large");
    assert(batch < batches_);

    auto* base{storage_.data() + batch * batch_stride() + first_array * array_stride()};
    return xpu::soa_view<T, exposed_arrays>{
      base, element_count()
    };
  }

  template <std::size_t exposed_arrays = num_arrays, std::size_t first_array = 0uz> [[nodiscard]]
  auto view(std::size_t batch) const noexcept -> xpu::soa_view<const T, exposed_arrays> {
    static_assert(exposed_arrays <= num_arrays, "ERROR: exposed arrays is greater than number of arrays");
    static_assert(xpu::detail::checked_add(first_array, exposed_arrays) <= num_arrays, "ERROR: number of viewed arrays is too large");
    assert(batch < batches_);

    const auto* base{storage_.data() + batch * batch_stride() + first_array * array_stride()};
    return xpu::soa_view<const T, exposed_arrays>{
      base, element_count()
    };
  }
};

} // namespace xpu

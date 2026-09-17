#pragma once

#include <type_traits>
#include <xpu/array.hpp>
#include <xpu/buffer.hpp>
#include <xpu/config.hpp>
#include <xpu/detail/checked.hpp>
#include <xpu/memory.hpp>

#include <cstddef>
#include <cassert>

namespace xpu {

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
  static_cast<void>(bytes<T>(storage_count));
  }

  [[nodiscard]] CUDA_CALLABLE
  constexpr auto count() const noexcept -> std::size_t {
    return count_;
  }

  [[nodiscard]] CUDA_CALLABLE
  constexpr auto stride() const noexcept -> std::size_t {
    return xpu::handle_pad<T>(count_);
  }

  template <typename Self> [[nodiscard]] CUDA_CALLABLE
  constexpr auto operator[](this Self&& self, std::size_t arr_idx) noexcept {
    assert(arr_idx < exposed_arrays);

    using element_t = std::conditional_t<
      std::is_const_v<std::remove_reference_t<Self>>,
      const T, T
    >;

    return xpu::assume_aligned<element_t>(
      self.base_ + (arr_idx * self.stride())
    );
  }

  template <typename Self> [[nodiscard]] CUDA_CALLABLE
  constexpr auto pointers(this Self&& self) noexcept {
    auto ptrs{xpu::array<decltype(self[0uz]), exposed_arrays>{}};

    for (auto i{0uz}; i < exposed_arrays; ++i) {
      ptrs[i] = self[i];
    }

    return ptrs;
  }
};

template <typename T, std::size_t exposed_arrays, std::size_t storage_arrays = exposed_arrays>
class soa_batch_view {
  static_assert(exposed_arrays > 0, "ERROR: number of arrays must be greater than zero");
  static_assert(exposed_arrays <= storage_arrays, "ERROR: exposed arrays exceeds storage arrays");

private:
  T* base_;
  std::size_t batches_;
  std::size_t count_;

public:
  CUDA_CALLABLE
  explicit constexpr soa_batch_view(
    T* base,
    std::size_t batches,
    std::size_t count
  ) noexcept
    : base_{base}
    , batches_{batches}
    , count_{count}
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
    return xpu::handle_pad<T>(count_);
  }

  [[nodiscard]] CUDA_CALLABLE
  constexpr auto batch_stride() const noexcept -> std::size_t {
    return storage_arrays * array_stride();
  }

  template <typename Self> [[nodiscard]] CUDA_CALLABLE
  constexpr auto view(this Self&& self, std::size_t batch) noexcept {
    assert(batch < self.batches_);

    using element_t = std::conditional_t<
      std::is_const_v<std::remove_reference_t<Self>>,
      const T, T
    >;

    return xpu::soa_view<element_t, exposed_arrays>{
      self.base_ + (batch * self.batch_stride()), self.count_
    };
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
          xpu::handle_pad<T>(count)
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

  template <typename Self> [[nodiscard]]
  auto operator[](this Self&& self, std::size_t arr_idx) noexcept {
    assert(arr_idx < num_arrays);

    using element_t = std::conditional_t<
      std::is_const_v<std::remove_reference_t<Self>>,
      const T, T
    >;

    return xpu::assume_aligned<element_t>(
      self.buffer_.data() + (arr_idx * self.stride())
    );
  }

  template <
    std::size_t exposed_arrays = num_arrays,
    std::size_t first_array = 0uz,
    typename Self
  > [[nodiscard]]
  auto view(this Self&& self) noexcept {
    static_assert(xpu::detail::checked_add(first_array, exposed_arrays) <= num_arrays);

    using element_t = std::conditional_t<
      std::is_const_v<std::remove_reference_t<Self>>,
      const T, T
    >;
    
    return soa_view<element_t, exposed_arrays>{
      self.buffer_.data() + (first_array * self.stride()), self.count_
    };
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
            xpu::handle_pad<T>(count)
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

  template <
    std::size_t exposed_arrays = num_arrays,
    std::size_t first_array = 0uz,
    typename Self
  > [[nodiscard]]
  auto view(this Self&& self) noexcept {
    static_assert(
      xpu::detail::checked_add(first_array, exposed_arrays) <= num_arrays,
      "ERROR: number of viewed arrays is too large"
    );

    using element_t = std::conditional_t<
      std::is_const_v<std::remove_reference_t<Self>>,
      const T, T
    >;

    return xpu::soa_batch_view<element_t, exposed_arrays, num_arrays>{
      self.storage_.data() + (first_array * self.array_stride()),
      self.batches_, self.count_
    };
  }

  template <
    std::size_t exposed_arrays = num_arrays,
    std::size_t first_array = 0uz,
    typename Self
  > [[nodiscard]]
  auto view(this Self&& self, std::size_t batch) noexcept {
    static_assert(exposed_arrays <= num_arrays, "ERROR: exposed arrays is greater than number of arrays");
    static_assert(xpu::detail::checked_add(first_array, exposed_arrays) <= num_arrays,
      "ERROR: number of viewed arrays is too large");
    assert(batch < self.batches_);

    using element_t = std::conditional_t<
      std::is_const_v<std::remove_reference_t<Self>>,
      const T, T
    >;

    return xpu::soa_view<element_t, exposed_arrays>{
      self.storage_.data() + (batch * self.batch_stride()) + (first_array * self.array_stride()),
      self.element_count()
    };
  }
};

} // namespace xpu

#pragma once

#include <cassert>
#include <cstddef>
#include <iostream>
#include <xpu/config.hpp>
#include <xpu/memory.hpp>

#include <type_traits>

namespace xpu {

/** @brief Access a contiguous sequence without owning its storage.
 *  @note The underlying allocation must outlive the view.
 */
template <typename T>
class buffer_view {
private:
  T* data_;
  std::size_t count_;

public:
  /** @brief View @p count elements starting at @p base. */
  CUDA_CALLABLE
  explicit constexpr buffer_view(T* base, std::size_t count) noexcept
    : data_{base}
    , count_{count}
  { }

  /** @brief Return the logical number of elements. */
  [[nodiscard]] CUDA_CALLABLE
  constexpr auto count() const noexcept -> std::size_t {
    return count_;
  }

  /** @brief Access element @p idx with constness inherited from the view.
   *  @pre @p idx is less than count().
   */
  template <typename Self> [[nodiscard]] CUDA_CALLABLE
  constexpr auto& operator[](this Self&& self, std::size_t idx) noexcept {
    assert(idx < self.count_);

    using element_t = std::conditional_t<
      std::is_const_v<std::remove_reference_t<Self>>,
      const T, T
    >;

    return static_cast<element_t*>(self.data_)[idx];
  }
};

/** @brief Own a contiguous sequence in aligned host or CUDA device memory.
 *  @details Padding extends the allocation, not the logical element count.
 */
template <typename T>
class buffer {
private:
  std::size_t count_;
  xpu::unique_ptr<T> data_;

public:
  /** @brief Allocate padded storage for @p count elements and initialize it
   *  with T{}.
   */
  explicit buffer(std::size_t count)
    : count_{count}
    , data_{xpu::make_unique<T>(xpu::handle_pad<T>(count))}
  { }

  /** @brief Return the logical number of elements. */
  [[nodiscard]]
  constexpr auto count() const noexcept -> std::size_t {
    return count_;
  }

  /** @brief Return the number of elements allocated, including padding. */
  [[nodiscard]]
  constexpr auto capacity() const noexcept -> std::size_t {
    return xpu::handle_pad<T>(count_);
  }

  /** @brief Access the allocation; the pointer is to host or device memory
   *  according to the active backend.
   */
  template <typename Self> [[nodiscard]]
  auto data(this Self&& self) noexcept {
    using element_t = std::conditional_t<
      std::is_const_v<std::remove_reference_t<Self>>,
      const T, T
    >;

    return xpu::assume_aligned<element_t>(self.data_.get());
  }

  /** @brief Borrow the logical elements without transferring ownership. */
  template <typename Self> [[nodiscard]]
  auto view(this Self&& self) noexcept {
    using element_t = std::conditional_t<
      std::is_const_v<std::remove_reference_t<Self>>,
      const T, T
    >;

    return xpu::buffer_view<element_t>{
      self.data(), self.count_
    };
  }
};

} // namespace xpu

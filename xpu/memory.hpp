#pragma once

#include <xpu/algorithm.hpp>
#include <xpu/config.hpp>
#include <xpu/detail/checked.hpp>
#include <cstring>
#include <memory>
#include <type_traits>

namespace xpu {

/** @brief The greater of alignof(T) and the active backend's alignment. */
template <typename T>
inline constexpr auto default_align{(xpu::alignment_bytes > alignof(T)) ? xpu::alignment_bytes : alignof(T)};

/** @brief True when sizeof(T) is smaller than the active backend's alignment. */
template <typename T>
inline constexpr auto is_padded{sizeof(T) < xpu::alignment_bytes};

/** @brief Return the byte size of @p count objects, aborting on overflow. */
template <typename T> [[nodiscard]] CUDA_CALLABLE
inline constexpr auto bytes(std::size_t count) noexcept -> std::size_t {
  static_assert(std::is_trivially_copyable_v<T>, "ERROR: xpu::bytes requires a trivially copyable type");
  const auto byte_count{xpu::detail::checked_bytes<T>(count)};

  return byte_count;
}

/** @brief Round @p unpadded up to a multiple of alignment_bytes / sizeof(T)
 *  when T is smaller than the alignment; otherwise return it unchanged.
 *  @details Aborts if rounding overflows std::size_t.
 */
template <typename T> [[nodiscard]] CUDA_CALLABLE
inline constexpr auto handle_pad(std::size_t unpadded) noexcept -> std::size_t {
  if constexpr (is_padded<T>) {
    constexpr auto lanes{xpu::alignment_bytes / sizeof(T)};
    const auto padded_count{xpu::detail::checked_round_up(unpadded, lanes)};

    return padded_count;
  } else {
    return unpadded;
  }
}

/** @brief Allocate aligned host or CUDA device storage for @p count elements
 *  of T without initializing them.
 *  @return The allocated pointer, or nullptr when @p count is zero.
 *  @details Aborts on byte-count overflow or allocation failure.
 */
template <typename T> [[nodiscard]]
inline auto alloc(std::size_t count) -> T* {
  static_assert(std::is_trivially_copyable_v<T>);
  if (count == 0uz) { return nullptr; }

  const auto bytes{xpu::detail::checked_bytes<T>(count)};

#if defined(XPU_CUDA)
  auto ptr{static_cast<void*>(nullptr)};
  if(cudaMalloc(&ptr, bytes) != cudaSuccess) { ptr = nullptr; }
#else
  auto ptr{::operator new(bytes, std::align_val_t{default_align<T>}, std::nothrow)};
#endif

  if (!ptr) {
    std::fprintf(
      stderr,
      "xpu: failed to allocate %zu bytes\n",
      bytes
    );
    std::abort();
  }
  
  return static_cast<T*>(ptr);
}

/** @brief Release memory obtained from alloc(); a null pointer is ignored. */
template <typename T>
inline auto free(T* ptr) noexcept -> void {
  if (!ptr) { return; }

#if defined(XPU_CUDA)
  cudaFree(ptr);
#else
  ::operator delete(ptr, std::align_val_t(default_align<T>));
#endif
}

/** @brief Deleter for memory obtained from alloc(). */
template <typename T>
struct deleter {
  auto operator()(T* ptr) const noexcept -> void {
    xpu::free(ptr);
  }
};

/** @brief Owning pointer for an XPU allocation. */
template <typename T>
using unique_ptr = std::unique_ptr<T[], xpu::deleter<T>>;

/** @brief Allocate aligned storage for @p count elements and fill each with
 *  @p value.
 */
template <typename T>
auto make_unique(std::size_t count, T value = T{}) -> unique_ptr<T> {
  auto* RESTRICT ptr{xpu::alloc<T>(count)};

  xpu::fill_n(ptr, count, value);

  return xpu::unique_ptr<T>{ptr};
}

/** @brief Convey the allocation alignment of @p ptr to the compiler.
 *  @pre @p ptr has the alignment returned by default_align<T> when padding applies.
 */
template <typename T> [[nodiscard]] CUDA_CALLABLE
constexpr auto assume_aligned(T* ptr) noexcept -> T* {
  if constexpr (is_padded<T>) {
    return std::assume_aligned<default_align<T>>(ptr);
  } else {
    return ptr;
  }
}

/** @brief Set the first @p bytes bytes at @p dst to the byte value @p value. */
inline auto memset(
  void* RESTRICT dst,
  int value,
  std::size_t bytes
) noexcept -> void {
  if (bytes == 0uz) { return; }

#if defined(XPU_CUDA)
  xpu::cu_check(cudaMemset(dst, value, bytes));
#else
  std::memset(dst, value, bytes);
#endif
}

/** @brief Copy @p bytes bytes from @p src to @p dst.
 *  @details CUDA infers the transfer direction from the pointers.
 *  @pre Source and destination ranges do not overlap.
 */
inline auto memcpy(
  void* RESTRICT dst,
  const void* RESTRICT src,
  std::size_t bytes
) noexcept -> void {
  if (bytes == 0uz) { return; }

#if defined(XPU_CUDA)
  xpu::cu_check(cudaMemcpy(dst, src, bytes, cudaMemcpyDefault));
#else
  std::memcpy(dst, src, bytes);
#endif
}

/** @brief Copy @p count trivially copyable elements from @p src to @p dst. */
template <typename T>
inline auto copy_n(
  T* RESTRICT dst,
  const T* RESTRICT src,
  std::size_t count
) noexcept -> void {
  static_assert(
    std::is_trivially_copyable_v<T>,
    "ERROR: xpu::copy_n requires trivially copyable type"
  );

  const auto byte_count{xpu::detail::checked_bytes<T>(count)};

  xpu::memcpy(dst, src, byte_count);
}

/** @brief Zero the object representation of @p count arithmetic elements. */
template <typename T>
inline auto zero_n(
  T* RESTRICT dst,
  std::size_t count
) noexcept -> void {
  static_assert(
    std::is_arithmetic_v<T>,
    "ERROR: xpu::zero_n requires arithmetic type"
  );

  const auto byte_count{xpu::detail::checked_bytes<T>(count)};

  xpu::memset(dst, 0, byte_count);
}

} // namespace xpu

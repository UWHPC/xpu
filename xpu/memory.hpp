#pragma once

#include <print>
#include <xpu/algorithm.hpp>
#include <xpu/config.hpp>
#include <xpu/detail/checked.hpp>
#include <cstring>
#include <memory>
#include <type_traits>

namespace xpu {

template <typename T>
constexpr auto default_align{(xpu::alignment_bytes > alignof(T)) ? xpu::alignment_bytes : alignof(T)};

template <typename T>
constexpr auto is_padded{sizeof(T) < xpu::alignment_bytes};

template <typename T> [[nodiscard]] CUDA_CALLABLE
 constexpr auto bytes(std::size_t count) noexcept -> std::size_t {
  static_assert(std::is_trivially_copyable_v<T>, "ERROR: xpu::bytes requires a trivially copyable type");
  const auto byte_count{xpu::detail::checked_bytes<T>(count)};

  return byte_count;
}

template <typename T> [[nodiscard]] CUDA_CALLABLE
 constexpr auto handle_pad(std::size_t unpadded) noexcept -> std::size_t {
  if constexpr (is_padded<T>) {
    constexpr auto lanes{xpu::alignment_bytes / sizeof(T)};
    const auto padded_count{xpu::detail::checked_round_up(unpadded, lanes)};

    return padded_count;
  } else {
    return unpadded;
  }
}

template <typename T> [[nodiscard]]
auto alloc(std::size_t count) -> T* {
  static_assert(std::is_trivially_copyable_v<T>);
  if (count == 0uz) { return nullptr; }

#if defined(XPU_CUDA)
  auto ptr{static_cast<void*>(nullptr)};
  if(cudaMalloc(&ptr, bytes<T>(count)) != cudaSuccess) { ptr = nullptr; }
#else
  auto ptr{::operator new(bytes<T>(count), std::align_val_t{default_align<T>}, std::nothrow)};
#endif

  if (!ptr) {
    std::fprintf(
      stderr,
      "xpu: failed to allocate {:d} bytes",
      bytes<T>(count)
    );
    std::abort();
  }
  
  return static_cast<T*>(ptr);
}

template <typename T>
auto free(T* ptr) noexcept -> void {
  if (!ptr) { return; }

#if defined(XPU_CUDA)
  cudaFree(ptr);
#else
  ::operator delete(ptr, std::align_val_t(default_align<T>));
#endif
}

template <typename T>
struct deleter {
  auto operator()(T* ptr) const noexcept -> void {
    xpu::free(ptr);
  }
};

template <typename T>
using unique_ptr = std::unique_ptr<T[], xpu::deleter<T>>;

template <typename T>
auto make_unique(T value = T{}) -> unique_ptr<T> {
  return xpu::make_unique<T>(1uz, std::move(value));
}

template <typename T>
auto make_unique(std::size_t count, T value = T{}) -> unique_ptr<T> {
  auto* RESTRICT ptr{xpu::alloc<T>(count)};

  xpu::fill_n(ptr, count, value);

  return xpu::unique_ptr<T>{ptr};
}

template <typename T> [[nodiscard]] CUDA_CALLABLE
constexpr auto assume_aligned(T* ptr) noexcept -> T* {
  if constexpr (!is_padded<T>) { return ptr; }
  return std::assume_aligned<default_align<T>>(ptr);
}


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

template <typename T>
auto copy_n(
  T* RESTRICT dst,
  const T* RESTRICT src,
  std::size_t count
) noexcept -> void {
  static_assert(
    std::is_trivially_copyable_v<T>,
    "ERROR: xpu::copy_n requires trivially copyable type"
  );

  xpu::memcpy(dst, src, bytes<T>(count));
}

template <typename T>
auto zero_n(
  T* RESTRICT dst,
  std::size_t count
) noexcept -> void {
  static_assert(
    std::is_arithmetic_v<T>,
    "ERROR: xpu::zero_n requires arithmetic type"
  );

  xpu::memset(dst, 0, bytes<T>(count));
}

} // namespace xpu

#pragma once

#include <xpu/algorithm.hpp>
#include <xpu/config.hpp>
#include <xpu/detail/checked.hpp>
#include <cstring>
#include <memory>
#include <type_traits>

namespace xpu {

template <typename T>
inline constexpr auto default_align{(xpu::simd_bytes > alignof(T)) ? xpu::simd_bytes : alignof(T)};

template <typename T>
inline constexpr auto is_padded{!xpu::xpu_cuda && sizeof(T) < xpu::simd_bytes};

template <typename T> [[nodiscard]] CUDA_CALLABLE
inline constexpr auto bytes(std::size_t count) noexcept -> std::size_t {
  static_assert(std::is_trivially_copyable_v<T>, "ERROR: xpu::bytes requires a trivially copyable type");
  const auto byte_count{xpu::detail::checked_bytes<T>(count)};

  return byte_count;
}

template <typename T> [[nodiscard]] CUDA_CALLABLE
inline constexpr auto handle_pad(std::size_t unpadded) noexcept -> std::size_t {
  if constexpr (is_padded<T>) {
    constexpr auto lanes{xpu::simd_bytes / sizeof(T)};
    const auto padded_count{xpu::detail::checked_round_up(unpadded, lanes)};

    return padded_count;
  } else {
    return unpadded;
  }
}

template <typename T> [[nodiscard]]
constexpr auto checked_padding(std::size_t count) noexcept -> std::size_t {
  auto padded_count{count};

  if constexpr (is_padded<T>) {
    constexpr auto lanes{xpu::simd_bytes / sizeof(T)};

    padded_count = xpu::detail::checked_round_up(count, lanes);
  }

  return padded_count;
}

template <typename T> [[nodiscard]]
inline auto alloc(std::size_t count) -> T* {
  static_assert(std::is_trivially_copyable_v<T>);
  if (count == 0u) { return nullptr; }

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

template <typename T>
inline auto free(void* ptr) noexcept -> void {
#if defined(XPU_CUDA)
  cudaFree(ptr);
#else
  ::operator delete(ptr, std::align_val_t(default_align<T>));
#endif
}

struct deleter {
  template <typename T>
  auto operator()(T* ptr) const noexcept -> void {
    xpu::free<T>(ptr);
  }
};

template <typename T>
class unique_ptr {
private:
  std::unique_ptr<T[], deleter> data_;

public:
  unique_ptr() noexcept
    : data_{}
  { }

  explicit unique_ptr(std::size_t count, T value = T{}) {
    auto ptr{xpu::alloc<T>(count)};
    xpu::fill_n(ptr, count, value);
    data_.reset(ptr);
  }

  [[nodiscard]]
  auto get() const noexcept -> const T* { return data_.get(); }

  [[nodiscard]]
  auto get() noexcept -> T* { return data_.get(); }
};

template <typename T> [[nodiscard]] CUDA_CALLABLE
inline auto assume_aligned(T* ptr) noexcept -> T* {
  if constexpr (is_padded<T>) {
    return std::assume_aligned<default_align<T>>(ptr);
  } else {
    return ptr;
  }
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

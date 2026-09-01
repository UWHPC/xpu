#pragma once

#include <xpu/algorithm.hpp>
#include <xpu/config.hpp>
#include <cstring>
#include <memory>
#include <type_traits>

namespace xpu {

template <typename T>
inline constexpr std::size_t default_align{(xpu::simd_bytes > alignof(T)) ? xpu::simd_bytes : alignof(T)};

template <typename T>
inline constexpr bool is_padded{!xpu::xpu_cuda && sizeof(T) < xpu::simd_bytes};

template <typename T> [[nodiscard]] CUDA_CALLABLE
inline constexpr std::size_t bytes(std::size_t count) noexcept {
  static_assert(std::is_trivially_copyable_v<T>, "ERROR: xpu::bytes requires a trivially copyable type");
  return count * sizeof(T);
}

template <typename T> [[nodiscard]] CUDA_CALLABLE
inline constexpr std::size_t handle_pad(std::size_t unpadded) noexcept {
  if constexpr (is_padded<T>) {
    constexpr auto lanes{xpu::simd_bytes / sizeof(T)};
    return xpu::ceiling_div(unpadded, lanes) * lanes;
  } else {
    return unpadded;
  }
}

template <typename T> [[nodiscard]]
inline T* alloc(std::size_t count) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (count == 0u) { return nullptr; }

  const auto bytes{count * sizeof(T)};

#if defined(XPU_CUDA)
  void* ptr{};
  if(cudaMalloc(&ptr, bytes) != cudaSuccess) { ptr = nullptr; }
#else
  void* ptr{::operator new(bytes, std::align_val_t{default_align<T>}, std::nothrow)};
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
inline void free(void* ptr) {
#if defined(XPU_CUDA)
  cudaFree(ptr);
#else
  ::operator delete(ptr, std::align_val_t(default_align<T>));
#endif
}

struct deleter {
  template <typename T>
  void operator()(T* ptr) const {
    xpu::free<T>(ptr);
  }
};

template <typename T>
class unique_ptr {
private:
  std::unique_ptr<T[], deleter> data_;

public:
  unique_ptr()
    : data_{}
  { }

  explicit unique_ptr(std::size_t count, T value = T{}) {
    T* ptr{xpu::alloc<T>(count)};
    xpu::fill_n(ptr, count, value);
    data_.reset(ptr);
  }

  [[nodiscard]]
  const T* get() const noexcept { return data_.get(); }

  [[nodiscard]]
  T* get() noexcept { return data_.get(); }
};

template <typename T> [[nodiscard]] CUDA_CALLABLE
inline T* assume_aligned(T* ptr) noexcept {
  if constexpr (is_padded<T>) {
    return std::assume_aligned<default_align<T>>(ptr);
  } else {
    return ptr;
  }
}

inline void memset(
  void* RESTRICT dst,
  int value,
  std::size_t bytes
) {
  if (bytes == 0uz) { return; }

#if defined(XPU_CUDA)
  xpu::cu_check(cudaMemset(dst, value, bytes));
#else
  std::memset(dst, value, bytes);
#endif
}

inline void memcpy(
  void* RESTRICT dst,
  const void* RESTRICT src,
  std::size_t bytes
) {
  if (bytes == 0uz) { return; }

#if defined(XPU_CUDA)
  xpu::cu_check(cudaMemcpy(dst, src, bytes, cudaMemcpyDefault));
#else
  std::memcpy(dst, src, bytes);
#endif
}

template <typename T>
inline void copy_n(
  T* RESTRICT dst,
  const T* RESTRICT src,
  std::size_t count
) noexcept {
  static_assert(
    std::is_trivially_copyable_v<T>,
    "ERROR: xpu::copy_n requires trivially copyable type"
  );

  xpu::memcpy(dst, src, xpu::bytes<T>(count));
}

template <typename T>
inline void zero_n(
  T* RESTRICT dst,
  std::size_t count
) noexcept {
  static_assert(
    std::is_arithmetic_v<T>,
    "ERROR: xpu::zero_n requires arithmetic type"
  );

  xpu::memset(dst, 0, xpu::bytes<T>(count));
}

} // namespace xpu

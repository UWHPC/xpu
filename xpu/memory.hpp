#pragma once

#include <xpu/config.hpp>
#include <cstring>
#include <memory>
#include <type_traits>

namespace xpu {

template <typename T>
inline constexpr std::size_t default_align{
  (simd_bytes > alignof(T)) ? simd_bytes : alignof(T)
};

template<typename T> [[nodiscard]]
inline constexpr std::size_t round_up(std::size_t unpadded) {
  if constexpr (sizeof(T) >= simd_bytes) {
    return unpadded;
  } else {
    return (unpadded + (simd_bytes / sizeof(T)) - 1) & 
          ~((simd_bytes / sizeof(T)) - 1);
  }
}

template <typename T, std::size_t alignment = default_align<T>> [[nodiscard]]
inline T* alloc(std::size_t count) {
  const auto bytes{count * sizeof(T)};

#if defined(XPU_CUDA)
  void* ptr{};
  if(cudaMalloc(&ptr, bytes) != cudaSuccess) { ptr = nullptr; }
#else
  void* ptr{::operator new(bytes, std::align_val_t{alignment}, std::nothrow)};
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

template <typename T, std::size_t alignment = default_align<T>>
inline void free(void* ptr) {
#if defined(XPU_CUDA)
  cudaFree(ptr);
#else
  ::operator delete(ptr, std::align_val_t(alignment));
#endif
}

template <typename T>
inline void zero_n(T* ptr, std::size_t count) {
#if defined(XPU_CUDA)
  cuda_check(cudaMemset(ptr, 0, count * sizeof(T)));
#else
  std::fill_n(ptr, count, T{});
#endif
}

struct deleter {
  template <typename T>
  void operator()(T* ptr) const {
    xpu::free<T>(ptr);
  }
};

template <typename T>
using unique_ptr = std::unique_ptr<T, deleter>;

} // namespace xpu

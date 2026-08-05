#pragma once

#include <xpu/algorithm.hpp>
#include <xpu/config.hpp>
#include <cstring>
#include <memory>
#include <type_traits>

namespace xpu {

template <typename T>
inline constexpr std::size_t default_align{
  (simd_bytes > alignof(T)) ? simd_bytes : alignof(T)
};

template <typename T> [[nodiscard]] CUDA_CALLABLE
inline constexpr std::size_t handle_pad(std::size_t unpadded) {
  if constexpr (xpu::xpu_cuda || (sizeof(T) >= xpu::simd_bytes)) {
    return unpadded;
  } else {
    constexpr auto lanes{xpu::simd_bytes / sizeof(T)};
    return xpu::ceiling_div(unpadded, lanes) * lanes;
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
  const T* get() const { return data_.get(); } 

  [[nodiscard]]
  T* get() { return data_.get(); } 
};

} // namespace xpu

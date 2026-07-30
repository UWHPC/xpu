#pragma once

#include <xpu/algorithm.hpp>
#include <xpu/config.hpp>
#include <cstring>
#include <memory>
#include <type_traits>

namespace xpu {

namespace detail {

template<typename T> [[nodiscard]]
inline constexpr std::size_t round_up(std::size_t unpadded) {
  if constexpr (sizeof(T) >= simd_bytes) {
    return unpadded;
  } else {
    return (unpadded + (simd_bytes / sizeof(T)) - 1) & 
          ~((simd_bytes / sizeof(T)) - 1);
  }
}

template <typename T>
inline void zero_n(T* ptr, std::size_t count) {
#if defined(XPU_CUDA)
  cuda_check(cudaMemset(ptr, 0, count * sizeof(T)));
#else
  std::fill_n(ptr, count, T{});
#endif
}

} // namespace xpu::detail

template <typename T>
inline constexpr std::size_t default_align{
  (simd_bytes > alignof(T)) ? simd_bytes : alignof(T)
};

template <typename T> [[nodiscard]]
inline constexpr std::size_t handle_pad(std::size_t unpadded) {
  if constexpr (xpu::xpu_cuda) { return unpadded; }
  else { return xpu::detail::round_up<T>(unpadded); }
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

  explicit unique_ptr(std::size_t count) {
    T* ptr{xpu::alloc<T>(count)};
    xpu::detail::zero_n(ptr, count);
    data_.reset(ptr);
  }

  unique_ptr(std::size_t count, T value) {
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

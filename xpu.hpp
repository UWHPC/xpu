#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>

namespace xpu {

#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
  #define RESTRICT __restrict
#else
  #define RESTRICT
#endif


#ifndef XPU_SIMD_BYTES
  #if defined(__AVX512F__)
    #define XPU_SIMD_BYTES 64
  #elif defined(__AVX2__) || defined(__AVX__)
    #define XPU_SIMD_BYTES 32
  #else
    #define XPU_SIMD_BYTES 16
  #endif
#endif

inline constexpr std::size_t simd_bytes{XPU_SIMD_BYTES};
inline constexpr bool xpu_cuda{
#if defined(XPU_CUDA)
  true
#else
  false
#endif
};

#if defined(__CUDACC__)

template <typename T> __global__
void cudaBackendFill(
  T* RESTRICT ptr,
  std::size_t size,
  T value
) {
  const auto i{blockDim.x * blockIdx.x + threadIdx.x};
  if (i >= size) { return; }

  ptr[i] = value;
}

#endif

template <typename T, typename val>
inline void fill_n(
  T* RESTRICT ptr, 
  std::size_t size,
  val value
) {
#if defined(__CUDACC__)
  dim3 backendFillThreads(512);
  dim3 backendFillBlocks(
    cudaNumBlocks(size, backendFillThreads.x)
  );

  cudaBackendFill<<<
    backendFillBlocks, backendFillThreads
  >>>(
    ptr, size, value
  );
  CUDA_CHECK(cudaGetLastError());
#else
  std::fill_n(ptr, size, value);
#endif
}

template <std::size_t alignment>
inline void* alloc(std::size_t size) {
#if defined(XPU_CUDA)
  void* ptr{};
  CUDA_CHECK(cudaMalloc(&ptr, size));
  return ptr;
#else
  return ::operator new(size, std::align_val_t{alignment}, std::nothrow);
#endif
}

template <std::size_t alignment>
inline void free(void* ptr) {
#if defined(XPU_CUDA)
  cudaFree(ptr);
#else
  ::operator delete(ptr, std::align_val_t(alignment));
#endif
}

template <std::size_t alignment>
struct deleter {
  template <typename T>
  void operator()(T* ptr) const {
    free<alignment>(ptr);
  }
};

template <typename T, std::size_t num_arrays>
class soa {
private:
  static constexpr std::size_t elem_per_align_{simd_bytes / sizeof(T)};
  std::size_t num_elem_;
  std::size_t stride_length_;
  std::unique_ptr<T[], deleter<simd_bytes>> data_;

public:
  soa() : num_elem_{}, stride_length_{}, data_{} {}
  soa(soa&&) noexcept = default;
  soa& operator=(soa&&) noexcept = default;

  explicit soa(std::size_t num_elements)
  : num_elem_{num_elements}
  , stride_length_{xpu_cuda ? num_elements : round_up(num_elements)} {    
    const std::size_t total_elements{num_arrays * stride_length_};
    const std::size_t total_bytes{total_elements * sizeof(T)};

    T* ptr{static_cast<T*>(alloc<simd_bytes>(total_bytes))};
    if (!ptr) { throw std::bad_alloc(); }

    fill_n(ptr, stride_length_, 0);
    data_.reset(ptr);
  }

  [[nodiscard]] std::size_t num_elem() const {
    return num_elem_;
  }

  [[nodiscard]]
  std::size_t stride() const {
    return stride_length_;
  }

  [[nodiscard]]
  T* operator[](std::size_t array_index) {
    return data_.get() + array_index * stride();
  }

  [[nodiscard]]
  const T* operator[](std::size_t array_index) const {
    return data_.get() + array_index * stride();
  }

  [[nodiscard]]
  static constexpr std::size_t round_up(std::size_t unpadded) {
    return (unpadded + elem_per_align_ - 1) & 
          ~(elem_per_align_ - 1);
  }
};

} // xpu

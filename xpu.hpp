#pragma once

#include <cstddef>

namespace xpu {

template <std::size_t alignment>
inline void* alloc(std::size_t size) {
#ifdef USE_CUDA
  void* ptr{};
  CUDA_CHECK(cudaMalloc(&ptr, size));
  return ptr;
#else
  return ::operator new(size, std::align_val_t{alignment}, std::nothrow);
#endif
}

template <std::size_t alignment>
inline void free(void* ptr) {
#ifdef USE_CUDA
  cudaFree(ptr);
#else
  ::operator delete(ptr, std::align_val_t(alignment));
#endif
}

struct SoADeleter {
  template <typename T>
  void operator()(T* ptr) const {
    free<SIMD_BYTES>(ptr);
  }
};

template <typename T>
class soa {
private:
  static constexpr std::size_t elem_per_align_{SIMD_BYTES / sizeof(T)};
  std::size_t num_elem_;
  std::size_t stride_length_;
  std::unique_ptr<T[], SoADeleter> data_;

public:
  soa() : num_elem_{}, stride_length_{}, data_{} {}
  soa(soa&&) noexcept = default;
  soa& operator=(soa&&) noexcept = default;

  soa(std::size_t num_elements, std::size_t num_arrays)
  : num_elem_{num_elements}
  , stride_length_{round_up(num_elements)} {    
    const std::size_t total_elements{num_arrays * stride_length_};
    const std::size_t total_bytes{total_elements * sizeof(T)};

    T* ptr{static_cast<T*>(alloc<SIMD_BYTES>(total_bytes))};
    if (!ptr) { throw std::bad_alloc(); }

    #ifdef USE_CUDA
    CUDA_CHECK(cudaMemset(ptr, 0, total_bytes));
    #else
    std::fill_n(ptr, total_elements, T{});
    #endif

    data_.reset(ptr);
  }

  [[nodiscard]] std::size_t size() const {
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

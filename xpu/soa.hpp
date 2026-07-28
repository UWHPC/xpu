#pragma once

#include <xpu/config.hpp>
#include <xpu/memory.hpp>
#include <cstddef>

namespace xpu {

template <typename T, std::size_t num_arrays>
class soa {
private:
  std::size_t count_;
  std::size_t stride_size_;
  xpu::unique_ptr<T[]> data_;

public:
  soa() : count_{}, stride_size_{}, data_{} {}
  soa(soa&&) noexcept = default;
  soa& operator=(soa&&) noexcept = default;

  explicit soa(std::size_t count)
  : count_{count}
  , stride_size_{xpu_cuda ? count : xpu::round_up<T>(count)} {    
    const auto total_count{num_arrays * stride_size_};

    T* ptr{xpu::alloc<T>(total_count)};
    xpu::zero_n(ptr, total_count);
    
    data_.reset(ptr);
  }

  [[nodiscard]]
  std::size_t count() const {
    return count_;
  }

  [[nodiscard]]
  std::size_t stride() const {
    return stride_size_;
  }

  [[nodiscard]]
  T* operator[](std::size_t array_index) {
    return data_.get() + array_index * stride();
  }

  [[nodiscard]]
  const T* operator[](std::size_t array_index) const {
    return data_.get() + array_index * stride();
  }
};

} // namespace xpu

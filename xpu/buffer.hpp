#pragma once

#include <xpu/config.hpp>
#include <xpu/memory.hpp>

namespace xpu {

template <typename T>
class buffer {
private:
  std::size_t count_;
  xpu::unique_ptr<T> data_;

public:
  explicit buffer(std::size_t count)
    : count_{count}
    , data_{xpu::handle_pad<T>(count)}
  { }

  [[nodiscard]]
  std::size_t count() const {
    return count_;
  }

  [[nodiscard]]
  std::size_t capacity() const {
    return xpu::handle_pad<T>(count_);
  }

  [[nodiscard]]
  T* data() {
    return xpu::assume_aligned<T>(data_.get());
  }

  [[nodiscard]]
  const T* data() const {
    return xpu::assume_aligned<const T>(data_.get());
  }
};

} // namespace xpu

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
  buffer() : count_{}, data_{} {}
  buffer(soa&&) noexcept = default;
  buffer& operator=(buffer&&) noexcept = default;

  explicit buffer(std::size_t count)
    : count_{count}
    , data_{xpu::handle_pad<T>(count)}
  { }

  [[nodiscard]]
  std::size_t size() const {
    return count_;
  }

  [[nodiscard]]
  std::size_t capacity() const {
    return xpu::handle_pad<T>(count_);
  }

  [[nodiscard]] T* data() {
    return data_.get();
  }

  [[nodiscard]] const T* data() const {
    return data_.get();
  }

  [[nodiscard]]
  T& operator[](std::size_t i) {
    return data_.get()[i];
  }

  [[nodiscard]]
  const T& operator[](std::size_t i) const {
    return data_.get()[i];
  }

  // TODO: use a span for view.
};

}

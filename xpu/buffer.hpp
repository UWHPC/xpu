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
  constexpr auto count() const noexcept -> std::size_t {
    return count_;
  }

  [[nodiscard]]
  constexpr auto capacity() const noexcept -> std::size_t {
    return xpu::handle_pad<T>(count_);
  }

  [[nodiscard]]
  auto data() noexcept -> T* {
    return xpu::assume_aligned<T>(data_.get());
  }

  [[nodiscard]]
  auto data() const noexcept -> const T* {
    return xpu::assume_aligned<const T>(data_.get());
  }
};

} // namespace xpu

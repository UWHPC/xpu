#pragma once

#include <xpu/config.hpp>
#include <xpu/memory.hpp>

namespace xpu {

template <typename T>
class scalar {
private:
  xpu::unique_ptr<T> data_;

public:
  scalar() : data_{} {}

  explicit scalar(T value)
    : data_{1uz, value}
  { }

  // TODO: handle this completely; likely use some host <-> device.
};

}

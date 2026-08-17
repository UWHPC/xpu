#pragma once

#include <cmath>
#include <cstddef>
#include <cstdio>

namespace test {

inline constexpr auto count{
  static_cast<std::size_t>((1<<24)-1)
};

template <typename T>
bool near(T actual, T expected, T tolerance) {
  return std::abs(actual - expected) <= tolerance;
}

inline int fail(const char* message) {
  std::fputs(message, stderr);
  std::fputc('\n', stderr);
  return 1;
}

} // namespace test

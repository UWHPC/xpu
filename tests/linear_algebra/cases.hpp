#pragma once

#include "../support/check.hpp"

#include <xpu/buffer.hpp>
#include <xpu/linear_algebra.hpp>
#include <xpu/memory.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

template <typename T>
int run_linear_algebra_type() {
  constexpr auto order{3uz};
  constexpr auto stride{5uz};
  constexpr auto size{order * stride};
  constexpr T padding{T{19}};
  constexpr T tolerance{T{100} * std::numeric_limits<T>::epsilon()};

  constexpr std::array<T, order * order> original{
    T{0}, T{2}, T{1},
    T{1}, T{1}, T{0},
    T{2}, T{0}, T{1}
  };

  std::array<T, size> matrix_host{};
  std::array<T, size> inverse_host{};
  matrix_host.fill(padding);
  inverse_host.fill(padding);

  for (auto row{0uz}; row < order; ++row) {
    for (auto column{0uz}; column < order; ++column) {
      matrix_host[row * stride + column] = original[row * order + column];
    }
  }

  xpu::buffer<T> matrix{size};
  xpu::buffer<T> inverse{size};
  xpu::copy_n(matrix.data(), matrix_host.data(), size);
  xpu::copy_n(inverse.data(), inverse_host.data(), size);

  xpu::linalg::lu_factorization<T> factorization{order, stride};
  if (
    factorization.order() != order ||
    factorization.stride() != stride
  ) {
    return test::fail("lu_factorization shape accessors returned the wrong values");
  }

  if (
    factorization.factorize(matrix.data()) !=
    xpu::linalg::status::success
  ) {
    return test::fail("lu_factorization reported a nonsingular matrix as singular");
  }

  xpu::copy_n(matrix_host.data(), matrix.data(), size);
  auto diagonal_product{T{1}};
  for (auto i{0uz}; i < order; ++i) {
    diagonal_product *= matrix_host[i * stride + i];
    for (auto column{order}; column < stride; ++column) {
      if (matrix_host[i * stride + column] != padding) {
        return test::fail("factorize modified row padding");
      }
    }
  }

  if (!test::near(std::abs(diagonal_product), T{4}, tolerance)) {
    return test::fail("LU diagonal has the wrong determinant magnitude");
  }

  constexpr std::array<T, order> rhs_host{T{7}, T{3}, T{5}};
  constexpr std::array<T, order> expected_solution{T{1}, T{2}, T{3}};
  std::array<T, order> solution_host{};
  xpu::buffer<T> rhs{order};
  xpu::buffer<T> solution{order};
  xpu::copy_n(rhs.data(), rhs_host.data(), order);

  factorization.solve(matrix.data(), rhs.data(), solution.data());
  xpu::copy_n(solution_host.data(), solution.data(), order);

  for (auto i{0uz}; i < order; ++i) {
    if (!test::near(solution_host[i], expected_solution[i], tolerance)) {
      return test::fail("solve produced the wrong solution");
    }
  }

  factorization.invert(matrix.data(), inverse.data());
  xpu::copy_n(inverse_host.data(), inverse.data(), size);

  for (auto row{0uz}; row < order; ++row) {
    for (auto column{0uz}; column < order; ++column) {
      auto value{T{}};
      for (auto k{0uz}; k < order; ++k) {
        value += original[row * order + k] * inverse_host[k * stride + column];
      }

      const auto expected{(row == column) ? T{1} : T{0}};
      if (!test::near(value, expected, tolerance)) {
        return test::fail("invert did not produce a row-major inverse");
      }
    }

    for (auto column{order}; column < stride; ++column) {
      if (inverse_host[row * stride + column] != padding) {
        return test::fail("invert modified row padding");
      }
    }
  }

  xpu::linalg::transpose_square(inverse.data(), order, stride);
  xpu::copy_n(inverse_host.data(), inverse.data(), size);

  for (auto row{0uz}; row < order; ++row) {
    for (auto column{0uz}; column < order; ++column) {
      auto value{T{}};
      for (auto k{0uz}; k < order; ++k) {
        value += original[row * order + k] * inverse_host[column * stride + k];
      }

      const auto expected{(row == column) ? T{1} : T{0}};
      if (!test::near(value, expected, tolerance)) {
        return test::fail("transpose_square produced the wrong matrix");
      }
    }

    for (auto column{order}; column < stride; ++column) {
      if (inverse_host[row * stride + column] != padding) {
        return test::fail("transpose_square modified row padding");
      }
    }
  }

  matrix_host.fill(padding);
  constexpr std::array<T, order * order> singular{
    T{1}, T{0}, T{0},
    T{0}, T{1}, T{0},
    T{0}, T{0}, T{0}
  };
  for (auto row{0uz}; row < order; ++row) {
    for (auto column{0uz}; column < order; ++column) {
      matrix_host[row * stride + column] = singular[row * order + column];
    }
  }
  xpu::copy_n(matrix.data(), matrix_host.data(), size);

  if (
    factorization.factorize(matrix.data()) !=
    xpu::linalg::status::singular
  ) {
    return test::fail("lu_factorization did not report a singular matrix");
  }

  return 0;
}

inline int run_linear_algebra_cases() {
  if (const auto status{run_linear_algebra_type<float>()}; status != 0) {
    return status;
  }
  return run_linear_algebra_type<double>();
}

#pragma once

#include "../support/check.hpp"
#include "../support/aborts.hpp"

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
  constexpr auto padding{T{T{19}}};
  constexpr auto tolerance{T{T{100} * std::numeric_limits<T>::epsilon()}};

  constexpr auto original = std::array<T, order * order>{
    T{0}, T{2}, T{1},
    T{1}, T{1}, T{0},
    T{2}, T{0}, T{1}
  };

  auto matrix_host = std::array<T, size>{};
  auto inverse_host = std::array<T, size>{};
  matrix_host.fill(padding);
  inverse_host.fill(padding);

  for (auto row{0uz}; row < order; ++row) {
    for (auto column{0uz}; column < order; ++column) {
      matrix_host[row * stride + column] = original[row * order + column];
    }
  }

  auto matrix{xpu::buffer<T>{size}};
  auto inverse{xpu::buffer<T>{size}};
  xpu::copy_n(matrix.data(), matrix_host.data(), size);
  xpu::copy_n(inverse.data(), inverse_host.data(), size);

  auto factorization{xpu::linalg::lu_factorization<T>{order, stride}};
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

  constexpr auto rhs_host = std::array<T, order>{T{7}, T{3}, T{5}};
  constexpr auto expected_solution = std::array<T, order>{T{1}, T{2}, T{3}};
  auto solution_host = std::array<T, order>{};
  auto rhs{xpu::buffer<T>{order}};
  auto solution{xpu::buffer<T>{order}};
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
  constexpr auto singular = std::array<T, order * order>{
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

template <typename T>
int run_cholesky_type() {
  constexpr auto order{3uz};
  constexpr auto stride{5uz};
  constexpr auto size{order * stride};
  constexpr auto padding{T{19}};
  constexpr auto tolerance{T{100} * std::numeric_limits<T>::epsilon()};

  constexpr auto original = std::array<T, order * order>{
    T{4}, T{2}, T{2},
    T{2}, T{5}, T{1},
    T{2}, T{1}, T{6}
  };

  auto matrix_host = std::array<T, size>{};
  matrix_host.fill(padding);
  for (auto row{0uz}; row < order; ++row) {
    for (auto column{0uz}; column < order; ++column) {
      matrix_host[row * stride + column] = original[row * order + column];
    }
  }

  auto matrix{xpu::buffer<T>{size}};
  xpu::copy_n(matrix.data(), matrix_host.data(), size);

  auto factorization{xpu::linalg::cholesky_factorization<T>{order, stride}};
  if (
    factorization.order() != order ||
    factorization.stride() != stride
  ) {
    return test::fail("cholesky_factorization shape accessors returned the wrong values");
  }

  if (
    factorization.factorize(matrix.data()) !=
    xpu::linalg::status::success
  ) {
    return test::fail("cholesky_factorization reported a positive-definite matrix as not positive definite");
  }

  xpu::copy_n(matrix_host.data(), matrix.data(), size);
  auto diagonal_product{T{1}};
  for (auto i{0uz}; i < order; ++i) {
    diagonal_product *= matrix_host[i * stride + i];
    for (auto column{order}; column < stride; ++column) {
      if (matrix_host[i * stride + column] != padding) {
        return test::fail("Cholesky factorize modified row padding");
      }
    }
  }

  if (!test::near(diagonal_product * diagonal_product, T{80}, tolerance)) {
    return test::fail("Cholesky diagonal has the wrong determinant");
  }

  constexpr auto rhs_host = std::array<T, order>{T{14}, T{15}, T{22}};
  constexpr auto expected_solution = std::array<T, order>{T{1}, T{2}, T{3}};
  auto solution_host = std::array<T, order>{};
  auto rhs{xpu::buffer<T>{order}};
  auto solution{xpu::buffer<T>{order}};
  xpu::copy_n(rhs.data(), rhs_host.data(), order);

  factorization.solve(matrix.data(), rhs.data(), solution.data());
  xpu::copy_n(solution_host.data(), solution.data(), order);

  for (auto i{0uz}; i < order; ++i) {
    if (!test::near(solution_host[i], expected_solution[i], tolerance)) {
      return test::fail("Cholesky solve produced the wrong solution");
    }
  }

  matrix_host.fill(padding);
  constexpr auto not_positive_definite = std::array<T, order * order>{
    T{1}, T{0}, T{0},
    T{0}, T{1}, T{0},
    T{0}, T{0}, T{-1}
  };
  for (auto row{0uz}; row < order; ++row) {
    for (auto column{0uz}; column < order; ++column) {
      matrix_host[row * stride + column] =
        not_positive_definite[row * order + column];
    }
  }
  xpu::copy_n(matrix.data(), matrix_host.data(), size);

  if (
    factorization.factorize(matrix.data()) !=
    xpu::linalg::status::not_pd
  ) {
    return test::fail("cholesky_factorization did not report a non-positive-definite matrix");
  }

  return 0;
}

inline auto run_linear_algebra_checked_cases() -> int {
#if defined(__unix__)
  if (const auto failure{test::check_abort([] {
    auto factorization{xpu::linalg::cholesky_factorization<double>{std::numeric_limits<std::size_t>::max(), 1uz}};
  })}; failure) {

    return failure;
  }

  if (const auto failure{test::check_abort([] {
    auto factorization{xpu::linalg::cholesky_factorization<double>{1uz, std::numeric_limits<std::size_t>::max()}};
  })}; failure) {

    return failure;
  }

  if (const auto failure{test::check_abort([] {
    auto factorization{xpu::linalg::cholesky_factorization<double>{2uz, 1uz}};
  })}; failure) {

    return failure;
  }

  if (const auto failure{test::check_abort([] {
    auto factorization{xpu::linalg::lu_factorization<double>{std::numeric_limits<std::size_t>::max(), 1uz}};
  })}; failure) {

    return failure;
  }

  if (const auto failure{test::check_abort([] {
    auto factorization{xpu::linalg::lu_factorization<double>{1uz, std::numeric_limits<std::size_t>::max()}};
  })}; failure) {

    return failure;
  }

  if (const auto failure{test::check_abort([] {
    auto factorization{xpu::linalg::lu_factorization<double>{2uz, 1uz}};
  })}; failure) {

    return failure;
  }

  if (const auto failure{test::check_abort([] {
    xpu::linalg::transpose_square<double>(nullptr, std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max());
  })}; failure) {

    return failure;
  }

  if (const auto failure{test::check_abort([] {
    xpu::linalg::transpose_square<double>(nullptr, 1uz, std::numeric_limits<std::size_t>::max());
  })}; failure) {

    return failure;
  }

  if (const auto failure{test::check_abort([] {
    xpu::linalg::transpose_square<double>(nullptr, 2uz, 1uz);
  })}; failure) {

    return failure;
  }

#endif

  const auto success{0};

  return success;
}

inline int run_linear_algebra_cases() {
  if (const auto failure{run_linear_algebra_checked_cases()}; failure) {

    return failure;
  }

  if (const auto status{run_linear_algebra_type<float>()}; status != 0) {
    return status;
  }
  if (const auto status{run_linear_algebra_type<double>()}; status != 0) {
    return status;
  }
  if (const auto status{run_cholesky_type<float>()}; status != 0) {
    return status;
  }
  return run_cholesky_type<double>();
}

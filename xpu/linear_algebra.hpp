#pragma once

#include <xpu/buffer.hpp>
#include <xpu/config.hpp>
#include <xpu/detail/checked.hpp>
#include <xpu/launch.hpp>
#include <xpu/memory.hpp>

#if defined(XPU_CUDA)
  #include <cublas_v2.h>
  #include <cusolverDn.h>
#else
  #include <lapacke.h>
#endif

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace xpu {

/** @brief Matrix factorization and solve operations. */
namespace linalg {

/** @brief Result of a matrix factorization. */
enum class status {
  /** @brief Factorization succeeded. */
  success,
  /** @brief The matrix is singular. */
  singular,
  /** @brief The matrix is not positive definite. */
  not_pd
};

namespace detail {

/** @brief Report an unrecoverable linear algebra error and abort. */
[[noreturn]]
inline auto linalg_error(const char* message) noexcept -> void {
  std::fprintf(
    stderr,
    "xpu: %s\n",
    message
  );
  std::abort();
}

#if defined(XPU_CUDA)

/** @brief Write ones on the diagonal and zeros elsewhere in a strided matrix. */
template <supported_float T>
struct build_identity {
  T* matrix;
  std::size_t stride;

  DEVICE_ONLY
  auto operator()(const xpu::array<std::size_t, 2uz>& index) const -> void {
    const auto row{index[0]};
    const auto column{index[1]};

    const auto is_diagonal{row == column};
    const auto offset{row * stride + column};
    const auto value{is_diagonal ? T{1} : T{0}};

    matrix[offset] = value;
  }
};

/** @brief Swap each pair of elements across a square matrix's diagonal. */
template <supported_float T>
struct transpose_square {
  T* matrix;
  std::size_t stride;

  DEVICE_ONLY
  auto operator()(const xpu::array<std::size_t, 2uz>& index) const -> void {
    const auto row{index[0]};
    const auto column{index[1]};
    const auto skip_pair{row >= column};

    if (skip_pair) {

      return;
    }

    const auto upper_offset{row * stride + column};
    const auto lower_offset{column * stride + row};
    const auto temporary{matrix[upper_offset]};

    matrix[upper_offset] = matrix[lower_offset];
    matrix[lower_offset] = temporary;
  }
};

/** @brief Create a cuSOLVER handle or abort on failure. */
inline auto create_cusolver_handle() -> cusolverDnHandle_t {
  auto handle{cusolverDnHandle_t{}};
  xpu::cu_check(cusolverDnCreate(&handle));
  return handle;
}

/** @brief Return the number of T elements cuSOLVER needs as workspace to
 *  factor an @p order by @p order matrix with row stride @p stride by LU.
 */
template <supported_float T>
inline auto getrf_workspace_size(
  cusolverDnHandle_t handle,
  std::size_t order,
  std::size_t stride
) -> std::size_t {
  const auto vendor_order{xpu::detail::checked_cast<int>(order)};
  const auto vendor_stride{xpu::detail::checked_cast<int>(stride)};

  auto size{0};

  if constexpr (std::same_as<T, float>) {
    xpu::cu_check(cusolverDnSgetrf_bufferSize(
      handle,
      vendor_order, vendor_order,
      nullptr, vendor_stride,
      &size
    ));
  } else {
    xpu::cu_check(cusolverDnDgetrf_bufferSize(
      handle,
      vendor_order, vendor_order,
      nullptr, vendor_stride,
      &size
    ));
  }

  const auto workspace_size{xpu::detail::checked_cast<std::size_t>(size)};

  return workspace_size;
}

/** @brief Return the number of T elements cuSOLVER needs as workspace to
 *  Cholesky-factor an @p order by @p order matrix with row stride @p stride.
 */
template <supported_float T>
inline auto potrf_workspace_size(
  cusolverDnHandle_t handle,
  std::size_t order,
  std::size_t stride
) -> std::size_t {
  const auto vendor_order{xpu::detail::checked_cast<int>(order)};
  const auto vendor_stride{xpu::detail::checked_cast<int>(stride)};

  auto size{0};

  if constexpr (std::same_as<T, float>) {
    xpu::cu_check(cusolverDnSpotrf_bufferSize(
      handle, CUBLAS_FILL_MODE_UPPER,
      vendor_order,
      nullptr, vendor_stride,
      &size
    ));
  } else {
    xpu::cu_check(cusolverDnDpotrf_bufferSize(
      handle, CUBLAS_FILL_MODE_UPPER,
      vendor_order,
      nullptr, vendor_stride,
      &size
    ));
  }

  const auto workspace_size{xpu::detail::checked_cast<std::size_t>(size)};

  return workspace_size;
}

/** @brief Overwrite @p matrix with its LU factors and write pivot indices and
 *  the vendor result code to @p pivot and @p info.
 *  @details Uses the single- or double-precision cuSOLVER routine for T.
 */
template <supported_float T>
inline auto cusolver_getrf(
  cusolverDnHandle_t handle,
  std::size_t order,
  std::size_t stride,
  T* matrix,
  T* workspace,
  int* pivot,
  int* info
) -> void {
  const auto vendor_order{xpu::detail::checked_cast<int>(order)};
  const auto vendor_stride{xpu::detail::checked_cast<int>(stride)};

  if constexpr (std::same_as<T, float>) {
    xpu::cu_check(cusolverDnSgetrf(
      handle,
      vendor_order, vendor_order,
      matrix, vendor_stride,
      workspace, pivot, info
    ));
  } else {
    xpu::cu_check(cusolverDnDgetrf(
      handle,
      vendor_order, vendor_order,
      matrix, vendor_stride,
      workspace, pivot, info
    ));
  }
}

/** @brief Overwrite one triangle of @p matrix with its Cholesky factor and
 *  write the vendor result code to @p info.
 *  @details Passes CUBLAS_FILL_MODE_UPPER to cuSOLVER.
 */
template <supported_float T>
inline auto cusolver_potrf(
  cusolverDnHandle_t handle,
  std::size_t order,
  std::size_t stride,
  T* matrix,
  T* workspace,
  std::size_t workspace_size,
  int* info
) -> void {
  const auto vendor_order{xpu::detail::checked_cast<int>(order)};
  const auto vendor_stride{xpu::detail::checked_cast<int>(stride)};
  const auto vendor_workspace_size{xpu::detail::checked_cast<int>(workspace_size)};

  if constexpr (std::same_as<T, float>) {
    xpu::cu_check(cusolverDnSpotrf(
      handle, CUBLAS_FILL_MODE_UPPER,
      vendor_order,
      matrix, vendor_stride,
      workspace, vendor_workspace_size,
      info
    ));
  } else {
    xpu::cu_check(cusolverDnDpotrf(
      handle, CUBLAS_FILL_MODE_UPPER,
      vendor_order,
      matrix, vendor_stride,
      workspace, vendor_workspace_size,
      info
    ));
  }
}

/** @brief Overwrite @p solution with the solution of a system factored by
 *  cusolver_getrf().
 *  @details @p operation selects whether cuSOLVER solves the transposed system.
 *  The vendor result code is written to @p info.
 */
template <supported_float T>
inline auto cusolver_getrs(
  cusolverDnHandle_t handle,
  cublasOperation_t operation,
  std::size_t order,
  std::size_t right_hand_sides,
  const T* lower_upper,
  std::size_t lower_upper_stride,
  const int* pivot,
  T* solution,
  std::size_t solution_stride,
  int* info
) -> void {
  const auto vendor_order{xpu::detail::checked_cast<int>(order)};
  const auto vendor_right_hand_sides{xpu::detail::checked_cast<int>(right_hand_sides)};
  const auto vendor_lower_upper_stride{xpu::detail::checked_cast<int>(lower_upper_stride)};
  const auto vendor_solution_stride{xpu::detail::checked_cast<int>(solution_stride)};

  if constexpr (std::same_as<T, float>) {
    xpu::cu_check(cusolverDnSgetrs(
      handle, operation,
      vendor_order, vendor_right_hand_sides,
      lower_upper, vendor_lower_upper_stride,
      pivot,
      solution, vendor_solution_stride,
      info
    ));
  } else {
    xpu::cu_check(cusolverDnDgetrs(
      handle, operation,
      vendor_order, vendor_right_hand_sides,
      lower_upper, vendor_lower_upper_stride,
      pivot,
      solution, vendor_solution_stride,
      info
    ));
  }
}

/** @brief Overwrite @p solution with the solution of a system factored by
 *  cusolver_potrf(), writing the vendor result code to @p info.
 */
template <supported_float T>
inline auto cusolver_potrs(
  cusolverDnHandle_t handle,
  std::size_t order,
  std::size_t right_hand_sides,
  const T* factor,
  std::size_t factor_stride,
  T* solution,
  std::size_t solution_stride,
  int* info
) -> void {
  const auto vendor_order{xpu::detail::checked_cast<int>(order)};
  const auto vendor_right_hand_sides{xpu::detail::checked_cast<int>(right_hand_sides)};
  const auto vendor_factor_stride{xpu::detail::checked_cast<int>(factor_stride)};
  const auto vendor_solution_stride{xpu::detail::checked_cast<int>(solution_stride)};

  if constexpr (std::same_as<T, float>) {
    xpu::cu_check(cusolverDnSpotrs(
      handle, CUBLAS_FILL_MODE_UPPER,
      vendor_order, vendor_right_hand_sides,
      factor, vendor_factor_stride,
      solution, vendor_solution_stride,
      info
    ));
  } else {
    xpu::cu_check(cusolverDnDpotrs(
      handle, CUBLAS_FILL_MODE_UPPER,
      vendor_order, vendor_right_hand_sides,
      factor, vendor_factor_stride,
      solution, vendor_solution_stride,
      info
    ));
  }
}

#else

/** @brief Overwrite a row-major @p matrix with LU factors, fill @p pivot,
 *  and return LAPACKE's result code.
 */
template <supported_float T>
inline auto lapacke_getrf(
  T* RESTRICT matrix,
  lapack_int* RESTRICT pivot,
  std::size_t order,
  std::size_t stride
) -> lapack_int {
  const auto vendor_order{xpu::detail::checked_cast<lapack_int>(order)};
  const auto vendor_stride{xpu::detail::checked_cast<lapack_int>(stride)};

  if constexpr (std::same_as<T, float>) {
    return LAPACKE_sgetrf(
      LAPACK_ROW_MAJOR,
      vendor_order, vendor_order,
      matrix, vendor_stride,
      pivot
    );
  } else {
    return LAPACKE_dgetrf(
      LAPACK_ROW_MAJOR,
      vendor_order, vendor_order,
      matrix, vendor_stride,
      pivot
    );
  }
}

/** @brief Overwrite the lower triangle of a row-major @p matrix with its
 *  Cholesky factor and return LAPACKE's result code.
 */
template <supported_float T>
inline auto lapacke_potrf(
  T* RESTRICT matrix,
  std::size_t order,
  std::size_t stride
) -> lapack_int {
  const auto vendor_order{xpu::detail::checked_cast<lapack_int>(order)};
  const auto vendor_stride{xpu::detail::checked_cast<lapack_int>(stride)};

  if constexpr (std::same_as<T, float>) {
    return LAPACKE_spotrf(
      LAPACK_ROW_MAJOR, 'L',
      vendor_order,
      matrix, vendor_stride
    ); // use lower triangular
  } else {
    return LAPACKE_dpotrf(
      LAPACK_ROW_MAJOR, 'L',
      vendor_order,
      matrix, vendor_stride
    );
  }
}

/** @brief Overwrite @p inverse, containing LU factors, with the inverse
 *  computed using @p pivot; return LAPACKE's result code.
 */
template <supported_float T>
inline auto lapacke_getri(
  T* RESTRICT inverse,
  const lapack_int* RESTRICT pivot,
  std::size_t order,
  std::size_t stride
) -> lapack_int {
  const auto vendor_order{xpu::detail::checked_cast<lapack_int>(order)};
  const auto vendor_stride{xpu::detail::checked_cast<lapack_int>(stride)};

  if constexpr (std::same_as<T, float>) {
    return LAPACKE_sgetri(
      LAPACK_ROW_MAJOR,
      vendor_order,
      inverse, vendor_stride,
      pivot
    );
  } else {
    return LAPACKE_dgetri(
      LAPACK_ROW_MAJOR,
      vendor_order,
      inverse, vendor_stride,
      pivot
    );
  }
}

/** @brief Overwrite @p solution with the solution of one right-hand side
 *  using @p lower_upper and @p pivot; return LAPACKE's result code.
 */
template <supported_float T>
inline auto lapacke_getrs(
  const T* RESTRICT lower_upper,
  const lapack_int* RESTRICT pivot,
  T* RESTRICT solution,
  std::size_t order,
  std::size_t stride
) -> lapack_int {
  const auto vendor_order{xpu::detail::checked_cast<lapack_int>(order)};
  const auto vendor_stride{xpu::detail::checked_cast<lapack_int>(stride)};

  if constexpr (std::same_as<T, float>) {
    return LAPACKE_sgetrs(
      LAPACK_ROW_MAJOR, 'N',
      vendor_order, 1,
      lower_upper, vendor_stride,
      pivot,
      solution, 1
    );
  } else {
    return LAPACKE_dgetrs(
      LAPACK_ROW_MAJOR, 'N',
      vendor_order, 1,
      lower_upper, vendor_stride,
      pivot,
      solution, 1
    );
  }
}

/** @brief Overwrite @p solution with the solution of one right-hand side
 *  using the lower-triangular Cholesky factor; return LAPACKE's result code.
 */
template <supported_float T>
inline auto lapacke_potrs(
  const T* RESTRICT lower_upper,
  T* RESTRICT solution,
  std::size_t order,
  std::size_t stride
) -> lapack_int {
  const auto vendor_order{xpu::detail::checked_cast<lapack_int>(order)};
  const auto vendor_stride{xpu::detail::checked_cast<lapack_int>(stride)};

  if constexpr (std::same_as<T, float>) {
    return LAPACKE_spotrs(
      LAPACK_ROW_MAJOR, 'L',
      vendor_order, 1,
      lower_upper, vendor_stride,
      solution, 1
    );
  } else {
    return LAPACKE_dpotrs(
      LAPACK_ROW_MAJOR, 'L',
      vendor_order, 1,
      lower_upper, vendor_stride,
      solution, 1
    );
  }
}

#endif

} // namespace xpu::linalg::detail

/** @brief Transpose a square row-major matrix in place.
 *  @pre @p stride is at least @p order.
 */
template <supported_float T>
inline auto transpose_square(
  T* RESTRICT matrix,
  std::size_t order,
  std::size_t stride
) noexcept -> void {
  const auto empty_matrix{order == 0uz};

  if (empty_matrix) {

    return;
  }

  const auto invalid_stride{stride < order};

  if (invalid_stride) {
    detail::linalg_error("matrix stride is smaller than its order");
  }

  const auto matrix_size{xpu::detail::checked_mul(order, stride)};

  static_cast<void>(xpu::detail::checked_bytes<T>(matrix_size));

#if defined(XPU_CUDA)
  const auto range = xpu::range<2uz>{
    {0uz, 0uz},
    {order, order},
    {1uz, 1uz}
  };

  const auto transpose = detail::transpose_square<T>{
    matrix,
    stride
  };

  xpu::parallel_for(range, transpose);
#else
  for (auto row{0uz}; row < order; ++row) {
    for (auto column{row + 1uz}; column < order; ++column) {
      const auto col_idx{row * stride + column};
      const auto row_idx{column * stride + row};

      std::swap(
        matrix[col_idx],
        matrix[row_idx]
      );
    }
  }
#endif
}

/** @brief Reuse the Cholesky factor of a positive-definite matrix for
 *  multiple linear solves.
 */
template <supported_float T>
class cholesky_factorization {
private:
  std::size_t order_;
  std::size_t stride_;
  T* factor_{};

#if defined(XPU_CUDA)
  using dimension_type = int;
#else
  using dimension_type = lapack_int;
#endif

#if defined(XPU_CUDA)
  cusolverDnHandle_t handle_;
  xpu::buffer<T> workspace_;
  xpu::buffer<int> info_;
#endif

  /** @brief Return @p order after checking that both dimensions fit the
   *  backend integer type, @p stride is at least @p order, and the matrix
   *  byte count does not overflow.
   */
  [[nodiscard]]
  static auto checked_order(std::size_t order, std::size_t stride) noexcept -> std::size_t {
    static_cast<void>(xpu::detail::checked_cast<dimension_type>(order));
    static_cast<void>(xpu::detail::checked_cast<dimension_type>(stride));

    const auto invalid_stride{stride < order};

    if (invalid_stride) {
      detail::linalg_error("matrix stride is smaller than its order");
    }

    const auto matrix_size{xpu::detail::checked_mul(order, stride)};

    static_cast<void>(xpu::detail::checked_bytes<T>(matrix_size));

    return order;
  }

public:
  /** @brief Prepare to factor an @p order by @p order row-major matrix
   *  with @p stride elements between rows.
   */
  cholesky_factorization(std::size_t order, std::size_t stride)
    : order_{checked_order(order, stride)}
    , stride_{stride}
#if defined(XPU_CUDA)
    , handle_{detail::create_cusolver_handle()}
    , workspace_{detail::potrf_workspace_size<T>(handle_, order, stride)}
    , info_{1uz}
#endif
  { }

  /** @brief Destroy the cuSOLVER handle on CUDA. */
  ~cholesky_factorization() {
#if defined(XPU_CUDA)
    xpu::cu_check(cusolverDnDestroy(handle_));
#endif
  }

  /** @brief Return the matrix order. */
  [[nodiscard]]
  constexpr auto order() const noexcept -> std::size_t {
    return order_;
  }

  /** @brief Return the distance in elements between matrix rows. */
  [[nodiscard]]
  constexpr auto stride() const noexcept -> std::size_t {
    return stride_;
  }

  /** @brief Factor @p matrix in place for later solves.
   *  @return status::success if positive definite, otherwise status::not_pd.
   *  On success, the lower triangle contains the factor. Any call replaces
   *  the previous factorization.
   */
  [[nodiscard]]
  auto factorize(T* RESTRICT matrix) noexcept -> status {
    factor_ = nullptr;

#if defined(XPU_CUDA)
    detail::cusolver_potrf(
      handle_, order_, stride_,
      matrix, workspace_.data(), workspace_.size(), info_.data()
    );

    auto info{0};
    xpu::copy_n(&info, info_.data(), 1uz);
    if (info < 0) {
      detail::linalg_error("cuSOLVER potrf received an invalid argument");
    }
    if (info > 0) { return status::not_pd; }
#else
    const auto info{
      detail::lapacke_potrf(matrix, order_, stride_)
    };
    if (info < 0) {
      detail::linalg_error("LAPACKE potrf failed");
    }
    if (info > 0) { return status::not_pd; }
#endif

    factor_ = matrix;
    return status::success;
  }

  /** @brief Solve the most recently factorized system with right-hand side
   *  @p rhs, writing the result to @p solution without changing @p rhs.
   *  @pre @p factor is the matrix passed to the successful factorize() call.
   *  @pre @p rhs and @p solution do not alias.
   */
  auto solve(
    const T* RESTRICT factor,
    const T* RESTRICT rhs,
    T* RESTRICT solution
  ) noexcept -> void {
    if (!factor_) {
      detail::linalg_error("you must factor before you solve");
    }
    if (factor != factor_) {
      detail::linalg_error(
        "solve requires the most recently factorized matrix"
      );
    }
    if (rhs == solution) {
      detail::linalg_error("rhs and solution must not alias");
    }

    xpu::copy_n(solution, rhs, order_);

#if defined(XPU_CUDA)
    detail::cusolver_potrs(
      handle_, order_, 1uz,
      factor, stride_,
      solution, order_,
      info_.data()
    );

    auto info{0};
    xpu::copy_n(&info, info_.data(), 1uz);
    if (info != 0) {
      detail::linalg_error("cuSOLVER potrs received an invalid argument");
    }
#else
    const auto info{
      detail::lapacke_potrs(factor, solution, order_, stride_)
    };
    if (info != 0) {
      detail::linalg_error("LAPACKE potrs failed");
    }
#endif
  }

  auto operator=(const cholesky_factorization&) -> cholesky_factorization& = delete;
  cholesky_factorization(const cholesky_factorization&) = delete;
  auto operator=(cholesky_factorization&&) -> cholesky_factorization& = delete;
  cholesky_factorization(cholesky_factorization&&) = delete;
};

/** @brief Reuse the LU factor of a square matrix for linear solves or inversion.
 */
template <supported_float T>
class lu_factorization {
private:
  std::size_t order_;
  std::size_t stride_;
  T* lower_upper_{};

#if defined(XPU_CUDA)
  using dimension_type = int;
#else
  using dimension_type = lapack_int;
#endif

#if defined(XPU_CUDA)
  xpu::buffer<int> pivot_;
  cusolverDnHandle_t handle_;
  xpu::buffer<T> workspace_;
  xpu::buffer<int> info_;
#else
  xpu::buffer<lapack_int> pivot_;
#endif

  /** @brief Return @p order after checking that both dimensions fit the
   *  backend integer type, @p stride is at least @p order, and the matrix
   *  byte count does not overflow.
   */
  [[nodiscard]]
  static auto checked_order(std::size_t order, std::size_t stride) noexcept -> std::size_t {
    static_cast<void>(xpu::detail::checked_cast<dimension_type>(order));
    static_cast<void>(xpu::detail::checked_cast<dimension_type>(stride));

    const auto invalid_stride{stride < order};

    if (invalid_stride) {
      detail::linalg_error("matrix stride is smaller than its order");
    }

    const auto matrix_size{xpu::detail::checked_mul(order, stride)};

    static_cast<void>(xpu::detail::checked_bytes<T>(matrix_size));

    return order;
  }

public:
  /** @brief Prepare to factor an @p order by @p order row-major matrix
   *  with @p stride elements between rows.
   */
  lu_factorization(std::size_t order, std::size_t stride)
    : order_{checked_order(order, stride)}
    , stride_{stride}
#if defined(XPU_CUDA)
    , pivot_{order}
    , handle_{detail::create_cusolver_handle()}
    , workspace_{detail::getrf_workspace_size<T>(handle_, order, stride)}
    , info_{1uz}
#else
    , pivot_{order}
#endif
  { }

  /** @brief Destroy the cuSOLVER handle on CUDA. */
  ~lu_factorization() {
#if defined(XPU_CUDA)
    xpu::cu_check(cusolverDnDestroy(handle_));
#endif
  }

  /** @brief Return the matrix order. */
  [[nodiscard]]
  constexpr auto order() const noexcept -> std::size_t {
    return order_;
  }

  /** @brief Return the distance in elements between matrix rows. */
  [[nodiscard]]
  constexpr auto stride() const noexcept -> std::size_t {
    return stride_;
  }

  /** @brief Factor @p matrix in place for later solves or inversion.
   *  @return status::success if nonsingular, otherwise status::singular.
   *  Any call replaces the previous factorization.
   */
  [[nodiscard]]
  auto factorize(T* RESTRICT matrix) noexcept -> status {
    lower_upper_ = nullptr;

#if defined(XPU_CUDA)
    detail::cusolver_getrf(
      handle_, order_, stride_,
      matrix, workspace_.data(), pivot_.data(), info_.data()
    );

    auto info{0};
    xpu::copy_n(&info, info_.data(), 1uz);
    if (info < 0) {
      detail::linalg_error("cuSOLVER getrf received an invalid argument");
    }
    if (info > 0) { return status::singular; }
#else
    const auto info{
      detail::lapacke_getrf(
        matrix, pivot_.data(), order_, stride_
      )
    };
    if (info < 0) {
      detail::linalg_error("LAPACKE getrf failed");
    }
    if (info > 0) { return status::singular; }
#endif

    lower_upper_ = matrix;
    return status::success;
  }

  /** @brief Solve the most recently factorized system with right-hand side
   *  @p rhs, writing the result to @p solution without changing @p rhs.
   *  @pre @p lower_upper is the matrix passed to the successful factorize() call.
   *  @pre @p rhs and @p solution do not alias.
   */
  auto solve(
    const T* RESTRICT lower_upper,
    const T* RESTRICT rhs,
    T* RESTRICT solution
  ) noexcept -> void {
    if (lower_upper != lower_upper_) {
      detail::linalg_error(
        "solve requires the most recently factorized matrix"
      );
    }
    if (rhs == solution) {
      detail::linalg_error("rhs and solution must not alias");
    }

    xpu::copy_n(solution, rhs, order_);

#if defined(XPU_CUDA)
    detail::cusolver_getrs(
      handle_, CUBLAS_OP_T,
      order_, 1uz,
      lower_upper, stride_,
      pivot_.data(),
      solution, order_,
      info_.data()
    );

    auto info{0};
    xpu::copy_n(&info, info_.data(), 1uz);
    if (info != 0) {
      detail::linalg_error("cuSOLVER getrs received an invalid argument");
    }
#else
    const auto info{
      detail::lapacke_getrs(
        lower_upper, pivot_.data(), solution,
        order_, stride_
      )
    };
    if (info != 0) {
      detail::linalg_error("LAPACKE getrs failed");
    }
#endif
  }

  /** @brief Write the inverse of the most recently factorized matrix to
   *  @p inverse using the same row stride as the factorized matrix.
   *  @pre @p lower_upper is the matrix passed to the successful factorize() call.
   *  @pre @p lower_upper and @p inverse do not alias.
   */
  auto invert(
    const T* RESTRICT lower_upper,
    T* RESTRICT inverse
  ) noexcept -> void {
    if (lower_upper != lower_upper_) {
      detail::linalg_error(
        "invert requires the most recently factorized matrix"
      );
    }
    if (lower_upper == inverse) {
      detail::linalg_error("lower_upper and inverse must not alias");
    }

#if defined(XPU_CUDA)
    const auto range = xpu::range<2uz>{
      {0uz, 0uz},
      {order_, order_},
      {1uz, 1uz}
    };

    const auto initialize = detail::build_identity<T>{
      inverse,
      stride_
    };

    xpu::parallel_for(range, initialize);

    detail::cusolver_getrs(
      handle_, CUBLAS_OP_N,
      order_, order_,
      lower_upper, stride_,
      pivot_.data(),
      inverse, stride_,
      info_.data()
    );

    auto info{0};
    xpu::copy_n(&info, info_.data(), 1uz);
    if (info != 0) {
      detail::linalg_error("cuSOLVER getrs received an invalid argument");
    }
#else
    for (auto row{0uz}; row < order_; ++row) {
      xpu::copy_n(
        inverse + row * stride_,
        lower_upper + row * stride_,
        order_
      );
    }

    const auto info{
      detail::lapacke_getri(
        inverse, pivot_.data(), order_, stride_
      )
    };
    if (info != 0) {
      detail::linalg_error("LAPACKE getri failed");
    }
#endif
  }

  auto operator=(const lu_factorization&) -> lu_factorization& = delete;
  lu_factorization(const lu_factorization&) = delete;
  auto operator=(lu_factorization&&) -> lu_factorization& = delete;
  lu_factorization(lu_factorization&&) = delete;
};

} // namespace xpu::linalg

} // namespace xpu

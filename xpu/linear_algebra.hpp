#pragma once

#include <xpu/buffer.hpp>
#include <xpu/config.hpp>
#include <xpu/detail/checked.hpp>
#include <xpu/launch.hpp>
#include <xpu/memory.hpp>

#if defined(XPU_CUDA)
  #include <cublas_v2.h>
  #include <cusolverDn.h>
#elif defined(XPU_HIP)
  #include <hipsolver/hipsolver.h>
#else
  #include <lapacke.h>
#endif

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <utility>

// cuSOLVER and hipSOLVER share the dense API, so one GPU path calls either
#if defined(XPU_CUDA)
  #define XPU_SOLVER(name) cusolverDn##name
  #define XPU_SOLVER_NAME "cuSOLVER"
#elif defined(XPU_HIP)
  #define XPU_SOLVER(name) hipsolverDn##name
  #define XPU_SOLVER_NAME "hipSOLVER"
#endif

namespace xpu {

namespace linalg {

enum class status {
  success,
  singular,
  not_pd
};

namespace detail {

[[noreturn]]
inline auto linalg_error(const char* message) noexcept -> void {
  std::fprintf(
    stderr,
    "xpu: %s\n",
    message
  );
  std::abort();
}

#if defined(XPU_GPU)

#if defined(XPU_CUDA)
using solver_handle = cusolverDnHandle_t;
using solver_operation = cublasOperation_t;

inline constexpr auto solver_fill_upper{CUBLAS_FILL_MODE_UPPER};
inline constexpr auto solver_op_n{CUBLAS_OP_N};
inline constexpr auto solver_op_t{CUBLAS_OP_T};
#else
using solver_handle = hipsolverDnHandle_t;
using solver_operation = hipblasOperation_t;

inline constexpr auto solver_fill_upper{HIPBLAS_FILL_MODE_UPPER};
inline constexpr auto solver_op_n{HIPBLAS_OP_N};
inline constexpr auto solver_op_t{HIPBLAS_OP_T};
#endif

template <supported_float T>
struct build_identity {
  T* matrix;
  std::size_t stride;

  XPU_DEVICE_ONLY
  auto operator()(const xpu::array<std::size_t, 2uz>& index) const -> void {
    const auto row{index[0]};
    const auto column{index[1]};

    const auto is_diagonal{row == column};
    const auto offset{row * stride + column};
    const auto value{is_diagonal ? T{1} : T{0}};

    matrix[offset] = value;
  }
};

template <supported_float T>
struct transpose_square {
  T* matrix;
  std::size_t stride;

  XPU_DEVICE_ONLY
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

inline auto create_solver_handle() -> solver_handle {
  auto handle{solver_handle{}};
  xpu::cu_check(XPU_SOLVER(Create)(&handle));
  return handle;
}

template <supported_float T>
inline auto getrf_workspace_size(
  solver_handle handle,
  std::size_t order,
  std::size_t stride
) -> std::size_t {
  const auto vendor_order{xpu::detail::checked_cast<int>(order)};
  const auto vendor_stride{xpu::detail::checked_cast<int>(stride)};

  auto size{0};

  if constexpr (std::same_as<T, float>) {
    xpu::cu_check(XPU_SOLVER(Sgetrf_bufferSize)(
      handle,
      vendor_order, vendor_order,
      nullptr, vendor_stride,
      &size
    ));
  } else {
    xpu::cu_check(XPU_SOLVER(Dgetrf_bufferSize)(
      handle,
      vendor_order, vendor_order,
      nullptr, vendor_stride,
      &size
    ));
  }

  const auto workspace_size{xpu::detail::checked_cast<std::size_t>(size)};

  return workspace_size;
}

template <supported_float T>
inline auto potrf_workspace_size(
  solver_handle handle,
  std::size_t order,
  std::size_t stride
) -> std::size_t {
  const auto vendor_order{xpu::detail::checked_cast<int>(order)};
  const auto vendor_stride{xpu::detail::checked_cast<int>(stride)};

  auto size{0};

  if constexpr (std::same_as<T, float>) {
    xpu::cu_check(XPU_SOLVER(Spotrf_bufferSize)(
      handle, solver_fill_upper,
      vendor_order,
      nullptr, vendor_stride,
      &size
    ));
  } else {
    xpu::cu_check(XPU_SOLVER(Dpotrf_bufferSize)(
      handle, solver_fill_upper,
      vendor_order,
      nullptr, vendor_stride,
      &size
    ));
  }

  const auto workspace_size{xpu::detail::checked_cast<std::size_t>(size)};

  return workspace_size;
}

template <supported_float T>
inline auto solver_getrf(
  solver_handle handle,
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
    xpu::cu_check(XPU_SOLVER(Sgetrf)(
      handle,
      vendor_order, vendor_order,
      matrix, vendor_stride,
      workspace, pivot, info
    ));
  } else {
    xpu::cu_check(XPU_SOLVER(Dgetrf)(
      handle,
      vendor_order, vendor_order,
      matrix, vendor_stride,
      workspace, pivot, info
    ));
  }
}

template <supported_float T>
inline auto solver_potrf(
  solver_handle handle,
  std::size_t order,
  std::size_t stride,
  T* matrix,
  xpu::buffer_view<T> workspace,
  int* info
) -> void {
  const auto vendor_order{xpu::detail::checked_cast<int>(order)};
  const auto vendor_stride{xpu::detail::checked_cast<int>(stride)};
  const auto vendor_workspace_size{xpu::detail::checked_cast<int>(workspace.count())};

  if constexpr (std::same_as<T, float>) {
    xpu::cu_check(XPU_SOLVER(Spotrf)(
      handle, solver_fill_upper,
      vendor_order,
      matrix, vendor_stride,
      workspace.data(), vendor_workspace_size,
      info
    ));
  } else {
    xpu::cu_check(XPU_SOLVER(Dpotrf)(
      handle, solver_fill_upper,
      vendor_order,
      matrix, vendor_stride,
      workspace.data(), vendor_workspace_size,
      info
    ));
  }
}

template <supported_float T>
inline auto solver_getrs(
  solver_handle handle,
  solver_operation operation,
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
    xpu::cu_check(XPU_SOLVER(Sgetrs)(
      handle, operation,
      vendor_order, vendor_right_hand_sides,
      lower_upper, vendor_lower_upper_stride,
      pivot,
      solution, vendor_solution_stride,
      info
    ));
  } else {
    xpu::cu_check(XPU_SOLVER(Dgetrs)(
      handle, operation,
      vendor_order, vendor_right_hand_sides,
      lower_upper, vendor_lower_upper_stride,
      pivot,
      solution, vendor_solution_stride,
      info
    ));
  }
}

template <supported_float T>
inline auto solver_potrs(
  solver_handle handle,
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
    xpu::cu_check(XPU_SOLVER(Spotrs)(
      handle, solver_fill_upper,
      vendor_order, vendor_right_hand_sides,
      factor, vendor_factor_stride,
      solution, vendor_solution_stride,
      info
    ));
  } else {
    xpu::cu_check(XPU_SOLVER(Dpotrs)(
      handle, solver_fill_upper,
      vendor_order, vendor_right_hand_sides,
      factor, vendor_factor_stride,
      solution, vendor_solution_stride,
      info
    ));
  }
}

#else

template <supported_float T>
inline auto lapacke_getrf(
  T* XPU_RESTRICT matrix,
  lapack_int* XPU_RESTRICT pivot,
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

template <supported_float T>
inline auto lapacke_potrf(
  T* XPU_RESTRICT matrix,
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

template <supported_float T>
inline auto lapacke_getri(
  T* XPU_RESTRICT inverse,
  const lapack_int* XPU_RESTRICT pivot,
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

template <supported_float T>
inline auto lapacke_getrs(
  const T* XPU_RESTRICT lower_upper,
  const lapack_int* XPU_RESTRICT pivot,
  T* XPU_RESTRICT solution,
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

template <supported_float T>
inline auto lapacke_potrs(
  const T* XPU_RESTRICT lower_upper,
  T* XPU_RESTRICT solution,
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

template <supported_float T>
inline auto transpose_square(
  T* XPU_RESTRICT matrix,
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

#if defined(XPU_GPU)
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

template <supported_float T>
class cholesky_factorization {
private:
  std::size_t order_;
  std::size_t stride_;
  T* factor_{};

#if defined(XPU_GPU)
  using dimension_type = int;
#else
  using dimension_type = lapack_int;
#endif

#if defined(XPU_GPU)
  detail::solver_handle handle_;
  xpu::buffer<T> workspace_;
  xpu::buffer<int> info_;
#endif

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
  cholesky_factorization(std::size_t order, std::size_t stride)
    : order_{checked_order(order, stride)}
    , stride_{stride}
#if defined(XPU_GPU)
    , handle_{detail::create_solver_handle()}
    , workspace_{detail::potrf_workspace_size<T>(handle_, order, stride)}
    , info_{1uz}
#endif
  { }

  ~cholesky_factorization() {
#if defined(XPU_GPU)
    xpu::cu_check(XPU_SOLVER(Destroy)(handle_));
#endif
  }

  [[nodiscard]]
  constexpr auto order() const noexcept -> std::size_t {
    return order_;
  }

  [[nodiscard]]
  constexpr auto stride() const noexcept -> std::size_t {
    return stride_;
  }

  [[nodiscard]]
  auto factorize(T* XPU_RESTRICT matrix) noexcept -> status {
    factor_ = nullptr;

#if defined(XPU_GPU)
    detail::solver_potrf(
      handle_, order_, stride_,
      matrix, workspace_.view(), info_.data()
    );

    auto info{0};
    xpu::copy_n(&info, info_.data(), 1uz);
    if (info < 0) {
      detail::linalg_error(XPU_SOLVER_NAME " potrf received an invalid argument");
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

  auto solve(
    const T* XPU_RESTRICT factor,
    const T* XPU_RESTRICT rhs,
    T* XPU_RESTRICT solution
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

#if defined(XPU_GPU)
    detail::solver_potrs(
      handle_, order_, 1uz,
      factor, stride_,
      solution, order_,
      info_.data()
    );

    auto info{0};
    xpu::copy_n(&info, info_.data(), 1uz);
    if (info != 0) {
      detail::linalg_error(XPU_SOLVER_NAME " potrs received an invalid argument");
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

template <supported_float T>
class lu_factorization {
private:
  std::size_t order_;
  std::size_t stride_;
  T* lower_upper_{};

#if defined(XPU_GPU)
  using dimension_type = int;
#else
  using dimension_type = lapack_int;
#endif

#if defined(XPU_GPU)
  xpu::buffer<int> pivot_;
  detail::solver_handle handle_;
  xpu::buffer<T> workspace_;
  xpu::buffer<int> info_;
#else
  xpu::buffer<lapack_int> pivot_;
#endif

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
  lu_factorization(std::size_t order, std::size_t stride)
    : order_{checked_order(order, stride)}
    , stride_{stride}
#if defined(XPU_GPU)
    , pivot_{order}
    , handle_{detail::create_solver_handle()}
    , workspace_{detail::getrf_workspace_size<T>(handle_, order, stride)}
    , info_{1uz}
#else
    , pivot_{order}
#endif
  { }

  ~lu_factorization() {
#if defined(XPU_GPU)
    xpu::cu_check(XPU_SOLVER(Destroy)(handle_));
#endif
  }

  [[nodiscard]]
  constexpr auto order() const noexcept -> std::size_t {
    return order_;
  }

  [[nodiscard]]
  constexpr auto stride() const noexcept -> std::size_t {
    return stride_;
  }

  [[nodiscard]]
  auto factorize(T* XPU_RESTRICT matrix) noexcept -> status {
    lower_upper_ = nullptr;

#if defined(XPU_GPU)
    detail::solver_getrf(
      handle_, order_, stride_,
      matrix, workspace_.data(), pivot_.data(), info_.data()
    );

    auto info{0};
    xpu::copy_n(&info, info_.data(), 1uz);
    if (info < 0) {
      detail::linalg_error(XPU_SOLVER_NAME " getrf received an invalid argument");
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

  auto solve(
    const T* XPU_RESTRICT lower_upper,
    const T* XPU_RESTRICT rhs,
    T* XPU_RESTRICT solution
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

#if defined(XPU_GPU)
    detail::solver_getrs(
      handle_, detail::solver_op_t,
      order_, 1uz,
      lower_upper, stride_,
      pivot_.data(),
      solution, order_,
      info_.data()
    );

    auto info{0};
    xpu::copy_n(&info, info_.data(), 1uz);
    if (info != 0) {
      detail::linalg_error(XPU_SOLVER_NAME " getrs received an invalid argument");
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

  auto invert(
    const T* XPU_RESTRICT lower_upper,
    T* XPU_RESTRICT inverse
  ) noexcept -> void {
    if (lower_upper != lower_upper_) {
      detail::linalg_error(
        "invert requires the most recently factorized matrix"
      );
    }
    if (lower_upper == inverse) {
      detail::linalg_error("lower_upper and inverse must not alias");
    }

#if defined(XPU_GPU)
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

    detail::solver_getrs(
      handle_, detail::solver_op_n,
      order_, order_,
      lower_upper, stride_,
      pivot_.data(),
      inverse, stride_,
      info_.data()
    );

    auto info{0};
    xpu::copy_n(&info, info_.data(), 1uz);
    if (info != 0) {
      detail::linalg_error(XPU_SOLVER_NAME " getrs received an invalid argument");
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

#undef XPU_SOLVER
#undef XPU_SOLVER_NAME

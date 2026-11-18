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

#if defined(XPU_CUDA)

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

struct cusolver_handle {
  cusolverDnHandle_t handle{};

  cusolver_handle() {
    xpu::cu_check(cusolverDnCreate(&handle));
  }

  ~cusolver_handle() {
    static_cast<void>(cusolverDnDestroy(handle));
  }

  auto operator=(const cusolver_handle&) -> cusolver_handle& = delete;
  cusolver_handle(const cusolver_handle&) = delete;
};

// cusolverDnCreate costs ~0.4 ms, so factorizations on a thread share one handle
inline auto shared_cusolver_handle() -> cusolverDnHandle_t {
  thread_local const auto shared{cusolver_handle{}};

  return shared.handle;
}

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

template <supported_float T>
inline auto cusolver_potrf(
  cusolverDnHandle_t handle,
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
    xpu::cu_check(cusolverDnSpotrf(
      handle, CUBLAS_FILL_MODE_UPPER,
      vendor_order,
      matrix, vendor_stride,
      workspace.data(), vendor_workspace_size,
      info
    ));
  } else {
    xpu::cu_check(cusolverDnDpotrf(
      handle, CUBLAS_FILL_MODE_UPPER,
      vendor_order,
      matrix, vendor_stride,
      workspace.data(), vendor_workspace_size,
      info
    ));
  }
}

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

// The row-major matrix is handed to LAPACK as its column-major transpose, so
// no layout conversion is needed: getrf factors A^T and getrs solves with 'T'.
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
      LAPACK_COL_MAJOR,
      vendor_order, vendor_order,
      matrix, vendor_stride,
      pivot
    );
  } else {
    return LAPACKE_dgetrf(
      LAPACK_COL_MAJOR,
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
      LAPACK_COL_MAJOR, 'U',
      vendor_order,
      matrix, vendor_stride
    ); // column-major upper is the row-major lower triangle
  } else {
    return LAPACKE_dpotrf(
      LAPACK_COL_MAJOR, 'U',
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
      LAPACK_COL_MAJOR,
      vendor_order,
      inverse, vendor_stride,
      pivot
    );
  } else {
    return LAPACKE_dgetri(
      LAPACK_COL_MAJOR,
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
    return LAPACKE_sgetrs_work(
      LAPACK_COL_MAJOR, 'T',
      vendor_order, 1,
      lower_upper, vendor_stride,
      pivot,
      solution, vendor_order
    );
  } else {
    return LAPACKE_dgetrs_work(
      LAPACK_COL_MAJOR, 'T',
      vendor_order, 1,
      lower_upper, vendor_stride,
      pivot,
      solution, vendor_order
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
    return LAPACKE_spotrs_work(
      LAPACK_COL_MAJOR, 'U',
      vendor_order, 1,
      lower_upper, vendor_stride,
      solution, vendor_order
    );
  } else {
    return LAPACKE_dpotrs_work(
      LAPACK_COL_MAJOR, 'U',
      vendor_order, 1,
      lower_upper, vendor_stride,
      solution, vendor_order
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
  constexpr auto tile{32uz};
  constexpr auto parallel_order{512uz};
  const auto tiles{xpu::ceiling_div(order, tile)};

  // tile by tile, both swapped cache lines stay resident
  const auto transpose_tile_row{[=](std::size_t tile_row) {
    const auto row_begin{tile_row * tile};
    const auto row_end{xpu::min(order, row_begin + tile)};

    for (auto column_begin{row_begin}; column_begin < order; column_begin += tile) {
      const auto column_end{xpu::min(order, column_begin + tile)};

      for (auto row{row_begin}; row < row_end; ++row) {
        for (auto column{xpu::max(column_begin, row + 1uz)}; column < column_end; ++column) {
          const auto col_idx{row * stride + column};
          const auto row_idx{column * stride + row};

          std::swap(
            matrix[col_idx],
            matrix[row_idx]
          );
        }
      }
    }
  }};

  const auto serial{order < parallel_order};

  if (serial) {
    for (auto tile_row{0uz}; tile_row < tiles; ++tile_row) {
      transpose_tile_row(tile_row);
    }

    return;
  }

  #pragma omp parallel for schedule(dynamic)
  for (auto tile_row = 0uz; tile_row < tiles; ++tile_row) {
    transpose_tile_row(tile_row);
  }
#endif
}

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
#if defined(XPU_CUDA)
    , handle_{detail::shared_cusolver_handle()}
    , workspace_{detail::potrf_workspace_size<T>(handle_, order, stride)}
    , info_{1uz}
#endif
  { }

  ~cholesky_factorization() = default;

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

#if defined(XPU_CUDA)
    detail::cusolver_potrf(
      handle_, order_, stride_,
      matrix, workspace_.view(), info_.data()
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

#if defined(XPU_CUDA)
    detail::cusolver_potrs(
      handle_, order_, 1uz,
      factor, stride_,
      solution, order_,
      info_.data()
    );
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
#if defined(XPU_CUDA)
    , pivot_{order}
    , handle_{detail::shared_cusolver_handle()}
    , workspace_{detail::getrf_workspace_size<T>(handle_, order, stride)}
    , info_{1uz}
#else
    , pivot_{order}
#endif
  { }

  ~lu_factorization() = default;

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

#if defined(XPU_CUDA)
    detail::cusolver_getrs(
      handle_, CUBLAS_OP_T,
      order_, 1uz,
      lower_upper, stride_,
      pivot_.data(),
      solution, order_,
      info_.data()
    );
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

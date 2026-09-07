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
  singular
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

inline auto create_cusolver_handle() -> cusolverDnHandle_t {
  auto handle{cusolverDnHandle_t{}};
  xpu::cu_check(cusolverDnCreate(&handle));
  return handle;
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

#else

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

#endif

} // namespace xpu::linalg::detail

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
    , handle_{detail::create_cusolver_handle()}
    , workspace_{detail::getrf_workspace_size<T>(handle_, order, stride)}
    , info_{1uz}
#else
    , pivot_{order}
#endif
  { }

  ~lu_factorization() {
#if defined(XPU_CUDA)
    xpu::cu_check(cusolverDnDestroy(handle_));
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

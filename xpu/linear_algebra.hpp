#pragma once

#include <xpu/buffer.hpp>
#include <xpu/config.hpp>
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

template <supported_float T> __global__
auto cudaBuildIdentity(
  std::size_t order,
  std::size_t stride,
  T* RESTRICT matrix
) -> void {
  const auto [i]{xpu::global_index<1>()};
  const auto size{order * order};
  if (i >= size) { return; }

  const auto row{i / order};
  const auto column{i - row * order};
  const auto is_diagonal{row == column};
  const auto idx{row * stride + column};

  matrix[idx] = is_diagonal ? T{1} : T{0};
}

template <supported_float T> __global__
auto cudaTransposeSquare(
  std::size_t order,
  std::size_t stride,
  T* RESTRICT matrix
) -> void {
  const auto [column, row]{xpu::global_index<2>()};
  if (row >= order || column >= order || row >= column) { return; }

  const auto col_idx{row * stride + column};
  const auto row_idx{column * stride + row};
  const auto tmp{matrix[col_idx]};

  matrix[col_idx] = matrix[row_idx];
  matrix[row_idx] = tmp;
}

inline auto create_cusolver_handle() -> cusolverDnHandle_t {
  cusolverDnHandle_t handle{};
  xpu::cu_check(cusolverDnCreate(&handle));
  return handle;
}

template <supported_float T>
inline auto getrf_workspace_size(
  cusolverDnHandle_t handle,
  std::size_t order,
  std::size_t stride
) -> std::size_t {
  auto size{0};

  if constexpr (std::same_as<T, float>) {
    xpu::cu_check(cusolverDnSgetrf_bufferSize(
      handle,
      static_cast<int>(order), static_cast<int>(order),
      nullptr, static_cast<int>(stride),
      &size
    ));
  } else {
    xpu::cu_check(cusolverDnDgetrf_bufferSize(
      handle,
      static_cast<int>(order), static_cast<int>(order),
      nullptr, static_cast<int>(stride),
      &size
    ));
  }

  return static_cast<std::size_t>(size);
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
  if constexpr (std::same_as<T, float>) {
    xpu::cu_check(cusolverDnSgetrf(
      handle,
      static_cast<int>(order), static_cast<int>(order),
      matrix, static_cast<int>(stride),
      workspace, pivot, info
    ));
  } else {
    xpu::cu_check(cusolverDnDgetrf(
      handle,
      static_cast<int>(order), static_cast<int>(order),
      matrix, static_cast<int>(stride),
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
  if constexpr (std::same_as<T, float>) {
    xpu::cu_check(cusolverDnSgetrs(
      handle, operation,
      static_cast<int>(order), static_cast<int>(right_hand_sides),
      lower_upper, static_cast<int>(lower_upper_stride),
      pivot,
      solution, static_cast<int>(solution_stride),
      info
    ));
  } else {
    xpu::cu_check(cusolverDnDgetrs(
      handle, operation,
      static_cast<int>(order), static_cast<int>(right_hand_sides),
      lower_upper, static_cast<int>(lower_upper_stride),
      pivot,
      solution, static_cast<int>(solution_stride),
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
  if constexpr (std::same_as<T, float>) {
    return LAPACKE_sgetrf(
      LAPACK_ROW_MAJOR,
      static_cast<lapack_int>(order), static_cast<lapack_int>(order),
      matrix, static_cast<lapack_int>(stride),
      pivot
    );
  } else {
    return LAPACKE_dgetrf(
      LAPACK_ROW_MAJOR,
      static_cast<lapack_int>(order), static_cast<lapack_int>(order),
      matrix, static_cast<lapack_int>(stride),
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
  if constexpr (std::same_as<T, float>) {
    return LAPACKE_sgetri(
      LAPACK_ROW_MAJOR,
      static_cast<lapack_int>(order),
      inverse, static_cast<lapack_int>(stride),
      pivot
    );
  } else {
    return LAPACKE_dgetri(
      LAPACK_ROW_MAJOR,
      static_cast<lapack_int>(order),
      inverse, static_cast<lapack_int>(stride),
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
  if constexpr (std::same_as<T, float>) {
    return LAPACKE_sgetrs(
      LAPACK_ROW_MAJOR, 'N',
      static_cast<lapack_int>(order), 1,
      lower_upper, static_cast<lapack_int>(stride),
      pivot,
      solution, 1
    );
  } else {
    return LAPACKE_dgetrs(
      LAPACK_ROW_MAJOR, 'N',
      static_cast<lapack_int>(order), 1,
      lower_upper, static_cast<lapack_int>(stride),
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
  if (order == 0uz) { return; }

#if defined(XPU_CUDA)
  const dim3 threads{16u, 16u};
  const dim3 blocks{
    xpu::block_per_dim(order, threads.x),
    xpu::block_per_dim(order, threads.y)
  };

  detail::cudaTransposeSquare<<<
    blocks, threads
  >>>(
    order, stride,
    matrix
  );
  xpu::cu_check(cudaGetLastError());
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
  xpu::buffer<int> pivot_;
  cusolverDnHandle_t handle_;
  xpu::buffer<T> workspace_;
  xpu::buffer<int> info_;
#else
  xpu::buffer<lapack_int> pivot_;
#endif

public:
  lu_factorization(std::size_t order, std::size_t stride)
    : order_{order}
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
    const auto size{order_ * order_};
    const dim3 threads{256u};
    const dim3 blocks{xpu::block_per_dim(size, threads.x)};

    detail::cudaBuildIdentity<<<
      blocks, threads
    >>>(
      order_, stride_,
      inverse
    );
    xpu::cu_check(cudaGetLastError());

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

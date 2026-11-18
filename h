[1mdiff --git a/xpu/reduce.hpp b/xpu/reduce.hpp[m
[1mnew file mode 100644[m
[1mindex 0000000..a3e6f3c[m
[1m--- /dev/null[m
[1m+++ b/xpu/reduce.hpp[m
[36m@@ -0,0 +1,62 @@[m
[32m+[m[32m#pragma once[m
[32m+[m
[32m+[m[32m#include <xpu/array.hpp>[m
[32m+[m[32m#include <xpu/config.hpp>[m
[32m+[m[32m#include <xpu/launch.hpp>[m
[32m+[m[32m#include <xpu/math.hpp>[m
[32m+[m[32m#include <xpu/memory.hpp>[m
[32m+[m
[32m+[m[32m#include <cstddef>[m
[32m+[m
[32m+[m[32mnamespace xpu {[m
[32m+[m
[32m+[m[32mtemplate <arithmetic T>[m
[32m+[m[32minline auto accumulate(const T* ptr, std::size_t count) -> T {[m
[32m+[m[32m  const auto range = xpu::range<1uz>{[m
[32m+[m[32m    .begin = {0uz}, .end = {count}, .step = {1uz}[m
[32m+[m[32m  };[m
[32m+[m
[32m+[m[32m  const auto contribution = [ptr] CUDA_CALLABLE (const xpu::array<std::size_t, 1uz>& index) -> T {[m
[32m+[m[32m    return ptr[index[0]];[m
[32m+[m[32m  };[m
[32m+[m
[32m+[m[32m  const auto scratch_bytes{xpu::parallel_reduce_sum_bytes<T>(range, contribution)};[m
[32m+[m[32m  const auto scratch{xpu::make_unique<std::byte>(scratch_bytes)};[m
[32m+[m[32m  const auto sum_device{xpu::make_unique<T>()};[m
[32m+[m
[32m+[m[32m  xpu::parallel_reduce_sum(range, sum_device.get(), contribution, scratch.get(), scratch_bytes);[m
[32m+[m
[32m+[m[32m  auto sum{T{0}};[m
[32m+[m[32m  xpu::copy_n(&sum, sum_device.get(), 1uz);[m
[32m+[m
[32m+[m[32m  return sum;[m
[32m+[m[32m}[m
[32m+[m
[32m+[m[32mtemplate <arithmetic T>[m
[32m+[m[32minline auto average(const T* ptr, std::size_t count) -> T {[m
[32m+[m[32m  return accumulate(ptr, count) / static_cast<T>(count);[m
[32m+[m[32m}[m
[32m+[m
[32m+[m[32mtemplate <std::floating_point T>[m
[32m+[m[32minline auto geometric_mean(const T* ptr, std::size_t count) -> T {[m
[32m+[m[32m  const auto range = xpu::range<1uz>{[m
[32m+[m[32m    .begin = {0uz}, .end = {count}, .step = {1uz}[m
[32m+[m[32m  };[m
[32m+[m
[32m+[m[32m  const auto contribution = [ptr] CUDA_CALLABLE (const xpu::array<std::size_t, 1uz>& index) -> T {[m
[32m+[m[32m    return xpu::log(ptr[index[0]]);[m
[32m+[m[32m  };[m
[32m+[m
[32m+[m[32m  const auto scratch_bytes{xpu::parallel_reduce_sum_bytes<T>(range, contribution)};[m
[32m+[m[32m  const auto scratch{xpu::make_unique<std::byte>(scratch_bytes)};[m
[32m+[m[32m  const auto log_sum_device{xpu::make_unique<T>()};[m
[32m+[m
[32m+[m[32m  xpu::parallel_reduce_sum(range, log_sum_device.get(), contribution, scratch.get(), scratch_bytes);[m
[32m+[m
[32m+[m[32m  auto log_sum{T{0}};[m
[32m+[m[32m  xpu::copy_n(&log_sum, log_sum_device.get(), 1uz);[m
[32m+[m
[32m+[m[32m  return xpu::exp(log_sum / static_cast<T>(count));[m
[32m+[m[32m}[m
[32m+[m
[32m+[m[32m} // namespace xpu[m
[1mdiff --git a/xpu/xpu.hpp b/xpu/xpu.hpp[m
[1mindex c59b316..6146ce8 100644[m
[1m--- a/xpu/xpu.hpp[m
[1m+++ b/xpu/xpu.hpp[m
[36m@@ -7,4 +7,5 @@[m
 #include <xpu/math.hpp>[m
 #include <xpu/memory.hpp>[m
 #include <xpu/numeric.hpp>[m
[32m+[m[32m#include <xpu/reduce.hpp>[m
 #include <xpu/soa.hpp>[m

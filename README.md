# xpu

`xpu` is a small, header-only C++ library for code that runs on a CPU, NVIDIA
CUDA, or AMD HIP. It provides backend-aware allocation, contiguous buffers,
structure-of-arrays storage, math helpers, and basic GPU launch utilities.

The project is in early development. The API may change.

## Requirements

- C++23
- CMake 3.25 or newer
- CUDA 13.3 or newer for the CUDA backend
- A compatible CUDA host compiler, with GCC 15 or newer when GCC is used
- ROCm 7.0 or newer for the HIP backend, with hipCUB, rocPRIM and hipRAND
- A C++ standard library with `<mdspan>` for HIP builds: libstdc++ from GCC 15
  or newer, or libc++
- LAPACKE for the optional CPU linear-algebra component
- hipSOLVER for the optional HIP linear-algebra component
- Linux, or Windows through WSL2, for CUDA builds
- Linux for HIP builds

The base CPU-only library has no external dependencies.
CPU builds use OpenMP when CMake finds it; otherwise execution is serial.

On the CPU, `xpu::parallel_for` parallelizes the outermost range dimension
and applies OpenMP SIMD to the innermost dimension. One-dimensional ranges
use a combined `parallel for simd` loop. Callbacks must support independent
iterations, avoid unsynchronized shared writes, and must not throw exceptions.
SIMD code generation depends on the callback and compiler.
Zero-origin, unit-step CPU ranges select a specialized loop without coordinate
scaling or offsets. Range validation runs once before entering the parallel loop.
With OpenMP enabled, copyable CPU callbacks are copied per thread; mutations to
the callable's own state do not update the original. Captured pointers and
references still access the same data. Noncopyable callbacks are shared between
threads. Without OpenMP, the original callback is used directly.

## Building

CUDA is enabled by default when CMake finds a CUDA compiler. Set the target GPU
architecture explicitly:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++-15 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-15 \
  -DCMAKE_CUDA_ARCHITECTURES=86

cmake --build build
```

HIP is enabled by default when CMake finds ROCm's `clang++` and CUDA is not in
use. CUDA takes precedence when both toolchains are available. Point CMake at
ROCm's compiler and set the target GPU architecture explicitly:

```bash
cmake -S . -B build-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DXPU_ENABLE_CUDA=OFF

cmake --build build-hip
```

Without `CMAKE_HIP_ARCHITECTURES`, CMake targets the GPUs that
`rocm_agent_enumerator` reports, or the compiler's default when it finds none.

For a CPU-only build:

```bash
cmake -S . -B build-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DXPU_ENABLE_CUDA=OFF \
  -DXPU_ENABLE_HIP=OFF

cmake --build build-cpu
```

Enable the optional linear-algebra component with:

```bash
sudo apt install liblapacke-dev

cmake -S . -B build-cpu-linalg -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DXPU_ENABLE_CUDA=OFF \
  -DXPU_ENABLE_HIP=OFF \
  -DXPU_ENABLE_LINALG=ON

cmake --build build-cpu-linalg
```

To use xpu from another CMake project:

```cmake
add_subdirectory(external/xpu)
target_link_libraries(my_target PRIVATE xpu::xpu)
```

Link `xpu::linalg` instead when using `<xpu/linear_algebra.hpp>`.

Set `XPU_ENABLE_CUDA`, `XPU_ENABLE_HIP` and `XPU_ENABLE_LINALG` before
`add_subdirectory` when you need to select them explicitly.

## Buffers

`buffer<T>` owns a contiguous allocation. New buffers are initialized with
`T{}`.

```cpp
#include <xpu/xpu.hpp>

constexpr auto N{10'000uz};

xpu::buffer<float> values{N};
xpu::fill_n(values.data(), values.count(), 1.0f);

float* data{values.data()};
const auto count{values.count()};
const auto capacity{values.capacity()};
```

`count()` is the requested number of elements. `capacity()` includes any
alignment padding.

## Structure of arrays

`soa<T, N>` stores `N` equal-length arrays in one allocation. An enum with a
terminal count value is a convenient way to name the arrays.

```cpp
enum Axis : std::size_t { X, Y, Z, NUM_AXES };

xpu::soa<float, Axis::NUM_AXES> position{N};

float* x{position[Axis::X]};
float* y{position[Axis::Y]};
float* z{position[Axis::Z]};
```

- `count()` returns the logical elements in each array.
- `stride()` returns the distance between adjacent arrays.
- `storage_size()` returns the total elements used by the SoA layout.
- `operator[]` returns a pointer to one array.

## SoA views

`soa_view<T, N>` is a small, non-owning kernel argument containing an SoA base
pointer and element count.

```cpp
__global__
void add_components(xpu::soa_view<float, Axis::NUM_AXES> position) {
  const auto [i]{xpu::global_index<1>()};

  if (i >= position.count()) {
    return;
  }

  position[Axis::Z][i] =
    position[Axis::X][i] + position[Axis::Y][i];
}

const dim3 threads{256u};
const dim3 blocks{xpu::block_per_dim(position.count(), threads.x)};

add_components<<<blocks, threads>>>(position.view());
```

A const `soa` produces an `soa_view<const T, N>`. A view does not own memory and
is valid only while the original `soa` owns the allocation.

## Backend and memory model

`XPU_CUDA` or `XPU_HIP` selects the allocation backend:

| Backend | Allocation |
|---|---|
| CPU | aligned `operator new` |
| CUDA | `cudaMalloc` |
| HIP | `hipMalloc` |

The `xpu::xpu` CMake target sets `XPU_CUDA` for CUDA builds and `XPU_HIP` for
HIP builds. Do not set them on individual source files. They must have the same
value in every translation unit linked into a program. `<xpu/config.hpp>`
defines `XPU_GPU` for either GPU backend, and `xpu::xpu_cuda`, `xpu::xpu_hip`
and `xpu::xpu_gpu` expose the same choice as constants.

When CUDA is enabled, every translation unit that includes an xpu header must be
compiled by nvcc. These files normally use the `.cu` extension.

When HIP is enabled, every translation unit that includes an xpu header must be
compiled as HIP. Use the `.hip` extension or set the `LANGUAGE HIP` source file
property. The test suite does the latter to reuse its `.cu` sources.

GPU allocations are device memory. Pointers returned by `buffer` and `soa`
cannot be dereferenced by host code. Use `xpu::copy_n` or `xpu::memcpy` to move
data between host and device memory; the runtime infers the direction.

Allocation failure terminates the process with `std::abort`.

## Padding

CPU allocations are aligned to at least `xpu::simd_bytes`. GPU builds use
`xpu::cuda_align_bytes`, which is 128 bytes, instead. When the element type is
smaller than that alignment, capacities and SoA strides are padded to a multiple
of `alignment / sizeof(T)` elements (integer division), which fills whole
alignment blocks when `sizeof(T)` is a power of two.

The default SIMD width is 64 bytes with AVX-512, 32 bytes with AVX or AVX2, and
16 bytes otherwise. Pin it when layout must remain stable across machines:

```bash
cmake -S . -B build -DXPU_SIMD_BYTES=64
```

## Linear algebra

The optional linear-algebra component provides reusable partial-pivot LU
factorization for `float` and `double` matrices:

```cpp
#include <xpu/linear_algebra.hpp>

xpu::linalg::lu_factorization<double> factorization{order, stride};

if (
  factorization.factorize(matrix) ==
  xpu::linalg::status::success
) {
  factorization.solve(matrix, rhs, solution);
}
```

Use `solve()` for linear systems. Use `invert()` only when the inverse itself
is required. Matrices use row-major layout, and strides are measured in
elements. The component uses LAPACKE on the CPU, cuSOLVER on CUDA, and
hipSOLVER on HIP.

## Testing

```bash
./scripts/test.sh             # every available suite: CPU, CUDA, then HIP
./scripts/test.sh --sanitize  # CUDA Compute Sanitizer
./scripts/test.sh --cpp       # CPU test suite only
./scripts/test.sh --cu        # CUDA test suite only
./scripts/test.sh --hip       # HIP test suite only
```

The script finds ROCm's `clang++` through `HIPCXX` or
`$ROCM_PATH/llvm/bin/clang++`, where `ROCM_PATH` defaults to `/opt/rocm`. It
skips a GPU suite whose compiler is missing unless that suite was requested
explicitly. `--sanitize` applies to the CUDA suite only.

Behavioral tests are grouped by component under `tests/`. CPU and GPU entry
points are kept separate in `cpu.cpp` and `cuda.cu`, and HIP builds compile the
`cuda.cu` entry points as HIP. Backend-neutral cases live beside them in a
shared `cases.hpp`. Common test support lives in `tests/support`, umbrella-header
coverage lives in `tests/integration`, and every exported header also gets a
compile-only self-containment check.

GitHub Actions runs the CPU suite on Ubuntu 26.04 with GCC 15. CUDA runtime
testing is enabled when the repository variable `XPU_CUDA_CI` is `true` and a
self-hosted Linux x64 runner with the `gpu` label is available. HIP runtime
testing works the same way with the `XPU_HIP_CI` variable and a runner with the
`rocm` label.

## Current limitations

- A build uses one backend. CPU, CUDA and HIP cannot be mixed in one linked
  program.
- Random sequences are backend-specific. The same seed need not produce the
  same values on the CPU, CUDA and HIP.
- Accessors do not perform bounds checking.
- Multidimensional launch configuration is still a work in progress.
- Installed CMake package metadata is incomplete.

## License

`xpu` is available under the MIT License. See [LICENSE](LICENSE).

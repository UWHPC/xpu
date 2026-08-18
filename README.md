# xpu

`xpu` is a small, header-only C++ library for code that runs on a CPU or NVIDIA
CUDA. It provides backend-aware allocation, contiguous buffers,
structure-of-arrays storage, math helpers, and basic CUDA launch utilities.

The project is in early development. The API may change.

## Requirements

- C++23
- CMake 3.25 or newer
- CUDA 13.3 or newer for the CUDA backend
- A compatible CUDA host compiler, with GCC 14 or newer when GCC is used
- LAPACKE for the optional CPU linear-algebra component
- Linux, or Windows through WSL2, for CUDA builds

The base CPU-only library has no external dependencies.

## Building

CUDA is enabled by default when CMake finds a CUDA compiler. Set the target GPU
architecture explicitly:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-14 \
  -DCMAKE_CUDA_ARCHITECTURES=86

cmake --build build
```

For a CPU-only build:

```bash
cmake -S . -B build-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DXPU_ENABLE_CUDA=OFF

cmake --build build-cpu
```

Enable the optional linear-algebra component with:

```bash
sudo apt install liblapacke-dev

cmake -S . -B build-cpu-linalg -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DXPU_ENABLE_CUDA=OFF \
  -DXPU_ENABLE_LINALG=ON

cmake --build build-cpu-linalg
```

To use xpu from another CMake project:

```cmake
add_subdirectory(external/xpu)
target_link_libraries(my_target PRIVATE xpu::xpu)
```

Link `xpu::linalg` instead when using `<xpu/linear_algebra.hpp>`.

Set `XPU_ENABLE_CUDA` and `XPU_ENABLE_LINALG` before `add_subdirectory` when
you need to select them explicitly.

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

`count()` is the requested number of elements. `capacity()` includes any CPU
padding.

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

`XPU_CUDA` selects the allocation backend:

| Backend | Allocation |
|---|---|
| CPU | aligned `operator new` |
| CUDA | `cudaMalloc` |

The `xpu::xpu` CMake target sets `XPU_CUDA` for CUDA builds. Do not set it on
individual source files. It must have the same value in every translation unit
linked into a program.

When CUDA is enabled, every translation unit that includes an xpu header must be
compiled by nvcc. These files normally use the `.cu` extension.

CUDA allocations are device memory. Pointers returned by `buffer` and `soa`
cannot be dereferenced by host code. The library does not currently wrap memory
transfers, so use the CUDA runtime directly when transfers are required.

Allocation failure terminates the process with `std::abort`.

## Padding

CPU allocations are aligned to at least `xpu::simd_bytes`. Capacities and SoA
strides are padded to SIMD-lane multiples when the element type is smaller than
the SIMD width. CUDA uses a tight layout with no row padding.

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
elements.

## Testing

```bash
./scripts/test.sh             # CPU and CUDA test suites
./scripts/test.sh --sanitize  # CUDA Compute Sanitizer
./scripts/test.sh --cpp       # CPU test suite only
./scripts/test.sh --cu        # CUDA test suite only
```

Behavioral tests are grouped by component under `tests/`. CPU and CUDA entry
points are kept separate, while backend-neutral cases live beside them in a
shared `cases.hpp`. Common test support lives in `tests/support`, umbrella-header
coverage lives in `tests/integration`, and every exported header also gets a
compile-only self-containment check.

GitHub Actions runs the CPU suite on Ubuntu 24.04. CUDA runtime testing is
enabled when the repository variable `XPU_CUDA_CI` is `true` and a self-hosted
Linux x64 runner with the `gpu` label is available.

## Current limitations

- NVIDIA CUDA is the only GPU backend.
- CPU and CUDA backends cannot be mixed in one linked program.
- CUDA memory transfers are not wrapped.
- Accessors do not perform bounds checking.
- Multidimensional launch configuration is still a work in progress.
- Installed CMake package metadata is incomplete.

## License

`xpu` is available under the MIT License. See [LICENSE](LICENSE).

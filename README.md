# xpu

A portable CPU / GPU library.

The same types and the same call sites compile for the host or for CUDA, so you write the allocation and layout code once instead of wrapping every line in `#ifdef`.

xpu is a portability layer, not a parallel algorithms library. It gives you memory, alignment, layout, and a math surface that works on both sides. It does not try to be Thrust.

**Status: early. The API will change.**

## Requirements

- C++23
- GCC 14+ (nvcc rejects GCC 13 and MSVC as C++23 hosts, and it does that with a warning plus a silent fallback to C++14, not an error)
- CUDA 13.3+ and SM 7.5+ for the GPU path
- Linux, or Windows via WSL2
- CPU-only builds have no dependencies

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14 \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build
```

Or consume it directly:

```cmake
add_subdirectory(xpu)
target_link_libraries(my_target PRIVATE xpu::xpu)
```

`XPU_ENABLE_CUDA=OFF` builds the CPU-only path.

## Usage

```cpp
#include <xpu/xpu.hpp>

enum Grad : std::size_t { X, Y, Z, NUM };

// NUM arrays of num_particles elements, in one allocation
xpu::soa<double, Grad::NUM> grad{num_particles};

my_kernel<<<blocks, threads>>>(grad[Grad::X], grad[Grad::Y], grad.count());
```

The enum sentinel pattern is the intended idiom. It names your arrays and gives you the count in one declaration.

## The `XPU_CUDA` contract

Read this part.

`XPU_CUDA` selects the allocator. Define it and `xpu::alloc` calls `cudaMalloc`; leave it undefined and it calls aligned `operator new`. So:

> **`XPU_CUDA` must be defined identically for every translation unit that links together.**

If one TU is compiled with it and another without, one allocates with `cudaMalloc` and the other frees with `operator delete`. That is heap corruption, with no link error and no warning. Set it on the target so it propagates, which is what `xpu::xpu` does for you:

```cmake
target_compile_definitions(my_lib PUBLIC XPU_CUDA)
```

`XPU_CUDA` also requires nvcc. Every TU that includes an xpu header has to be a `.cu`, and `config.hpp` `#error`s if it isn't.

## Memory model

Under `XPU_CUDA` the pointers are **device memory**. `soa::operator[]` returns a `T*` the host cannot dereference.

That's the sharp edge: the signature is an identical `T*` in both builds, but in a CPU build you can read it and in a CUDA build you segfault. Passing it to a kernel is always fine. Touching it from host code is only fine without `XPU_CUDA`.

Allocation failure aborts with the requested byte count rather than throwing. OOM in a solver isn't recoverable, and a core dump at the failure point beats an unwound stack.

## Layout and alignment

`soa<T, N>` packs N arrays into one allocation. Rows are padded up to `xpu::simd_bytes` so every `soa[k]` starts aligned. `count()` is the logical element count, `stride()` is the padded distance between rows, so index with `soa[k][i]` and use `stride()` only if you're doing pointer arithmetic yourself.

On the CUDA path the padding is dropped deliberately. Coalescing wants a tight stride, and a CUDA allocation is already over-aligned at the base.

`simd_bytes` is detected from the architecture macros your compiler flags set, so it's 64 with AVX-512 and 16 without. Pin it if you need the value stable across machines:

```
-DXPU_SIMD_BYTES=64
```

## What's in it

| header | |
|---|---|
| `config.hpp` | backend detection, `cuda_check`, `simd_bytes`, the `xstd` alias |
| `memory.hpp` | `alloc<T>` / `free<T>` / `zero_n` / `deleter` / `unique_ptr` |
| `math.hpp` | the `<cmath>` surface, plus `sincos`, `rsqrt`, `norm3d`, `ceiling_div` |
| `launch.hpp` | `num_blocks`, `global_index<Dims>` |
| `algorithm.hpp` | `fill_n`, plus `min` / `max` |
| `soa.hpp` | N arrays of M elements, one allocation, aligned rows |

`xstd` aliases `cuda::std` under CUDA and `std` otherwise, so `xpu::exp` and `xpu::complex` are host- and device-callable without hand-written wrappers.

## License

See [LICENSE](LICENSE).

# xpu

A portable CPU / GPU library.

One header. The same types and the same call sites compile for the host or for
CUDA, so you write the allocation and layout code once instead of wrapping every
line in `#ifdef`.

xpu is a portability layer, not a parallel algorithms library. It gives you
memory, alignment, and layout. It does not try to be Thrust.

**Status: early. The API will change.**

## Requirements

- C++23;
- CUDA 13.3+ and SM 7.5+ for the GPU path;
- Linux, or Windows via WSL2;
- CPU-only builds have no dependencies

Set the same standard for both languages, since this header is included from
`.cu` files:

```cmake
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 23)
```

## Usage

```cpp
#include "xpu.hpp"

enum Grad : std::size_t { X, Y, Z, NUM };

// NUM arrays of num_particles elements, in one allocation.
xpu::soa<double> grad{num_particles, Grad::NUM};

my_kernel<<<blocks, threads>>>(grad[Grad::X], grad[Grad::Y], grad.num_elem());
```

The enum sentinel pattern is the intended idiom: it names your arrays and gives
you the count in one declaration.

## The `XPU_CUDA` contract

Read this part.

`XPU_CUDA` selects the allocator. Define it and `xpu::alloc` calls `cudaMalloc`;
leave it undefined and it calls aligned `operator new`. So:

> **`XPU_CUDA` must be defined identically for every translation unit that links
> together.**

If a `.cu` is compiled with it and a `.cpp` without, one allocates with
`cudaMalloc` and the other frees with `operator delete`. That is heap
corruption. There is no link error and no warning, and it will not reproduce
consistently.

Set it once, on the target, so it propagates:

```cmake
target_compile_definitions(my_lib PUBLIC XPU_CUDA)
```

Never per-file, and never in a source file above the `#include`.

## Memory model

Under `XPU_CUDA` the pointers are **device memory**. `soa::operator[]` returns a
`T*` that the host cannot dereference.

This is the sharp edge of the design: the signature is an identical `T*` in both
builds, but in a CPU build you can read it and in a CUDA build you segfault.
Passing it to a kernel is always fine. Touching it from host code is only fine
without `XPU_CUDA`.

## Layout and alignment

`soa` packs N arrays into a single allocation. Each array is padded up to
`SIMD_BYTES` so that every `soa[k]` starts aligned, and `stride()` gives you the
padded distance between them. Use `size()` for the logical element count and
`stride()` for indexing into rows.

On the CUDA path the padding is dropped deliberately. Coalescing wants a tight
stride, and the base of a CUDA allocation is already over-aligned.

`SIMD_BYTES` is detected from the architecture macros your compiler flags set,
which means it varies per translation unit if your flags do. Pin it if you care:

```
-DXPU_SIMD_BYTES=64
```

## API

| | |
|---|---|
| `xpu::alloc<A>(bytes)` | Allocate `bytes`, aligned to `A`. Returns `nullptr` on failure. |
| `xpu::free<A>(ptr)` | Release memory from `alloc<A>`. The alignment must match. |
| `xpu::soa<T>` | N arrays of M elements, one allocation, aligned rows. Move-only. |

## License

See [LICENSE](LICENSE).

cmake_minimum_required(VERSION 3.25)

if(
  NOT DEFINED CMAKE_EXPORT_COMPILE_COMMANDS
  OR CMAKE_EXPORT_COMPILE_COMMANDS STREQUAL ""
)
  set(
    CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE BOOL "Generate compile_commands.json for tooling" FORCE
  )
endif()

project(xpu LANGUAGES CXX)

option(XPU_ENABLE_CUDA "Build the CUDA backend if a CUDA compiler is available" ON)
option(XPU_ENABLE_HIP "Build the HIP backend if a HIP compiler is available and CUDA is not used" ON)
option(XPU_ENABLE_LINALG "Build the optional linear-algebra component" OFF)
option(XPU_ARCH_NATIVE "Tune for the host architecture (-march=native)" ON)
option(XPU_BUILD_TESTS "Build the xpu tests" ${PROJECT_IS_TOP_LEVEL})

set(XPU_SIMD_BYTES "" CACHE STRING
  "Pin the SIMD width in bytes (16/32/64). Empty means detect from the arch flags.")

# CUDA
set(XPU_CUDA OFF)

if(XPU_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)

  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES native)
    endif()

    enable_language(CUDA)

    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.3)
      message(FATAL_ERROR
        "xpu needs CUDA 13.3 or newer for -std=c++23 (found ${CMAKE_CUDA_COMPILER_VERSION}). "
        "Configure with -DXPU_ENABLE_CUDA=OFF to build the CPU-only path.")
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
       AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 15)
      message(FATAL_ERROR
        "nvcc rejects GCC ${CMAKE_CXX_COMPILER_VERSION} as a C++23 host compiler and "
        "silently downgrades to C++15. Use GCC 15+, e.g. "
        "-DCMAKE_CXX_COMPILER=g++-15 -DCMAKE_CUDA_HOST_COMPILER=g++-15")
    endif()

    if(MSVC)
      message(FATAL_ERROR
        "MSVC has no /std:c++23, so nvcc ignores -std=c++23 and falls back to C++15. "
        "Build the CUDA path under WSL2 or Linux.")
    endif()

    set(CMAKE_CUDA_EXTENSIONS OFF)

    find_package(CUDAToolkit REQUIRED)

    set(XPU_CUDA ON)
    message(STATUS
      "xpu: CUDA ${CMAKE_CUDA_COMPILER_VERSION}, arch ${CMAKE_CUDA_ARCHITECTURES}, "
      "host ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
  else()
    message(STATUS "xpu: no CUDA compiler found")
  endif()
endif()

# HIP
set(XPU_HIP OFF)

if(XPU_ENABLE_HIP AND NOT XPU_CUDA)
  include(CheckLanguage)
  check_language(HIP)

  if(CMAKE_HIP_COMPILER)
    enable_language(HIP)

    if(DEFINED CMAKE_HIP_PLATFORM AND NOT CMAKE_HIP_PLATFORM STREQUAL "amd")
      message(FATAL_ERROR
        "xpu's HIP backend targets AMD GPUs, but CMAKE_HIP_PLATFORM is ${CMAKE_HIP_PLATFORM}. "
        "Use the CUDA backend on NVIDIA, or configure with -DXPU_ENABLE_HIP=OFF.")
    endif()

    set(CMAKE_HIP_EXTENSIONS OFF)

    # Let the ROCm packages, and the dependencies they look up, find ROCm
    list(APPEND CMAKE_PREFIX_PATH ${CMAKE_HIP_COMPILER_ROCM_ROOT} $ENV{ROCM_PATH} /opt/rocm)

    # hip-config probes for GPUs unless GPU_TARGETS is set, so reuse CMake's list
    if(NOT DEFINED GPU_TARGETS)
      set(GPU_TARGETS ${CMAKE_HIP_ARCHITECTURES})
    endif()

    foreach(package IN ITEMS hip rocprim hipcub hiprand)
      find_package(${package} CONFIG)

      if(NOT ${package}_FOUND)
        message(FATAL_ERROR
          "xpu: found a HIP compiler but not the ROCm ${package} package. Point ROCM_PATH or "
          "CMAKE_PREFIX_PATH at ROCm, or configure with -DXPU_ENABLE_HIP=OFF to build the CPU-only path.")
      endif()
    endforeach()

    if(hip_VERSION VERSION_LESS 7.0)
      message(FATAL_ERROR
        "xpu needs ROCm 7.0 or newer for the HIP backend (found HIP ${hip_VERSION}). "
        "Configure with -DXPU_ENABLE_HIP=OFF to build the CPU-only path.")
    endif()

    # hipCUB's C++23 path uses std::extents, which needs <mdspan> from the standard library
    include(CheckSourceCompiles)
    block()
      # try_compile adds a -std of its own from the CMAKE_HIP_* standard settings after
      # CMAKE_REQUIRED_FLAGS, so pin that one to C++23 as well
      set(CMAKE_HIP_STANDARD 23)
      set(CMAKE_REQUIRED_FLAGS -std=c++23)
      set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
      check_source_compiles(HIP [=[
#include <mdspan>
int main() { return static_cast<int>(std::extents<int, 1>::rank()) - 1; }
]=] XPU_HIP_HAS_MDSPAN)
    endblock()

    if(NOT XPU_HIP_HAS_MDSPAN)
      message(FATAL_ERROR
        "The HIP compiler's C++ standard library has no std::extents in <mdspan>, which hipCUB "
        "needs under C++23. Install GCC 15 (ROCm's clang uses the newest GCC it finds) or use "
        "libc++, then reconfigure in a fresh build directory.")
    endif()

    set(XPU_HIP ON)
    message(STATUS
      "xpu: HIP ${hip_VERSION}, arch ${CMAKE_HIP_ARCHITECTURES}, "
      "compiler ${CMAKE_HIP_COMPILER_ID} ${CMAKE_HIP_COMPILER_VERSION}")
  else()
    message(STATUS "xpu: no HIP compiler found")
  endif()
endif()

if(XPU_CUDA OR XPU_HIP)
  set(XPU_GPU ON)
else()
  set(XPU_GPU OFF)
  message(STATUS "xpu: building the CPU-only path")
endif()

# Targets
add_library(xpu INTERFACE)
add_library(xpu::xpu ALIAS xpu)

target_include_directories(xpu INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(xpu INTERFACE cxx_std_23)

if(XPU_CUDA)
  target_compile_definitions(xpu INTERFACE XPU_CUDA)
  target_link_libraries(xpu INTERFACE CUDA::cudart)

  target_compile_options(xpu INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-std=c++23>)
elseif(XPU_HIP)
  target_compile_definitions(xpu INTERFACE XPU_HIP)
  # hip::hipcub pulls in hip::device, whose -x hip and --offload-arch would land on every CXX
  # source of a consumer, so take the hipCUB and rocPRIM headers without it
  target_link_libraries(xpu INTERFACE hip::host roc::rocprim hip::hiprand)
  target_include_directories(xpu SYSTEM INTERFACE
    $<TARGET_PROPERTY:hip::hipcub,INTERFACE_INCLUDE_DIRECTORIES>
  )

  target_compile_options(xpu INTERFACE $<$<COMPILE_LANGUAGE:HIP>:-std=c++23>)
else()
  find_package(OpenMP QUIET COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(xpu INTERFACE OpenMP::OpenMP_CXX)
  endif()
endif()

if(XPU_ENABLE_LINALG)
  add_library(xpu_linalg INTERFACE)
  add_library(xpu::linalg ALIAS xpu_linalg)
  set_target_properties(xpu_linalg PROPERTIES EXPORT_NAME linalg)

  target_link_libraries(xpu_linalg INTERFACE xpu::xpu)

  if(XPU_CUDA)
    target_link_libraries(xpu_linalg INTERFACE
      CUDA::cusolver
      CUDA::cublas
    )
  elseif(XPU_HIP)
    find_package(hipsolver CONFIG)

    if(NOT hipsolver_FOUND)
      message(FATAL_ERROR
        "xpu: XPU_ENABLE_LINALG on the HIP backend needs the ROCm hipsolver package. "
        "Install it, or configure with -DXPU_ENABLE_LINALG=OFF.")
    endif()

    target_link_libraries(xpu_linalg INTERFACE roc::hipsolver)
  else()
    find_package(LAPACK REQUIRED)
    find_path(XPU_LAPACKE_INCLUDE_DIR NAMES lapacke.h REQUIRED)
    find_library(XPU_LAPACKE_LIBRARY NAMES lapacke REQUIRED)

    target_include_directories(xpu_linalg SYSTEM INTERFACE
      ${XPU_LAPACKE_INCLUDE_DIR}
    )
    target_link_libraries(xpu_linalg INTERFACE
      ${XPU_LAPACKE_LIBRARY}
      LAPACK::LAPACK
    )
  endif()
endif()

if(XPU_ARCH_NATIVE AND NOT MSVC)
  target_compile_options(xpu INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
    $<$<COMPILE_LANGUAGE:HIP>:-march=native>
  )
endif()

if(NOT XPU_SIMD_BYTES STREQUAL "")
  target_compile_definitions(xpu INTERFACE XPU_SIMD_BYTES=${XPU_SIMD_BYTES})
  message(STATUS "xpu: SIMD width pinned to ${XPU_SIMD_BYTES} bytes")
endif()

# Tests
if(XPU_BUILD_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
  enable_testing()
  add_subdirectory(tests)
endif()

# Install
if(PROJECT_IS_TOP_LEVEL)
  include(GNUInstallDirs)

  install(DIRECTORY xpu DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(TARGETS xpu EXPORT xpuTargets)
  if(TARGET xpu_linalg)
    install(TARGETS xpu_linalg EXPORT xpuTargets)
  endif()
  install(EXPORT xpuTargets
    NAMESPACE xpu::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xpu
  )
endif()

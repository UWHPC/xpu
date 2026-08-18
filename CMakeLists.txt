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
       AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
      message(FATAL_ERROR
        "nvcc rejects GCC ${CMAKE_CXX_COMPILER_VERSION} as a C++23 host compiler and "
        "silently downgrades to C++14. Use GCC 14+, e.g. "
        "-DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14")
    endif()

    if(MSVC)
      message(FATAL_ERROR
        "MSVC has no /std:c++23, so nvcc ignores -std=c++23 and falls back to C++14. "
        "Build the CUDA path under WSL2 or Linux.")
    endif()

    set(CMAKE_CUDA_EXTENSIONS OFF)

    find_package(CUDAToolkit REQUIRED)

    set(XPU_CUDA ON)
    message(STATUS
      "xpu: CUDA ${CMAKE_CUDA_COMPILER_VERSION}, arch ${CMAKE_CUDA_ARCHITECTURES}, "
      "host ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
  else()
    message(STATUS "xpu: no CUDA compiler found, building the CPU-only path")
  endif()
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

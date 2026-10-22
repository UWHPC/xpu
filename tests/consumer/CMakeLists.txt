cmake_minimum_required(VERSION 3.25)
project(xpu_consumer LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
include("${XPU_INSTALL_PREFIX}/lib/cmake/xpu/xpuTargets.cmake")

add_executable(xpu_consumer main.cu)
target_link_libraries(xpu_consumer PRIVATE xpu::xpu xpu::linalg)

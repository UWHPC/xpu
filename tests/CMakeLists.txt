function(xpu_configure_test_target target)
  target_link_libraries(${target} PRIVATE xpu::xpu)

  if(XPU_CUDA)
    set_target_properties(${target} PROPERTIES
      CUDA_STANDARD 23
      CUDA_STANDARD_REQUIRED ON
    )
  else()
    set_target_properties(${target} PROPERTIES
      CXX_STANDARD 23
      CXX_STANDARD_REQUIRED ON
    )
  endif()
endfunction()

function(xpu_add_backend_test module)
  if(XPU_CUDA)
    set(source ${module}/cuda.cu)
  else()
    set(source ${module}/cpu.cpp)
  endif()

  if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source})
    return()
  endif()

  set(target xpu_test_${module})
  add_executable(${target} ${source})
  xpu_configure_test_target(${target})
  add_test(NAME ${target} COMMAND ${target})
endfunction()

foreach(module IN ITEMS
  algorithm
  buffer
  config
  launch
  math
  memory
  soa
  integration
)
  xpu_add_backend_test(${module})
endforeach()

# Keep header self-containment separate from behavioral tests. This list follows
# the headers exported by xpu/xpu.hpp; linear_algebra.hpp is still a scaffold and
# is not exported yet.
foreach(header IN ITEMS
  algorithm
  buffer
  config
  launch
  math
  memory
  soa
  xpu
)
  if(XPU_CUDA)
    set(extension cu)
  else()
    set(extension cpp)
  endif()

  set(source ${CMAKE_CURRENT_BINARY_DIR}/include_${header}.${extension})
  file(GENERATE
    OUTPUT ${source}
    CONTENT "#include <xpu/${header}.hpp>\n"
  )

  set(target xpu_include_${header})
  add_library(${target} OBJECT ${source})
  xpu_configure_test_target(${target})
endforeach()

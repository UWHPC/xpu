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
  if(module STREQUAL "linear_algebra" AND NOT TARGET xpu::linalg)
    return()
  endif()

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
  if(module STREQUAL "linear_algebra")
    target_link_libraries(${target} PRIVATE xpu::linalg)
  endif()
  add_test(NAME ${target} COMMAND ${target})
endfunction()

foreach(module IN ITEMS
  algorithm
  buffer
  config
  launch
  linear_algebra
  math
  memory
  soa
  integration
)
  xpu_add_backend_test(${module})
endforeach()

foreach(header IN ITEMS
  algorithm
  buffer
  config
  launch
  linear_algebra
  math
  memory
  soa
  xpu
)
  if(header STREQUAL "linear_algebra" AND NOT TARGET xpu::linalg)
    continue()
  endif()

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
  if(header STREQUAL "linear_algebra")
    target_link_libraries(${target} PRIVATE xpu::linalg)
  endif()
endforeach()

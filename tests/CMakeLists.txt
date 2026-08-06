if(XPU_CUDA)
  add_executable(xpu_smoke smoke.cu)
  set_target_properties(xpu_smoke PROPERTIES
    CUDA_STANDARD 23
    CUDA_STANDARD_REQUIRED ON
  )
else()
  add_executable(xpu_smoke smoke.cpp)
  set_target_properties(xpu_smoke PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
  )
endif()

target_link_libraries(xpu_smoke PRIVATE xpu::xpu)
add_test(NAME xpu_smoke COMMAND xpu_smoke)

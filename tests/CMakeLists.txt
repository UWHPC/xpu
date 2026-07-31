if(XPU_CUDA)
  add_executable(xpu_smoke smoke.cu)
  set_target_properties(xpu_smoke PROPERTIES
    CUDA_STANDARD 23
    CUDA_STANDARD_REQUIRED ON
  )
  target_link_libraries(xpu_smoke PRIVATE xpu::xpu)
  add_test(NAME xpu_smoke COMMAND xpu_smoke)
else()
  message(STATUS "xpu: CPU-only build, skipping the CUDA smoke test")
endif()

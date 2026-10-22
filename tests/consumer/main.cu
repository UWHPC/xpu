#include <xpu/soa.hpp>

#ifndef XPU_CUDA
  #error "The installed xpu::xpu target must enable the CUDA backend"
#endif

__global__ void copy_second_array(xpu::soa_view<int, 2> view) {
  if (threadIdx.x == 0) {
    view[1][0] = view[0][0];
  }
}

int main() {
  return 0;
}

#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")

at::Tensor ind2ptr_cuda(at::Tensor ind, int64_t M);
at::Tensor ptr2ind_cuda(at::Tensor ptr, int64_t E);

at::Tensor ind2ptr(at::Tensor ind, int64_t M) {
  CHECK_CUDA(ind);
  return ind2ptr_cuda(ind, M);
}

at::Tensor ptr2ind(at::Tensor ptr, int64_t E) {
  CHECK_CUDA(ptr);
  return ptr2ind_cuda(ptr, E);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ind2ptr", &ind2ptr, "Ind2Ptr (CUDA)");
  m.def("ptr2ind", &ptr2ind, "Ptr2Ind (CUDA)");
}

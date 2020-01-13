#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")

at::Tensor rowptr_cuda(at::Tensor row, int64_t size);

at::Tensor rowptr(at::Tensor row, int64_t size) {
  CHECK_CUDA(row);
  return rowptr_cuda(row, size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rowptr", &rowptr, "Rowptr (CUDA)");
}

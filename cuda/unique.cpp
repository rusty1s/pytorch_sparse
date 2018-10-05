#include <torch/torch.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")

std::tuple<at::Tensor, at::Tensor> unique_cuda(at::Tensor src);

std::tuple<at::Tensor, at::Tensor> unique(at::Tensor src) {
  CHECK_CUDA(src);
  return unique_cuda(src);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("unique", &unique, "Unique (CUDA)");
}

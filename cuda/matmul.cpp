#include <torch/torch.h>

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")

std::tuple<at::Tensor, at::Tensor> spspmm_cuda(at::Tensor A, at::Tensor B);

std::tuple<at::Tensor, at::Tensor> spspmm(at::Tensor A, at::Tensor B) {
  CHECK_CUDA(A);
  CHECK_CUDA(B);
  return spspmm_cuda(A, B);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spspmm", &spspmm, "Sparse-Sparse Matrix Multiplication (CUDA)");
}

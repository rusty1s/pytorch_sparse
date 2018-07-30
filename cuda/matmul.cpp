#include <torch/torch.h>

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")

at::Tensor spspmm_cuda(at::Tensor matrix1, at::Tensor matrix2);

at::Tensor spspmm(at::Tensor matrix1, at::Tensor matrix2) {
  CHECK_CUDA(matrix1);
  CHECK_CUDA(matrix2);
  return spspmm_cuda(matrix1, matrix2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spspmm", &spspmm, "Sparse-Sparse Matrix Multiplication (CUDA)");
}

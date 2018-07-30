#include <torch/torch.h>

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")

at::SparseTensor spspmm(at::SparseTensor matrix1, at::SparseTensor matrix2) {
  return matrix1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spspmm", &spspmm, "Sparse-Sparse Matrix Multiplication (CUDA)");
}

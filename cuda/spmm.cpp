#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")

at::Tensor spmm_cuda(at::Tensor rowptr, at::Tensor col, at::Tensor val,
                     at::Tensor mat);

at::Tensor spmm(at::Tensor rowptr, at::Tensor col, at::Tensor val,
                at::Tensor mat) {
  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  CHECK_CUDA(val);
  CHECK_CUDA(mat);
  return spmm_cuda(rowptr, col, val, mat);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spmm", &spmm, "Sparse Matrix Multiplication (CUDA)");
}

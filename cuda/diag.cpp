#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")

at::Tensor non_diag_mask_cuda(at::Tensor index, int64_t M, int64_t N,
                              int64_t k);

at::Tensor non_diag_mask(at::Tensor index, int64_t M, int64_t N, int64_t k) {
  CHECK_CUDA(index);
  return non_diag_mask_cuda(index, M, N, k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("non_diag_mask", &non_diag_mask, "Non-Diagonal Mask (CUDA)");
}

#include <torch/torch.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")

std::tuple<at::Tensor, at::Tensor>
spspmm_cuda(at::Tensor indexA, at::Tensor valueA, at::Tensor indexB,
            at::Tensor valueB, int m, int k, int n);

std::tuple<at::Tensor, at::Tensor> spspmm(at::Tensor indexA, at::Tensor valueA,
                                          at::Tensor indexB, at::Tensor valueB,
                                          int m, int k, int n) {
  CHECK_CUDA(indexA);
  CHECK_CUDA(valueA);
  CHECK_CUDA(indexB);
  CHECK_CUDA(valueB);
  return spspmm_cuda(indexA, valueA, indexB, valueB, m, k, n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spspmm", &spspmm, "Sparse-Sparse Matrix Multiplication (CUDA)");
}

#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")

std::tuple<at::Tensor, at::Tensor>
spspmm_cuda(at::Tensor indexA, at::Tensor valueA, at::Tensor indexB,
            at::Tensor valueB, size_t m, size_t k, size_t n);
at::Tensor spspmm_bw_cuda(at::Tensor index, at::Tensor indexA,
                          at::Tensor valueA, at::Tensor indexB,
                          at::Tensor valueB, size_t rowA_max, size_t rowB_max);

std::tuple<at::Tensor, at::Tensor> spspmm(at::Tensor indexA, at::Tensor valueA,
                                          at::Tensor indexB, at::Tensor valueB,
                                          size_t m, size_t k, size_t n) {
  CHECK_CUDA(indexA);
  CHECK_CUDA(valueA);
  CHECK_CUDA(indexB);
  CHECK_CUDA(valueB);
  return spspmm_cuda(indexA, valueA, indexB, valueB, m, k, n);
}

at::Tensor spspmm_bw(at::Tensor index, at::Tensor indexA, at::Tensor valueA,
                     at::Tensor indexB, at::Tensor valueB, size_t rowA_max,
                     size_t rowB_max) {
  CHECK_CUDA(index);
  CHECK_CUDA(indexA);
  CHECK_CUDA(valueA);
  CHECK_CUDA(indexB);
  CHECK_CUDA(valueB);
  return spspmm_bw_cuda(index, indexA, valueA, indexB, valueB, rowA_max,
                        rowB_max);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spspmm", &spspmm, "Sparse-Sparse Matrix Multiplication (CUDA)");
  m.def("spspmm_bw", &spspmm_bw,
        "Sparse-Sparse Matrix Multiplication Backward (CUDA)");
}

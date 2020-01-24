#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")

std::tuple<at::Tensor, at::Tensor, at::optional<at::Tensor>>
spspmm_cuda(at::Tensor rowptrA, at::Tensor colA,
            at::optional<at::Tensor> valueA, at::Tensor rowptrB,
            at::Tensor colB, at::optional<at::Tensor> valueB, int M, int N,
            int K);

std::tuple<at::Tensor, at::Tensor, at::optional<at::Tensor>>
spspmm(at::Tensor rowptrA, at::Tensor colA, at::optional<at::Tensor> valueA,
       at::Tensor rowptrB, at::Tensor colB, at::optional<at::Tensor> valueB,
       int M, int N, int K) {
  CHECK_CUDA(rowptrA);
  CHECK_CUDA(colA);
  if (valueA.has_value())
    CHECK_CUDA(valueA.value());
  CHECK_CUDA(rowptrB);
  CHECK_CUDA(colB);
  if (valueB.has_value())
    CHECK_CUDA(valueB.value());
  return spspmm_cuda(rowptrA, colA, valueA, rowptrB, colB, valueB, M, N, K);
}

// std::tuple<at::Tensor, at::Tensor>
// spspmm_cuda(at::Tensor indexA, at::Tensor valueA, at::Tensor indexB,
//             at::Tensor valueB, size_t m, size_t k, size_t n);
// at::Tensor spspmm_bw_cuda(at::Tensor index, at::Tensor indexA,
//                           at::Tensor valueA, at::Tensor indexB,
//                           at::Tensor valueB, size_t rowA_max, size_t
//                           rowB_max);

// std::tuple<at::Tensor, at::Tensor> spspmm(at::Tensor indexA, at::Tensor
// valueA,
//                                           at::Tensor indexB, at::Tensor
//                                           valueB, size_t m, size_t k, size_t
//                                           n) {
//   CHECK_CUDA(indexA);
//   CHECK_CUDA(valueA);
//   CHECK_CUDA(indexB);
//   CHECK_CUDA(valueB);
//   return spspmm_cuda(indexA, valueA, indexB, valueB, m, k, n);
// }

// at::Tensor spspmm_bw(at::Tensor index, at::Tensor indexA, at::Tensor valueA,
//                      at::Tensor indexB, at::Tensor valueB, size_t rowA_max,
//                      size_t rowB_max) {
//   CHECK_CUDA(index);
//   CHECK_CUDA(indexA);
//   CHECK_CUDA(valueA);
//   CHECK_CUDA(indexB);
//   CHECK_CUDA(valueB);
//   return spspmm_bw_cuda(index, indexA, valueA, indexB, valueB, rowA_max,
//                         rowB_max);
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spspmm", &spspmm, "Sparse-Sparse Matrix Multiplication (CUDA)");
  // m.def("spspmm_bw", &spspmm_bw,
  //       "Sparse-Sparse Matrix Multiplication Backward (CUDA)");
}

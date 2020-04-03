#include <Python.h>
#include <torch/script.h>

#ifdef WITH_CUDA
#include "cuda/padding_cuda.h"
#endif

#ifdef _WIN32
PyMODINIT_FUNC PyInit__padding(void) { return NULL; }
#endif

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<int64_t>, std::vector<int64_t>>
padded_index(torch::Tensor rowptr, torch::Tensor col, torch::Tensor rowcount,
             torch::Tensor binptr) {
  return padded_index_cuda(rowptr, col, rowcount, binptr);
}

torch::Tensor padded_index_select(torch::Tensor src, torch::Tensor index,
                                  torch::Tensor fill_value) {
  return padded_index_select_cuda(src, index, fill_value);
}

static auto registry =
    torch::RegisterOperators()
        .op("torch_sparse::padded_index", &padded_index)
        .op("torch_sparse::padded_index_select", &padded_index_select);

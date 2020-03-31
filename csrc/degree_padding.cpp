#include <Python.h>
#include <torch/script.h>

#ifdef WITH_CUDA
#include "cuda/degree_padding_cuda.h"
#endif

#ifdef _WIN32
PyMODINIT_FUNC PyInit__degree_padding(void) { return NULL; }
#endif

std::vector<torch::Tensor> bin_assignment(torch::Tensor rowcount,
                                          torch::Tensor bin_strategy) {
  if (rowcount.device().is_cuda()) {
#ifdef WITH_CUDA
    return bin_assignment_cuda(rowcount, bin_strategy);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    AT_ERROR("Not implemented yet");
  }
}

std::tuple<torch::Tensor, torch::Tensor>
padded_index_select(torch::Tensor src, torch::Tensor rowptr, torch::Tensor col,
                    torch::Tensor index, int64_t length,
                    torch::Tensor fill_value) {
  if (src.device().is_cuda()) {
#ifdef WITH_CUDA
    return padded_index_select_cuda(src, rowptr, col, index, length,
                                    fill_value);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    AT_ERROR("Not implemented yet");
  }
}

static auto registry =
    torch::RegisterOperators()
        .op("torch_sparse::bin_assignment", &bin_assignment)
        .op("torch_sparse::padded_index_select", &padded_index_select);

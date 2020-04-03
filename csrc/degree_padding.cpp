#include <Python.h>
#include <torch/script.h>

#ifdef WITH_CUDA
#include "cuda/degree_padding_cuda.h"
#endif

#ifdef _WIN32
PyMODINIT_FUNC PyInit__degree_padding(void) { return NULL; }
#endif

std::tuple<std::vector<torch::Tensor>, std::vector<int64_t>>
bin_assignment(torch::Tensor rowcount, torch::Tensor binptr) {
  if (rowcount.device().is_cuda()) {
#ifdef WITH_CUDA
    return bin_assignment_cuda(rowcount, binptr);
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

// std::tuple<torch::Tensor, torch::Tensor>
// padded_index_select2(torch::Tensor src, torch::Tensor rowptr, torch::Tensor
// col,
//                      torch::Tensor bin, torch::Tensor index,
//                      std::vector<int64_t> node_counts,
//                      std::vector<int64_t> lengths, torch::Tensor fill_value)
//                      {
//   if (src.device().is_cuda()) {
// #ifdef WITH_CUDA
//     return padded_index_select_cuda2(src, rowptr, col, bin, index,
//     node_counts,
//                                      lengths, fill_value);
// #else
//     AT_ERROR("Not compiled with CUDA support");
// #endif
//   } else {
//     AT_ERROR("Not implemented yet");
//   }
// }

static auto registry =
    torch::RegisterOperators()
        .op("torch_sparse::bin_assignment", &bin_assignment)
        .op("torch_sparse::padded_index_select", &padded_index_select);
// .op("torch_sparse::padded_index_select2", &padded_index_select2);

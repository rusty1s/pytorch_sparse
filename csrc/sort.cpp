#include <Python.h>
#include <torch/script.h>

#include "cpu/sort_cpu.h"

#ifdef _WIN32
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__sort_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__sort_cpu(void) { return NULL; }
#endif
#endif

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sort(const torch::Tensor &row, const torch::Tensor &col, const int64_t num_rows,
     const bool compressed) {
  if (row.device().is_cuda()) {
#ifdef WITH_CUDA
    AT_ERROR("No CUDA version supported");
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return sort_cpu(row, col, num_rows, compressed);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_sparse::sort", &sort);

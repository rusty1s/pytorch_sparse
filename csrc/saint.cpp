#include <Python.h>
#include <torch/script.h>

#include "cpu/saint_cpu.h"

#ifdef _WIN32
PyMODINIT_FUNC PyInit__saint(void) { return NULL; }
#endif

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
subgraph(torch::Tensor idx, torch::Tensor rowptr, torch::Tensor row,
         torch::Tensor col) {
  if (idx.device().is_cuda()) {
#ifdef WITH_CUDA
    AT_ERROR("No CUDA version supported");
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return subgraph_cpu(idx, rowptr, row, col);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_sparse::saint_subgraph", &subgraph);

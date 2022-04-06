#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>

#include "cpu/saint_cpu.h"

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__saint_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__saint_cpu(void) { return NULL; }
#endif
#endif
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

#include <Python.h>
#include <torch/script.h>

#include "cpu/relabel_cpu.h"

#ifdef _WIN32
PyMODINIT_FUNC PyInit__relabel(void) { return NULL; }
#endif

std::tuple<torch::Tensor, torch::Tensor> relabel(torch::Tensor col,
                                                 torch::Tensor idx) {
  if (col.device().is_cuda()) {
#ifdef WITH_CUDA
    AT_ERROR("No CUDA version supported");
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return relabel_cpu(col, idx);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_sparse::relabel", &relabel);

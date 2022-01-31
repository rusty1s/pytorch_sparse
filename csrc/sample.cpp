#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>

#include "cpu/sample_cpu.h"

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__sample_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__sample_cpu(void) { return NULL; }
#endif
#endif
#endif

SPARSE_API std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sample_adj(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
           int64_t num_neighbors, bool replace) {
  if (rowptr.device().is_cuda()) {
#ifdef WITH_CUDA
    AT_ERROR("No CUDA version supported");
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return sample_adj_cpu(rowptr, col, idx, num_neighbors, replace);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_sparse::sample_adj", &sample_adj);

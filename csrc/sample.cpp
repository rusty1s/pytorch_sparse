#include <Python.h>
#include <torch/script.h>

#include "cpu/sample_cpu.h"

#ifdef _WIN32
PyMODINIT_FUNC PyInit__sample(void) { return NULL; }
#endif

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sample_adj(torch::Tensor rowptr, torch::Tensor col, torch::Tensor rowcount,
           torch::Tensor idx, int64_t num_neighbors, bool replace) {
  if (rowptr.device().is_cuda()) {
#ifdef WITH_CUDA
    AT_ERROR("No CUDA version supported");
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return sample_adj_cpu(rowptr, col, rowcount, idx, num_neighbors, replace);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_sparse::sample_adj", &sample_adj);

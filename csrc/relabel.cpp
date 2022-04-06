#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>

#include "cpu/relabel_cpu.h"

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__relabel_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__relabel_cpu(void) { return NULL; }
#endif
#endif
#endif

SPARSE_API std::tuple<torch::Tensor, torch::Tensor> relabel(torch::Tensor col,
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

SPARSE_API std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>,
           torch::Tensor>
relabel_one_hop(torch::Tensor rowptr, torch::Tensor col,
                torch::optional<torch::Tensor> optional_value,
                torch::Tensor idx, bool bipartite) {
  if (rowptr.device().is_cuda()) {
#ifdef WITH_CUDA
    AT_ERROR("No CUDA version supported");
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return relabel_one_hop_cpu(rowptr, col, optional_value, idx, bipartite);
  }
}

static auto registry =
    torch::RegisterOperators()
        .op("torch_sparse::relabel", &relabel)
        .op("torch_sparse::relabel_one_hop", &relabel_one_hop);

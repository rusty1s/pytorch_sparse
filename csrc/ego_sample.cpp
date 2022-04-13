#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>

#include "cpu/ego_sample_cpu.h"

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__ego_sample_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__ego_sample_cpu(void) { return NULL; }
#endif
#endif
#endif

// Returns `rowptr`, `col`, `n_id`, `e_id`, `ptr`, `root_n_id`
SPARSE_API std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
ego_k_hop_sample_adj(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
                     int64_t depth, int64_t num_neighbors, bool replace) {
  if (rowptr.device().is_cuda()) {
#ifdef WITH_CUDA
    AT_ERROR("No CUDA version supported");
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return ego_k_hop_sample_adj_cpu(rowptr, col, idx, depth, num_neighbors,
                                    replace);
  }
}

static auto registry = torch::RegisterOperators().op(
    "torch_sparse::ego_k_hop_sample_adj", &ego_k_hop_sample_adj);

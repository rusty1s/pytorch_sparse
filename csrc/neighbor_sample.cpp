#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>

#include "cpu/neighbor_sample_cpu.h"

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__neighbor_sample_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__neighbor_sample_cpu(void) { return NULL; }
#endif
#endif
#endif

// Returns 'output_node', 'row', 'col', 'output_edge'
SPARSE_API std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
neighbor_sample(const torch::Tensor &colptr, const torch::Tensor &row,
                const torch::Tensor &input_node,
                const std::vector<int64_t> num_neighbors, const bool replace,
                const bool directed) {
  return neighbor_sample_cpu(colptr, row, input_node, num_neighbors, replace,
                             directed);
}

SPARSE_API std::tuple<c10::Dict<node_t, torch::Tensor>, c10::Dict<rel_t, torch::Tensor>,
           c10::Dict<rel_t, torch::Tensor>, c10::Dict<rel_t, torch::Tensor>>
hetero_neighbor_sample(
    const std::vector<node_t> &node_types,
    const std::vector<edge_t> &edge_types,
    const c10::Dict<rel_t, torch::Tensor> &colptr_dict,
    const c10::Dict<rel_t, torch::Tensor> &row_dict,
    const c10::Dict<node_t, torch::Tensor> &input_node_dict,
    const c10::Dict<rel_t, std::vector<int64_t>> &num_neighbors_dict,
    const int64_t num_hops, const bool replace, const bool directed) {
  return hetero_neighbor_sample_cpu(
      node_types, edge_types, colptr_dict, row_dict, input_node_dict,
      num_neighbors_dict, num_hops, replace, directed);
}

static auto registry =
    torch::RegisterOperators()
        .op("torch_sparse::neighbor_sample", &neighbor_sample)
        .op("torch_sparse::hetero_neighbor_sample", &hetero_neighbor_sample);

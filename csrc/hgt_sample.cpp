#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>

#include "cpu/hgt_sample_cpu.h"

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__hgt_sample_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__hgt_sample_cpu(void) { return NULL; }
#endif
#endif
#endif

// Returns 'output_node_dict', 'row_dict', 'col_dict', 'output_edge_dict'
SPARSE_API std::tuple<c10::Dict<node_t, torch::Tensor>, c10::Dict<rel_t, torch::Tensor>,
           c10::Dict<rel_t, torch::Tensor>, c10::Dict<rel_t, torch::Tensor>>
hgt_sample(const c10::Dict<std::string, torch::Tensor> &colptr_dict,
           const c10::Dict<std::string, torch::Tensor> &row_dict,
           const c10::Dict<std::string, torch::Tensor> &input_node_dict,
           const c10::Dict<std::string, std::vector<int64_t>> &num_samples_dict,
           const int64_t num_hops) {

  return hgt_sample_cpu(colptr_dict, row_dict, input_node_dict,
                        num_samples_dict, num_hops);
}

static auto registry =
    torch::RegisterOperators().op("torch_sparse::hgt_sample", &hgt_sample);

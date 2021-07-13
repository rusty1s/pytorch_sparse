#include <Python.h>
#include <torch/script.h>

#include "cpu/hgt_sample_cpu.h"

#ifdef _WIN32
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__hgt_sample_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__hgt_sample_cpu(void) { return NULL; }
#endif
#endif

std::tuple<c10::Dict<node_t, torch::Tensor>, c10::Dict<rel_t, torch::Tensor>,
           c10::Dict<rel_t, torch::Tensor>, c10::Dict<rel_t, torch::Tensor>>
hgt_sample(const c10::Dict<std::string, torch::Tensor> &rowptr_dict,
           const c10::Dict<std::string, torch::Tensor> &col_dict,
           const c10::Dict<std::string, torch::Tensor> &input_node_dict,
           const c10::Dict<std::string, std::vector<int64_t>> &num_samples_dict,
           const int64_t num_hops) {

  return hgt_sample_cpu(rowptr_dict, col_dict, input_node_dict,
                        num_samples_dict, num_hops);
}

static auto registry =
    torch::RegisterOperators().op("torch_sparse::hgt_sample", &hgt_sample);

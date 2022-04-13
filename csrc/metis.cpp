#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>

#include "cpu/metis_cpu.h"

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__metis_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__metis_cpu(void) { return NULL; }
#endif
#endif
#endif

SPARSE_API torch::Tensor partition(torch::Tensor rowptr, torch::Tensor col,
                        torch::optional<torch::Tensor> optional_value,
                        int64_t num_parts, bool recursive) {
  if (rowptr.device().is_cuda()) {
#ifdef WITH_CUDA
    AT_ERROR("No CUDA version supported");
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return partition_cpu(rowptr, col, optional_value, torch::nullopt, num_parts,
                         recursive);
  }
}

SPARSE_API torch::Tensor partition2(torch::Tensor rowptr, torch::Tensor col,
                         torch::optional<torch::Tensor> optional_value,
                         torch::optional<torch::Tensor> optional_node_weight,
                         int64_t num_parts, bool recursive) {
  if (rowptr.device().is_cuda()) {
#ifdef WITH_CUDA
    AT_ERROR("No CUDA version supported");
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return partition_cpu(rowptr, col, optional_value, optional_node_weight,
                         num_parts, recursive);
  }
}

SPARSE_API torch::Tensor mt_partition(torch::Tensor rowptr, torch::Tensor col,
                           torch::optional<torch::Tensor> optional_value,
                           torch::optional<torch::Tensor> optional_node_weight,
                           int64_t num_parts, bool recursive,
                           int64_t num_workers) {
  if (rowptr.device().is_cuda()) {
#ifdef WITH_CUDA
    AT_ERROR("No CUDA version supported");
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return mt_partition_cpu(rowptr, col, optional_value, optional_node_weight,
                            num_parts, recursive, num_workers);
  }
}

static auto registry = torch::RegisterOperators()
                           .op("torch_sparse::partition", &partition)
                           .op("torch_sparse::partition2", &partition2)
                           .op("torch_sparse::mt_partition", &mt_partition);

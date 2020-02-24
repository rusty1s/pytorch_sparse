#include <Python.h>
#include <torch/script.h>

#include "cpu/metis_wrapper_cpu.h"

#include <metis.h>

#ifdef _WIN32
PyMODINIT_FUNC PyInit__metis_wrapper(void) { return NULL; }
#endif

torch::Tensor partition_kway(torch::Tensor rowptr, torch::Tensor col,
                             int64_t num_parts) {
  if (rowptr.device().is_cuda()) {
#ifdef WITH_CUDA
    AT_ERROR("No CUDA version supported");
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return partition_kway_cpu(rowptr, col, num_parts);
  }
}

static auto registry = torch::RegisterOperators().op(
    "torch_sparse::partition_kway", &partition_kway);

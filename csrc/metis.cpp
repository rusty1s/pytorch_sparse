#include <Python.h>
#include <torch/script.h>

#include "cpu/metis_cpu.h"

#ifdef _WIN32
PyMODINIT_FUNC PyInit__metis(void) { return NULL; }
#endif

torch::Tensor partition(torch::Tensor rowptr, torch::Tensor col,
                        int64_t num_parts, bool recursive) {
  if (rowptr.device().is_cuda()) {
#ifdef WITH_CUDA
    AT_ERROR("No CUDA version supported");
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return partition_cpu(rowptr, col, num_parts, recursive);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_sparse::partition", &partition);

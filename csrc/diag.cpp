#include <Python.h>
#include <torch/script.h>

#include "cpu/diag_cpu.h"

#ifdef WITH_CUDA
#include "cuda/diag_cuda.h"
#endif

#ifdef _WIN32
PyMODINIT_FUNC PyInit__diag(void) { return NULL; }
#endif

torch::Tensor non_diag_mask(torch::Tensor row, torch::Tensor col, int64_t M,
                            int64_t N, int64_t k) {
  if (row.device().is_cuda()) {
#ifdef WITH_CUDA
    return non_diag_mask_cuda(row, col, M, N, k);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return non_diag_mask_cpu(row, col, M, N, k);
  }
}

static auto registry = torch::RegisterOperators().op(
    "torch_sparse::non_diag_mask", &non_diag_mask);

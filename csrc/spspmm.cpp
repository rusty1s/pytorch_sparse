#include <Python.h>
#include <torch/script.h>

#include "cpu/spspmm_cpu.h"

#ifdef WITH_CUDA
#include "cuda/spspmm_cuda.h"
#endif

#ifdef _WIN32
PyMODINIT_FUNC PyInit__spspmm(void) { return NULL; }
#endif

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
spspmm_sum(torch::Tensor rowptrA, torch::Tensor colA,
           torch::optional<torch::Tensor> optional_valueA,
           torch::Tensor rowptrB, torch::Tensor colB,
           torch::optional<torch::Tensor> optional_valueB, int64_t K) {
  if (rowptrA.device().is_cuda()) {
#ifdef WITH_CUDA
    return spspmm_cuda(rowptrA, colA, optional_valueA, rowptrB, colB,
                       optional_valueB, K, "sum");
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return spspmm_cpu(rowptrA, colA, optional_valueA, rowptrB, colB,
                      optional_valueB, K, "sum");
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_sparse::spspmm_sum", &spspmm_sum);

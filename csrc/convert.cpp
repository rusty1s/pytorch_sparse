#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>

#include "cpu/convert_cpu.h"

#ifdef WITH_CUDA
#include "cuda/convert_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__convert_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__convert_cpu(void) { return NULL; }
#endif
#endif
#endif

SPARSE_API torch::Tensor ind2ptr(torch::Tensor ind, int64_t M) {
  if (ind.device().is_cuda()) {
#ifdef WITH_CUDA
    return ind2ptr_cuda(ind, M);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return ind2ptr_cpu(ind, M);
  }
}

SPARSE_API torch::Tensor ptr2ind(torch::Tensor ptr, int64_t E) {
  if (ptr.device().is_cuda()) {
#ifdef WITH_CUDA
    return ptr2ind_cuda(ptr, E);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return ptr2ind_cpu(ptr, E);
  }
}

static auto registry = torch::RegisterOperators()
                           .op("torch_sparse::ind2ptr", &ind2ptr)
                           .op("torch_sparse::ptr2ind", &ptr2ind);

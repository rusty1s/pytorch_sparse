#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>
#include "sparse.h"

#ifdef WITH_CUDA
#include <cuda.h>
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__version_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__version_cpu(void) { return NULL; }
#endif
#endif
#endif

SPARSE_API int64_t cuda_version() {
#ifdef WITH_CUDA
  return CUDA_VERSION;
#else
  return -1;
#endif
}

static auto registry =
    torch::RegisterOperators().op("torch_sparse::cuda_version", &cuda_version);

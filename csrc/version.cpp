#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include "sparse.h"
#include <torch/library.h>

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

namespace sparse {
SPARSE_API int64_t cuda_version() noexcept {
#ifdef WITH_CUDA
  return CUDA_VERSION;
#else
  return -1;
#endif
}

TORCH_LIBRARY_FRAGMENT(torch_sparse, m) {
  m.def("_cuda_version", &cuda_version);
}

} // namespace sparse

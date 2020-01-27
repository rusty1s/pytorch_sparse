#include <torch/script.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")

torch::Tensor ind2ptr_cuda(torch::Tensor ind, int64_t M);
torch::Tensor ptr2ind_cuda(torch::Tensor ptr, int64_t E);

torch::Tensor ind2ptr(torch::Tensor ind, int64_t M) {
  CHECK_CUDA(ind);
  return ind2ptr_cuda(ind, M);
}

torch::Tensor ptr2ind(torch::Tensor ptr, int64_t E) {
  CHECK_CUDA(ptr);
  return ptr2ind_cuda(ptr, E);
}

static auto registry =
    torch::RegisterOperators("torch_sparse_cuda::ind2ptr", &ind2ptr)
        .op("torch_sparse_cuda::ptr2ind", &ptr2ind);

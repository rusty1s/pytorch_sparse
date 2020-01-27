#include <torch/script.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")

torch::Tensor non_diag_mask_cuda(torch::Tensor row, torch::Tensor col,
                                 int64_t M, int64_t N, int64_t k);

torch::Tensor non_diag_mask(torch::Tensor row, torch::Tensor col, int64_t M,
                            int64_t N, int64_t k) {
  CHECK_CUDA(row);
  CHECK_CUDA(col);
  return non_diag_mask_cuda(row, col, M, N, k);
}

static auto registry = torch::RegisterOperators(
    "torch_sparse_cuda::non_diag_mask", &non_diag_mask);

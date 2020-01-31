#include <torch/script.h>

#define CHECK_CUDA(x)                                                          \
  AT_ASSERTM(x.device().is_cuda(), #x " must be CUDA tensor")

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
spspmm_cuda(torch::Tensor rowptrA, torch::Tensor colA,
            torch::optional<torch::Tensor> valueA, torch::Tensor rowptrB,
            torch::Tensor colB, torch::optional<torch::Tensor> valueB,
            int64_t M, int64_t N, int64_t K);

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
spspmm(torch::Tensor rowptrA, torch::Tensor colA,
       torch::optional<torch::Tensor> valueA, torch::Tensor rowptrB,
       torch::Tensor colB, torch::optional<torch::Tensor> valueB, int64_t M,
       int64_t N, int64_t K) {
  CHECK_CUDA(rowptrA);
  CHECK_CUDA(colA);
  if (valueA.has_value())
    CHECK_CUDA(valueA.value());
  CHECK_CUDA(rowptrB);
  CHECK_CUDA(colB);
  if (valueB.has_value())
    CHECK_CUDA(valueB.value());
  return spspmm_cuda(rowptrA, colA, valueA, rowptrB, colB, valueB, M, N, K);
}

static auto registry =
    torch::RegisterOperators("torch_sparse_cuda::spspmm", &spspmm);

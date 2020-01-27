#include <torch/script.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_cuda(torch::Tensor rowptr, torch::Tensor col,
          torch::optional<torch::Tensor> value_opt, torch::Tensor mat,
          std::string reduce);

torch::Tensor spmm_val_bw_cuda(torch::Tensor row, torch::Tensor rowptr,
                               torch::Tensor col, torch::Tensor mat,
                               torch::Tensor grad, std::string reduce);

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm(torch::Tensor rowptr, torch::Tensor col,
     torch::optional<torch::Tensor> value_opt, torch::Tensor mat,
     std::string reduce) {
  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  if (value_opt.has_value())
    CHECK_CUDA(value_opt.value());
  CHECK_CUDA(mat);
  return spmm_cuda(rowptr, col, value_opt, mat, reduce);
}

torch::Tensor spmm_val_bw(torch::Tensor row, torch::Tensor rowptr,
                          torch::Tensor col, torch::Tensor mat,
                          torch::Tensor grad, std::string reduce) {
  CHECK_CUDA(row);
  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  CHECK_CUDA(mat);
  CHECK_CUDA(grad);
  return spmm_val_bw_cuda(row, rowptr, col, mat, grad, reduce);
}

static auto registry =
    torch::RegisterOperators("torch_sparse_cuda::spmm", &spmm)
        .op("torch_sparse_cuda::spmm_val_bw", &spmm_val_bw);

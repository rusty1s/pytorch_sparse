#include "spmm_cuda.h"

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_cuda(torch::Tensor rowptr, torch::Tensor col,
          torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
          std::string reduce) {
  return std::make_tuple(mat, optional_value);
}

torch::Tensor spmm_value_bw_cuda(torch::Tensor row, torch::Tensor rowptr,
                                 torch::Tensor col, torch::Tensor mat,
                                 torch::Tensor grad, std::string reduce) {
  return row;
}

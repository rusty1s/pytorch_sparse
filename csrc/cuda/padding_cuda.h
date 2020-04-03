#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
padded_index_cuda(torch::Tensor rowptr, torch::Tensor rowcount,
                  torch::Tensor binptr);

torch::Tensor padded_index_select_cuda(torch::Tensor src, torch::Tensor col,
                                       torch::Tensor index,
                                       torch::Tensor fill_value);

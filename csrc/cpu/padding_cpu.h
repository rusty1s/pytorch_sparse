#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<int64_t>, std::vector<int64_t>>
padded_index_cpu(torch::Tensor rowptr, torch::Tensor col,
                 torch::Tensor rowcount, torch::Tensor binptr);

torch::Tensor padded_index_select_cpu(torch::Tensor src, torch::Tensor index,
                                      torch::Tensor fill_value);

torch::Tensor padded_index_scatter_cpu(torch::Tensor src, torch::Tensor index,
                                       int64_t N);

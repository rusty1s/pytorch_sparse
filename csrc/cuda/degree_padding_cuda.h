#pragma once

#include <torch/extension.h>

std::vector<torch::Tensor> bin_assignment_cuda(torch::Tensor rowcount,
                                               torch::Tensor bin_strategy);
std::tuple<torch::Tensor, torch::Tensor>
padded_index_select_cuda(torch::Tensor src, torch::Tensor rowptr,
                         torch::Tensor col, torch::Tensor index,
                         int64_t length);

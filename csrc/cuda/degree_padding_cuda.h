#pragma once

#include <torch/extension.h>

std::tuple<std::vector<torch::Tensor>, std::vector<int64_t>>
bin_assignment_cuda(torch::Tensor rowcount, torch::Tensor binptr);

std::tuple<torch::Tensor, torch::Tensor>
padded_index_select_cuda(torch::Tensor src, torch::Tensor rowptr,
                         torch::Tensor col, torch::Tensor index, int64_t length,
                         torch::Tensor fill_value);

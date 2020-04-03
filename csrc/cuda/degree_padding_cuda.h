#pragma once

#include <torch/extension.h>

std::tuple<std::vector<torch::Tensor>, std::vector<int64_t>>
bin_assignment_cuda(torch::Tensor rowcount, torch::Tensor binptr);

std::tuple<torch::Tensor, torch::Tensor>
padded_index_select_cuda(torch::Tensor src, torch::Tensor rowptr,
                         torch::Tensor col, torch::Tensor index, int64_t length,
                         torch::Tensor fill_value);

// std::tuple<torch::Tensor, torch::Tensor> padded_index_select_cuda2(
//     torch::Tensor src, torch::Tensor rowptr, torch::Tensor col,
//     torch::Tensor bin, torch::Tensor index, std::vector<int64_t> node_counts,
//     std::vector<int64_t> lengths, torch::Tensor fill_value);

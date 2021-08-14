#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sort_cpu(const torch::Tensor &row, const torch::Tensor &col,
         const int64_t num_rows, const bool compressed);

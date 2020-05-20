#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sample_adj_cpu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor rowcount,
               torch::Tensor idx, int64_t num_neighbors, bool replace);

#pragma once

#include "../extensions.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
ego_k_hop_sample_adj_cpu(torch::Tensor rowptr, torch::Tensor col,
                         torch::Tensor idx, int64_t depth,
                         int64_t num_neighbors, bool replace);

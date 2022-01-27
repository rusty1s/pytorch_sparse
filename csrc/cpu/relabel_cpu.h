#pragma once

#include "../extensions.h"

std::tuple<torch::Tensor, torch::Tensor> relabel_cpu(torch::Tensor col,
                                                     torch::Tensor idx);

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>,
           torch::Tensor>
relabel_one_hop_cpu(torch::Tensor rowptr, torch::Tensor col,
                    torch::optional<torch::Tensor> optional_value,
                    torch::Tensor idx, bool bipartite);

#pragma once

#include "../extensions.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
subgraph_cpu(torch::Tensor idx, torch::Tensor rowptr, torch::Tensor row,
             torch::Tensor col);

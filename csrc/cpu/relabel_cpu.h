#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> relabel_cpu(torch::Tensor col,
                                                     torch::Tensor idx);

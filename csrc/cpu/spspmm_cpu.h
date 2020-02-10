#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
spspmm_cpu(torch::Tensor rowptrA, torch::Tensor colA,
           torch::optional<torch::Tensor> optional_valueA,
           torch::Tensor rowptrB, torch::Tensor colB,
           torch::optional<torch::Tensor> optional_valueB, int64_t K,
           std::string reduce);

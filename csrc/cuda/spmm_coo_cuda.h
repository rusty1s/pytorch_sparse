#pragma once

#include "../extensions.h"

std::tuple<torch::Tensor, torch::optional<torch::Tensor>> spmm_coo_cuda(
    const torch::Tensor row,
    const torch::Tensor col,
    const torch::optional<torch::Tensor> optional_value,
    torch::Tensor mat,
    int64_t dim_size,
    std::string reduce);

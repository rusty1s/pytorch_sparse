#pragma once

#include "../extensions.h"

std::tuple<torch::Tensor, torch::Tensor> spmm_coo_cuda(
    const torch::Tensor row,
    const torch::Tensor col,
    torch::Tensor mat,
    int64_t dim_size,
    std::string reduce);

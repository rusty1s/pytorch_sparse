#pragma once

#include "../extensions.h"

std::tuple<torch::Tensor, torch::Tensor> spmm_coo_cuda(
    torch::Tensor src,
    const torch::Tensor edge_start,
    const torch::Tensor edge_end,
    int64_t res_dim,
    std::string reduce);

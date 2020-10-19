#pragma once

#include <torch/extension.h>

int64_t cuda_version();

torch::Tensor ind2ptr(torch::Tensor ind, int64_t M);
torch::Tensor ptr2ind(torch::Tensor ptr, int64_t E);

torch::Tensor partition(torch::Tensor rowptr, torch::Tensor col,
                        torch::optional<torch::Tensor> optional_value,
                        int64_t num_parts, bool recursive);

std::tuple<torch::Tensor, torch::Tensor> relabel(torch::Tensor col,
                                                 torch::Tensor idx);

torch::Tensor random_walk(torch::Tensor rowptr, torch::Tensor col,
                          torch::Tensor start, int64_t walk_length);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
subgraph(torch::Tensor idx, torch::Tensor rowptr, torch::Tensor row,
         torch::Tensor col);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sample_adj(torch::Tensor rowptr, torch::Tensor col, torch::Tensor rowcount,
           torch::Tensor idx, int64_t num_neighbors, bool replace);

torch::Tensor spmm_sum(torch::optional<torch::Tensor> opt_row,
                       torch::Tensor rowptr, torch::Tensor col,
                       torch::optional<torch::Tensor> opt_value,
                       torch::optional<torch::Tensor> opt_colptr,
                       torch::optional<torch::Tensor> opt_csr2csc,
                       torch::Tensor mat);

torch::Tensor spmm_mean(torch::optional<torch::Tensor> opt_row,
                        torch::Tensor rowptr, torch::Tensor col,
                        torch::optional<torch::Tensor> opt_value,
                        torch::optional<torch::Tensor> opt_rowcount,
                        torch::optional<torch::Tensor> opt_colptr,
                        torch::optional<torch::Tensor> opt_csr2csc,
                        torch::Tensor mat);

std::tuple<torch::Tensor, torch::Tensor>
spmm_min(torch::Tensor rowptr, torch::Tensor col,
         torch::optional<torch::Tensor> opt_value, torch::Tensor mat);

std::tuple<torch::Tensor, torch::Tensor>
spmm_max(torch::Tensor rowptr, torch::Tensor col,
         torch::optional<torch::Tensor> opt_value, torch::Tensor mat);

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
spspmm_sum(torch::Tensor rowptrA, torch::Tensor colA,
           torch::optional<torch::Tensor> optional_valueA,
           torch::Tensor rowptrB, torch::Tensor colB,
           torch::optional<torch::Tensor> optional_valueB, int64_t K);

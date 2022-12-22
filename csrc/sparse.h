#pragma once

#include "extensions.h"
#include "macros.h"

namespace sparse {
SPARSE_API int64_t cuda_version() noexcept;

namespace detail {
SPARSE_INLINE_VARIABLE int64_t _cuda_version = cuda_version();
} // namespace detail
} // namespace sparse

SPARSE_API torch::Tensor ind2ptr(torch::Tensor ind, int64_t M);
SPARSE_API torch::Tensor ptr2ind(torch::Tensor ptr, int64_t E);

SPARSE_API torch::Tensor
partition(torch::Tensor rowptr, torch::Tensor col,
          torch::optional<torch::Tensor> optional_value, int64_t num_parts,
          bool recursive);

SPARSE_API torch::Tensor
partition2(torch::Tensor rowptr, torch::Tensor col,
           torch::optional<torch::Tensor> optional_value,
           torch::optional<torch::Tensor> optional_node_weight,
           int64_t num_parts, bool recursive);

SPARSE_API torch::Tensor
mt_partition(torch::Tensor rowptr, torch::Tensor col,
             torch::optional<torch::Tensor> optional_value,
             torch::optional<torch::Tensor> optional_node_weight,
             int64_t num_parts, bool recursive, int64_t num_workers);

SPARSE_API std::tuple<torch::Tensor, torch::Tensor> relabel(torch::Tensor col,
                                                            torch::Tensor idx);

SPARSE_API std::tuple<torch::Tensor, torch::Tensor,
                      torch::optional<torch::Tensor>, torch::Tensor>
relabel_one_hop(torch::Tensor rowptr, torch::Tensor col,
                torch::optional<torch::Tensor> optional_value,
                torch::Tensor idx, bool bipartite);

SPARSE_API torch::Tensor random_walk(torch::Tensor rowptr, torch::Tensor col,
                                     torch::Tensor start, int64_t walk_length);

SPARSE_API std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
subgraph(torch::Tensor idx, torch::Tensor rowptr, torch::Tensor row,
         torch::Tensor col);

SPARSE_API
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sample_adj(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
           int64_t num_neighbors, bool replace);

SPARSE_API torch::Tensor spmm_sum(torch::optional<torch::Tensor> opt_row,
                                  torch::Tensor rowptr, torch::Tensor col,
                                  torch::optional<torch::Tensor> opt_value,
                                  torch::optional<torch::Tensor> opt_colptr,
                                  torch::optional<torch::Tensor> opt_csr2csc,
                                  torch::Tensor mat);

SPARSE_API torch::Tensor spmm_mean(torch::optional<torch::Tensor> opt_row,
                                   torch::Tensor rowptr, torch::Tensor col,
                                   torch::optional<torch::Tensor> opt_value,
                                   torch::optional<torch::Tensor> opt_rowcount,
                                   torch::optional<torch::Tensor> opt_colptr,
                                   torch::optional<torch::Tensor> opt_csr2csc,
                                   torch::Tensor mat);

SPARSE_API std::tuple<torch::Tensor, torch::Tensor>
spmm_min(torch::Tensor rowptr, torch::Tensor col,
         torch::optional<torch::Tensor> opt_value, torch::Tensor mat);

SPARSE_API std::tuple<torch::Tensor, torch::Tensor>
spmm_max(torch::Tensor rowptr, torch::Tensor col,
         torch::optional<torch::Tensor> opt_value, torch::Tensor mat);

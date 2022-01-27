#pragma once

#include "../extensions.h"

typedef std::string node_t;
typedef std::tuple<std::string, std::string, std::string> edge_t;
typedef std::string rel_t;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
neighbor_sample_cpu(const torch::Tensor &colptr, const torch::Tensor &row,
                    const torch::Tensor &input_node,
                    const std::vector<int64_t> num_neighbors,
                    const bool replace, const bool directed);

std::tuple<c10::Dict<node_t, torch::Tensor>, c10::Dict<rel_t, torch::Tensor>,
           c10::Dict<rel_t, torch::Tensor>, c10::Dict<rel_t, torch::Tensor>>
hetero_neighbor_sample_cpu(
    const std::vector<node_t> &node_types,
    const std::vector<edge_t> &edge_types,
    const c10::Dict<rel_t, torch::Tensor> &colptr_dict,
    const c10::Dict<rel_t, torch::Tensor> &row_dict,
    const c10::Dict<node_t, torch::Tensor> &input_node_dict,
    const c10::Dict<rel_t, std::vector<int64_t>> &num_neighbors_dict,
    const int64_t num_hops, const bool replace, const bool directed);

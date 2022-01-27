#pragma once

#include "../extensions.h"

typedef std::string node_t;
typedef std::string rel_t;
typedef std::tuple<std::string, std::string, std::string> edge_t;

std::tuple<c10::Dict<node_t, torch::Tensor>, c10::Dict<rel_t, torch::Tensor>,
           c10::Dict<rel_t, torch::Tensor>, c10::Dict<rel_t, torch::Tensor>>
hgt_sample_cpu(const c10::Dict<rel_t, torch::Tensor> &colptr_dict,
               const c10::Dict<rel_t, torch::Tensor> &row_dict,
               const c10::Dict<node_t, torch::Tensor> &input_node_dict,
               const c10::Dict<node_t, std::vector<int64_t>> &num_samples_dict,
               const int64_t num_hops);

#pragma once

#include <torch/extension.h>

// Node type is a string and the edge type is a triplet of string representing 
// (source_node_type, relation_type, dest_node_type).
typedef std::string node_t;
typedef std::tuple<std::string, std::string, std::string> edge_t;

// As of PyTorch 1.9.0, c10::Dict does not support tuples or complex data type as key type. We work around this
// by representing edge types using a single int64_t and a c10::Dict that maps the int64_t index to edge_t.
void hg_sample_cpu(
	const c10::Dict<int64_t, torch::Tensor> &rowptr_store,
	const c10::Dict<int64_t, torch::Tensor> &col_store,  
	const c10::Dict<node_t, torch::Tensor> &origin_nodes_store,
	const c10::Dict<int64_t, edge_t> &edge_type_idx_to_name,
	int n,
	int num_layers
);

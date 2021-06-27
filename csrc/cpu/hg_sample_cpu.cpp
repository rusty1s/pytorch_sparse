#include "hg_sample_cpu.h"

#include <unordered_map>
#include <random>
#include "utils.h"

namespace {

void update_budget(
	int64_t added_node_idx,
	const node_t &added_node_type,
	const c10::Dict<int64_t, torch::Tensor> &rowptr_store,
	const c10::Dict<int64_t, torch::Tensor> &col_store,
	const c10::Dict<int64_t, edge_t> &edge_type_idx_to_name,
	std::unordered_map<node_t, std::unordered_set<int64_t>> &sampled_nodes_store,
	std::unordered_map<node_t, std::unordered_map<int64_t, float>> *budget_store
) {
	for (const auto &i : rowptr_store) {
		const auto &edge_type_idx = i.key();
		const auto &edge_type = edge_type_idx_to_name.at(edge_type_idx);
		const auto &source_node_type = std::get<0>(edge_type);
		const auto &dest_node_type = std::get<2>(edge_type);

		// Skip processing the (rowptr, col) if the node types do not match
		if (added_node_type.compare(dest_node_type) != 0) {
			continue;
		}

		int64_t *row_ptr_raw = i.value().data_ptr<int64_t>();
		int64_t *col_raw = col_store.at(edge_type_idx).data_ptr<int64_t>();

		// Get the budget map and sampled_nodes for the source node type of the relation
		const std::unordered_set<int64_t> &sampled_nodes = sampled_nodes_store[source_node_type];
		std::unordered_map<int64_t, float> &budget = (*budget_store)[source_node_type];

		int64_t row_start_idx = row_ptr_raw[added_node_idx];
		int64_t row_end_idx = row_ptr_raw[added_node_idx + 1];
		if (row_start_idx != row_end_idx) {
			// Compute the norm of degree and update the budget for the neighbors of added_node_idx
			float norm_deg = 1. / (float)(row_end_idx - row_start_idx);
			for (int64_t j = row_start_idx; j < row_end_idx; j++) {
				if (sampled_nodes.find(col_raw[j]) == sampled_nodes.end()) {
					budget[col_raw[j]] += norm_deg;
				}
			}
		}
	}
}

// Sample n nodes according to its type budget map. The probability that node i is sampled is calculated by
// prob[i] = budget[i]^2 / l2_norm(budget)^2.
std::unordered_set<int64_t> sample_nodes(const std::unordered_map<int64_t, float> &budget, int n) {	
	// Compute the squared L2 norm	
	float norm = 0.0;
	for (const auto &i : budget) {
		norm += i.second * i.second;	
	}

	// Generate n sorted random values between 0 and norm
	std::vector<float> samples(n);
	std::uniform_real_distribution<float> dist(0.0, norm);
	std::default_random_engine gen{std::random_device{}()};
	std::generate(std::begin(samples), std::end(samples), [&]{ return dist(gen); });
	std::sort(samples.begin(), samples.end());

	// Iterate through the budget map to compute the cumulative probability cum_prob[i] for node_i. The j-th
	// sample is assigned to node_i iff cum_prob[i-1] < samples[j] < cum_prob[i]. The implementation assigns
	// two iterators on budget and samples respectively, then computes the node samples in linear time by 
	// alternatingly incrementing the two iterators based on their values.
	std::unordered_set<int64_t> sampled_nodes;
	sampled_nodes.reserve(samples.size());
	auto j = samples.begin();
	float cum_prob = 0.0;
	for (const auto &i : budget) {
		cum_prob += i.second * i.second;

		// Increment iterator j until its value is greater than the current cum_prob
		while (*j < cum_prob && j != samples.end()) {
			sampled_nodes.insert(i.first);
			j++;
		}

		// Terminate early after we complete the sampling
		if (j == samples.end()) {
			break;
		}
	}

	return sampled_nodes;
}

}  // namespace

// TODO: Add the appropriate return type
void hg_sample_cpu(
	const c10::Dict<int64_t, torch::Tensor> &rowptr_store,
	const c10::Dict<int64_t, torch::Tensor> &col_store,  
	const c10::Dict<node_t, torch::Tensor> &origin_nodes_store,
	const c10::Dict<int64_t, edge_t> &edge_type_idx_to_name,
	int n,
	int num_layers
) {
	// Verify input
	for (const auto &kv : rowptr_store) {
		CHECK_CPU(kv.value());
	}
	
	for (const auto &kv : col_store) {
		CHECK_CPU(kv.value());
	}
	
	for (const auto &kv : origin_nodes_store) {
		CHECK_CPU(kv.value());
	  	CHECK_INPUT(kv.value().dim() == 1);
	}
	
	// Initialize various data structures for the sampling process
	std::unordered_map<node_t, std::unordered_set<int64_t>> sampled_nodes_store;
	for (const auto &kv : origin_nodes_store) {
		const auto &node_type = kv.key();
		const auto &origin_nodes = kv.value();
		const int64_t *raw_origin_nodes = origin_nodes.data_ptr<int64_t>();

		// Add each origin node to the sampled_nodes_store
		for (int64_t i = 0; i < origin_nodes.numel(); i++) {
			sampled_nodes_store[node_type].insert(raw_origin_nodes[i]);
		}
	}

	std::unordered_map<node_t, std::unordered_map<int64_t, float>> budget_store;
	for (const auto &kv : origin_nodes_store) {
		const node_t &node_type = kv.key();
		const auto &origin_nodes = kv.value();
		const int64_t *raw_origin_nodes = origin_nodes.data_ptr<int64_t>();

		// Update budget for each origin node
		for (int64_t i = 0; i < origin_nodes.numel(); i++) {
			update_budget(
				raw_origin_nodes[i],
				node_type,
				rowptr_store,
				col_store,
				edge_type_idx_to_name,
				sampled_nodes_store,
				&budget_store
			);
		}
	}


	// Sampling process
	for (int l = 0; l < num_layers; l++) {	
		for (auto &i : budget_store) {
			const auto &node_type = i.first;
			auto &budget = i.second;
			auto &sampled_nodes = sampled_nodes_store[node_type];

			// Perform sampling
			std::unordered_set<int64_t> new_samples = sample_nodes(budget, n);

			// Remove sampled nodes from the budget and add them to the sampled node store
			for (const auto &sample : new_samples) {
				sampled_nodes.insert(sample);
				budget.erase(sample);
			}

			// Update the budget
			for (const auto &sample : new_samples) {
				update_budget(
					sample,
					node_type,
					rowptr_store,
					col_store,
					edge_type_idx_to_name,
					sampled_nodes_store,
					&budget_store
				);
			}
		}
	}

	// Re-index
	c10::Dict<std::string, std::vector<int64_t>> type_to_n_ids;

}

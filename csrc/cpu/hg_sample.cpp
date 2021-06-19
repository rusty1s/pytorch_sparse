#include "hg_sample.h"

#include "utils.h"

// For now, I am assuming that the node type is just a string and the relation type is a 
// triplet of (source_node_type, dest_node_type, relation_type).
typedef std::string node_t;
typedef std::tuple<node_t, node_t, std::string> rel_t;


// TODO: Add the appropriate return type
void hg_sample_cpu(
	const c10::Dict<rel_t, torch::Tensor> &rowptr_store,
	const c10::Dict<rel_t, torch::Tensor> &col_store,
	const c10::Dict<node_t, torch::Tensor> &origin_nodes_store,
	int n,
	int num_layers,
) {
	// Verify input
	for (const auto &kv : rowptr_store) {
		CHECK_CPU(kv.second);
	}
	
	for (const auto &kv : col_store) {
		CHECK_CPU(kv.second);
	}
	
	for (const auto &kv : origin_nodes_store) {
		CHECK_CPU(kv.second);
	  	CHECK_INPUT(kv.second.dim() == 1);
	}
	
	// Initialize various data structures for the sampling process
	c10::Dict<node_t, std::set<int64_t>> sampled_nodes_store;
	for (const auto &kv : origin_nodes_store) {
		const node_t &node_type = kv.first;
		const auto &origin_nodes = kv.second;
		const int64_t *raw_origin_nodes = origin_nodes.data_ptr<int64_t>();

		// Add each origin node to the sampled_nodes_store
		for (int64_t i = 0; i < origin_nodes.numel(); i++) {
			if (sampled_nodes_store.find(node_type) == sampled_nodes_store.end()) {
				sampled_nodes_store.insert(node_type, std::set<int64_t>());
			}
			sampled_nodes_store.at(node_type).add(raw_origin_nodes[i]);
		}
	}

	c10::Dict<node_t, c10::Dict<int64_t, float>> budget_store;
	for (const auto &kv : origin_nodes_store) {
		const node_t &node_type = kv.first;
		const auto &origin_nodes = kv.second;
		const int64_t *raw_origin_nodes = origin_nodes.data_ptr<int64_t>();

		// Update budget for each origin node
		for (int64_t i = 0; i < origin_nodes.numel(); i++) {
			update_budget(
				raw_origin_nodes[i],
				rowptr_store,
				col_store,
				sampled_nodes_store,
				&budget_store
			);
		}
	}


	// Sampling process
	for (int l = 0; l < num_layers; l++) {	
		for (const auto &i : _budget_store) {
			const auto &node_type = i.first;
			auto &per_type_budget = i.second;

			 vector<int64_t> samples = sample_nodes(per_type_budget, n);

			 // Remove sampled nodes from the budget
			 for (const auto &sample : samples) {
			 	per_type_budget.erase(*sample);
			 }

			 type_to_n_ids.insert(node_type, samples);
		}
	}

	// Re-index
	c10::Dict<string, std::vector<int64_t>> type_to_n_ids;

}

void update_budget(
	int64_t added_node_idx,
	const c10::Dict<rel_t, torch::Tensor> &rowptr_store,
	const c10::Dict<rel_t, torch::Tensor> &col_store,
	const c10::Dict<node_t, std::set<int64_t>> &sampled_nodes_store,
	c10::Dict<string, c10::Dict<int64_t, float>> *budget_store,
) {
	for (const auto &i : rowptr_store) {
		const rel_t &relation_type = i.first;
		int64_t *row_ptr_raw = i.second.data_ptr<int64_t>();
		int64_t *col_raw = col_store.at(relation_type).data_ptr<int64_t>();
		
		// Get the budget map and sampled_nodes for the source node type of the relation
		const auto &source_node_type = std::get<0>(relation_type);
		const std::set<int64_t> &sampled_nodes = sampled_nodes_store.at(source_node_type);
		c10::Dict<int64_t, float> *budget = &budget_store->at(source_node_type);
		
		int64_t row_start_idx = row_ptr_raw[added_node_idx];
		int64_t row_end_idx = row_ptr_raw[added_node_idx + 1];
		if (row_start_idx != row_end_idx) {
			// Compute the norm of degree and update the budget for the neighbors of added_node_idx
			double norm_deg = 1 / (double)(row_end_idx - row_start_idx);
			for (int64_t j = row_start_idx; j < row_end_idx; j++) {
				if (sampled_nodes.find(col_raw[j]) == sampled_nodes.end()) {
					const auto &it = budget->find(col_raw[j]);
					float val = it != budget->end() ? it.second : 0.0;
					budget->insert_or_assign(col_raw[j], val + norm_deg);
				}
			}
		}
	}
}

// Sample n nodes according to its type budget map. The probability that node i is sampled is calculated by
// prob[i] = budget[i]^2 / l2_norm(budget)^2.
vector<int64_t> sample_nodes(const c10::Dict<int64_t, float> &budget, int n) {	
	// Compute the squared L2 norm	
	float norm = 0.0;
	for (const auto &i : budget) {
		norm += i.second * i.second;	
	}

	// Generate n sorted random values between 0 and norm
	std::vector<double> samples(n);
	std::uniform_real_distribution<double> dist(0.0, norm);
	std::generate(std::begin(x), std::end(x), [&]{ return dist(gen); });
	std::sort(samples.begin(), samples.end());

	// Iterate through the budget map to compute the cumulative probability cum_prob[i] for node_i. The j-th
	// sample is assigned to node_i iff cum_prob[i-1] < samples[j] < cum_prob[i]. The implementation assigns
	// two iterators on budget and samples respectively, then computes the node samples in linear time by 
	// alternatingly incrementing the two iterators based on their values.
	vector<int64_t> sampled_nodes;
	sampled_nodes.reserve(samples.size());
	const auto &j = samples.begin();
	float cum_prob = 0.0;
	for (const auto &i : budget) {
		cum_prob += i.second * i.second;
		
		// Increment iterator j until its value is greater than the current cum_prob
		while (*j < cum_prob && j != samples.end()) {
			sampled_nodes.append(i.first);
			j++;
		}

		// Terminate early after we complete the sampling
		if (j == samples.end()) {
			break;
		}
	}

	return sampled_nodes;
}
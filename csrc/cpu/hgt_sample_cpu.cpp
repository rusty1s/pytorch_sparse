#include "hgt_sample_cpu.h"

#include <random>

edge_t split(const rel_t &rel_type) {
  std::vector<std::string> result(3);
  int start = 0, end = 0;
  for (int i = 0; i < 3; i++) {
    end = rel_type.find(delim, start);
    result[i] = rel_type.substr(start, end - start);
    start = end + 2;
  }
  return std::make_tuple(result[0], result[1], result[2]);
}

void update_budget(
    std::unordered_map<node_t, std::unordered_map<int64_t, float>> *budget_dict,
    const node_t &node_type, //
    const std::vector<int64_t> &sampled_nodes,
    const std::unordered_map<node_t, std::unordered_map<int64_t, int64_t>>
        &global_to_local_node_dict,
    const std::unordered_map<rel_t, edge_t> &rel_to_edge_type,
    const c10::Dict<rel_t, torch::Tensor> &rowptr_dict,
    const c10::Dict<rel_t, torch::Tensor> &col_dict, //
    const bool remove) {

  for (const auto &kv : rowptr_dict) {
    const auto &rel_type = kv.key();
    const auto &edge_type = rel_to_edge_type.at(rel_type);
    const auto &src_node_type = std::get<0>(edge_type);
    const auto &dst_node_type = std::get<2>(edge_type);

    if (node_type != dst_node_type)
      continue;

    const auto &global_to_local_node =
        global_to_local_node_dict.at(src_node_type);
    const auto *rowptr_data = kv.value().data_ptr<int64_t>();
    const auto *col_data = col_dict.at(rel_type).data_ptr<int64_t>();
    auto &budget = (*budget_dict)[src_node_type];

    for (const auto &v : sampled_nodes) {
      const int64_t row_start = rowptr_data[v], row_end = rowptr_data[v + 1];
      if (row_end != row_start) {
        const auto inv_deg = 1.f / float(row_end - row_start);
        for (int64_t j = row_start; j < row_end; j++) {
          const auto w = col_data[j];
          if (global_to_local_node.find(w) == global_to_local_node.end())
            budget[col_data[j]] += inv_deg;
        }
      }
    }
  }

  if (remove) {
    auto &budget = (*budget_dict)[node_type];
    for (const auto &v : sampled_nodes)
      budget.erase(v);
  }
}

std::unordered_set<int64_t>
sample_from(const std::unordered_map<int64_t, float> &budget,
            const int64_t num_samples) {

  // Compute the squared L2 norm:
  auto norm = 0.f;
  for (const auto &kv : budget)
    norm += kv.second * kv.second;

  // Generate `num_samples` sorted random values between `[0., norm)`:
  std::vector<float> samples(num_samples);
  std::uniform_real_distribution<float> dist(0.f, norm);
  std::default_random_engine gen{std::random_device{}()};
  std::generate(std::begin(samples), std::end(samples),
                [&] { return dist(gen); });
  std::sort(samples.begin(), samples.end());

  // Iterate through the budget to compute the cumulative probability
  // `cum_prob[i]` for node `i`. The j-th sample is assigned to node `i` iff
  // `cum_prob[i-1] < samples[j] < cum_prob[i]`.
  // The implementation assigns two iterators on budget and samples,
  // respectively, and then computes the node samples in linear time by
  // alternatingly incrementing the two iterators based on their values.
  std::unordered_set<int64_t> output;
  output.reserve(num_samples);

  auto j = samples.begin();
  auto cum_prob = 0.f;
  for (const auto &kv : budget) {
    cum_prob += kv.second * kv.second;

    // Increment iterator `j` until its value is greater than `cum_prob`:
    while (*j < cum_prob && j != samples.end()) {
      output.insert(kv.first);
      j++;
    }

    // Terminate early in case we have completed the sampling:
    if (j == samples.end())
      break;
  }

  return output;
}

std::tuple<c10::Dict<node_t, torch::Tensor>, c10::Dict<rel_t, torch::Tensor>,
           c10::Dict<rel_t, torch::Tensor>, c10::Dict<rel_t, torch::Tensor>>
hgt_sample_cpu(const c10::Dict<rel_t, torch::Tensor> &rowptr_dict,
               const c10::Dict<rel_t, torch::Tensor> &col_dict,
               const c10::Dict<node_t, torch::Tensor> &input_node_dict,
               const c10::Dict<node_t, std::vector<int64_t>> &num_samples_dict,
               int64_t num_hops) {

  // Create mapping to convert single string relations to edge type triplets:
  std::unordered_map<rel_t, edge_t> rel_to_edge_type;
  for (const auto &kv : rowptr_dict) {
    const auto &rel_type = kv.key();
    rel_to_edge_type[rel_type] = split(rel_type);
  }

  // Initialize various data structures for the sampling process:
  std::unordered_map<node_t, std::vector<int64_t>> sampled_nodes_dict;
  std::unordered_map<node_t, std::unordered_map<int64_t, int64_t>>
      global_to_local_node_dict;
  std::unordered_map<node_t, std::unordered_map<int64_t, float>> budget_dict;
  for (const auto &kv : num_samples_dict) {
    const auto &node_type = kv.key();
    sampled_nodes_dict[node_type];
    global_to_local_node_dict[node_type];
    budget_dict[node_type];
  }

  // Add all input nodes of every node type to the sampled output set (line 1):
  for (const auto &kv : input_node_dict) {
    const auto &node_type = kv.key();
    const auto &input_node = kv.value();
    const auto *input_node_data = input_node.data_ptr<int64_t>();

    auto &sampled_nodes = sampled_nodes_dict.at(node_type);
    auto &global_to_local_node = global_to_local_node_dict.at(node_type);

    // Add each origin node to the sampled output nodes:
    for (int64_t i = 0; i < input_node.numel(); i++) {
      const auto v = input_node_data[i];
      sampled_nodes.push_back(v);
      global_to_local_node[v] = i;
    }
  }

  // Update budget after all input nodes have been added to the sampled output
  // set (line 2-5):
  for (const auto &kv : sampled_nodes_dict) {
    update_budget(&budget_dict, kv.first, kv.second, global_to_local_node_dict,
                  rel_to_edge_type, rowptr_dict, col_dict, false);
  }

  // Sample nodes for each node type in each layer (line 6 - 18):
  for (int64_t ell = 0; ell < num_hops; ell++) {
    for (auto &kv : budget_dict) {
      const auto &node_type = kv.first;
      auto &budget = kv.second;
      const auto num_samples = num_samples_dict.at(node_type)[ell];

      // Sample `num_samples` nodes of `node_type` according to the budget
      // (line 9-11):
      const auto samples = sample_from(budget, num_samples);

      // Add sampled nodes to the sampled output set (line 13):
      auto &sampled_nodes = sampled_nodes_dict[node_type];
      auto &global_to_local_node = global_to_local_node_dict[node_type];
      std::vector<int64_t> newly_sampled_nodes;
      newly_sampled_nodes.reserve(samples.size());
      for (const auto &v : samples) {
        sampled_nodes.push_back(v);
        newly_sampled_nodes.push_back(v);
        global_to_local_node[v] = sampled_nodes.size();
      }

      // Add neighbors of newly sampled nodes to the bucket (line 14-15):
      update_budget(&budget_dict, node_type, newly_sampled_nodes,
                    global_to_local_node_dict, rel_to_edge_type, rowptr_dict,
                    col_dict, true);
    }
  }

  // Reconstruct the sampled adjacency matrix among the sampled nodes (line 19):
  c10::Dict<rel_t, torch::Tensor> output_row_dict;
  c10::Dict<rel_t, torch::Tensor> output_col_dict;
  c10::Dict<rel_t, torch::Tensor> output_edge_dict;
  for (const auto &kv : rowptr_dict) {
    const auto &rel_type = kv.key();
    const auto &edge_type = rel_to_edge_type.at(rel_type);
    const auto &src_node_type = std::get<0>(edge_type);
    const auto &dst_node_type = std::get<2>(edge_type);

    const auto *rowptr_data = kv.value().data_ptr<int64_t>();
    const auto *col_data = col_dict.at(rel_type).data_ptr<int64_t>();

    const auto &sampled_dst_nodes = sampled_nodes_dict[dst_node_type];
    const auto &global_to_local_src = global_to_local_node_dict[src_node_type];

    std::vector<int64_t> rows, cols, edges;
    for (int64_t i = 0; i < (int64_t)sampled_dst_nodes.size(); i++) {
      const auto v = sampled_dst_nodes[i];
      const int64_t row_start = rowptr_data[v], row_end = rowptr_data[v + 1];
      for (int64_t j = row_start; j < row_end; j++) {
        const auto w = col_data[j];
        if (global_to_local_src.find(w) != global_to_local_src.end()) {
          rows.push_back(i);
          cols.push_back(global_to_local_src.at(w));
          edges.push_back(j);
        }
      }
    }

    torch::Tensor out;
    out = torch::from_blob((int64_t *)rows.data(), {(int64_t)rows.size()},
                           at::kLong);
    output_row_dict.insert(rel_type, out.clone());
    out = torch::from_blob((int64_t *)cols.data(), {(int64_t)cols.size()},
                           at::kLong);
    output_col_dict.insert(rel_type, out.clone());
    out = torch::from_blob((int64_t *)edges.data(), {(int64_t)edges.size()},
                           at::kLong);
    output_edge_dict.insert(rel_type, out.clone());
  }

  // Generate tensor-valued output node dict (line 20):
  c10::Dict<node_t, torch::Tensor> output_node_dict;
  for (const auto &kv : sampled_nodes_dict) {
    const auto out = torch::from_blob((int64_t *)kv.second.data(),
                                      {(int64_t)kv.second.size()}, at::kLong);
    output_node_dict.insert(kv.first, out.clone());
  }

  return std::make_tuple(output_node_dict, output_row_dict, output_col_dict,
                         output_edge_dict);
}

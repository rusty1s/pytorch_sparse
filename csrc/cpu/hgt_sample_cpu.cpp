#include "hgt_sample_cpu.h"

#include "utils.h"

#ifdef _WIN32
#include <process.h>
#endif

#define MAX_NEIGHBORS 50

using namespace std;

edge_t split(const rel_t &rel_type) {
  vector<string> result(3);
  int start = 0, end;
  for (int i = 0; i < 3; i++) {
    end = rel_type.find("__", start);
    result[i] = rel_type.substr(start, end - start);
    start = end + 2;
  }
  return make_tuple(result[0], result[1], result[2]);
}

void update_budget_(
    unordered_map<node_t, unordered_map<int64_t, float>> *budget_dict,
    const node_t &node_type, const vector<int64_t> &samples,
    const unordered_map<node_t, unordered_map<int64_t, int64_t>>
        &to_local_node_dict,
    const unordered_map<rel_t, edge_t> &to_edge_type,
    const c10::Dict<rel_t, torch::Tensor> &colptr_dict,
    const c10::Dict<rel_t, torch::Tensor> &row_dict) {

  if (samples.empty())
    return;

  for (const auto &kv : colptr_dict) {
    const auto &rel_type = kv.key();
    const auto &edge_type = to_edge_type.at(rel_type);
    const auto &src_node_type = get<0>(edge_type);
    const auto &dst_node_type = get<2>(edge_type);

    if (node_type != dst_node_type)
      continue;

    const auto &to_local_src_node = to_local_node_dict.at(src_node_type);
    const auto *colptr_data = kv.value().data_ptr<int64_t>();
    const auto *row_data = row_dict.at(rel_type).data_ptr<int64_t>();
    auto &src_budget = budget_dict->at(src_node_type);

    for (const auto &w : samples) {
      const auto &col_start = colptr_data[w], &col_end = colptr_data[w + 1];
      if (col_end - col_start > MAX_NEIGHBORS) {
        // There might be same neighbors with large neighborhood sizes.
        // In order to prevent that we fill our budget with many values of low
        // probability, we instead sample a fixed amount without replacement:
        auto indices = choice(col_end - col_start, MAX_NEIGHBORS, false);
        auto *indices_data = indices.data_ptr<int64_t>();
        for (int64_t i = 0; i < indices.numel(); i++) {
          const auto &v = row_data[col_start + indices_data[i]];
          // Only add the neighbor in case we have not yet seen it before:
          if (to_local_src_node.find(v) == to_local_src_node.end())
            src_budget[v] += 1.f / float(MAX_NEIGHBORS);
        }

      } else if (col_end != col_start) {
        const auto inv_deg = 1.f / float(col_end - col_start);
        for (int64_t i = col_start; i < col_end; i++) {
          const auto &v = row_data[i];
          // Only add the neighbor in case we have not yet seen it before:
          if (to_local_src_node.find(v) == to_local_src_node.end())
            src_budget[v] += inv_deg;
        }
      }
    }
  }
}

vector<int64_t> sample_from(const unordered_map<int64_t, float> &budget,
                            const int64_t num_samples) {
  vector<int64_t> indices;
  vector<float> weights;
  indices.reserve(budget.size());
  weights.reserve(budget.size());
  for (const auto &kv : budget) {
    indices.push_back(kv.first);
    weights.push_back(kv.second * kv.second);
  }

  const auto weight = from_vector(weights, true);
  const auto sample = choice(budget.size(), num_samples, false, weight);
  const auto *sample_data = sample.data_ptr<int64_t>();

  vector<int64_t> out(sample.numel());
  for (int64_t i = 0; i < sample.numel(); i++) {
    out[i] = indices[sample_data[i]];
  }
  return out;
}

tuple<c10::Dict<node_t, torch::Tensor>, c10::Dict<rel_t, torch::Tensor>,
      c10::Dict<rel_t, torch::Tensor>, c10::Dict<rel_t, torch::Tensor>>
hgt_sample_cpu(const c10::Dict<rel_t, torch::Tensor> &colptr_dict,
               const c10::Dict<rel_t, torch::Tensor> &row_dict,
               const c10::Dict<node_t, torch::Tensor> &input_node_dict,
               const c10::Dict<node_t, vector<int64_t>> &num_samples_dict,
               const int64_t num_hops) {

  srand(time(NULL) + 1000 * getpid()); // Initialize random seed.

  // Create a mapping to convert single string relations to edge type triplets:
  unordered_map<rel_t, edge_t> to_edge_type;
  for (const auto &kv : colptr_dict) {
    const auto &rel_type = kv.key();
    to_edge_type[rel_type] = split(rel_type);
  }

  // Initialize some necessary data structures for the sampling process:
  unordered_map<node_t, vector<int64_t>> nodes_dict;
  unordered_map<node_t, unordered_map<int64_t, int64_t>> to_local_node_dict;
  unordered_map<node_t, unordered_map<int64_t, float>> budget_dict;
  for (const auto &kv : num_samples_dict) {
    const auto &node_type = kv.key();
    nodes_dict[node_type];
    to_local_node_dict[node_type];
    budget_dict[node_type];
  }

  // Add the input nodes to the sampled output nodes (line 1):
  for (const auto &kv : input_node_dict) {
    const auto &node_type = kv.key();
    const auto &input_node = kv.value();
    const auto *input_node_data = input_node.data_ptr<int64_t>();

    auto &nodes = nodes_dict.at(node_type);
    auto &to_local_node = to_local_node_dict.at(node_type);
    for (int64_t i = 0; i < input_node.numel(); i++) {
      const auto &v = input_node_data[i];
      nodes.push_back(v);
      to_local_node[v] = i;
    }
  }

  // Update the budget based on the initial input set (line 3-5):
  for (const auto &kv : nodes_dict) {
    const auto &node_type = kv.first;
    const auto &last_samples = kv.second;
    update_budget_(&budget_dict, node_type, last_samples, to_local_node_dict,
                   to_edge_type, colptr_dict, row_dict);
  }

  for (int64_t ell = 0; ell < num_hops; ell++) {
    unordered_map<node_t, vector<int64_t>> samples_dict;
    for (auto &kv : budget_dict) {
      const auto &node_type = kv.first;
      auto &budget = kv.second;
      const auto num_samples = num_samples_dict.at(node_type)[ell];

      // Sample `num_samples` nodes, according to the budget (line 9-11):
      const auto samples = sample_from(budget, num_samples);
      samples_dict[node_type] = samples;

      // Add samples to the sampled output nodes, and erase them from the budget
      // (line 13/15):
      auto &nodes = nodes_dict.at(node_type);
      auto &to_local_node = to_local_node_dict.at(node_type);
      for (const auto &v : samples) {
        to_local_node[v] = nodes.size();
        nodes.push_back(v);
        budget.erase(v);
      }
    }

    if (ell < num_hops - 1) {
      // Add neighbors of newly sampled nodes to the budget (line 14):
      // Note that we do not need to update the budget in the last iteration.
      for (const auto &kv : samples_dict) {
        const auto &node_type = kv.first;
        const auto &last_samples = kv.second;
        update_budget_(&budget_dict, node_type, last_samples,
                       to_local_node_dict, to_edge_type, colptr_dict, row_dict);
      }
    }
  }

  c10::Dict<node_t, torch::Tensor> out_node_dict;
  c10::Dict<rel_t, torch::Tensor> out_row_dict;
  c10::Dict<rel_t, torch::Tensor> out_col_dict;
  c10::Dict<rel_t, torch::Tensor> out_edge_dict;

  // Reconstruct the sampled adjacency matrix among the sampled nodes (line 19):
  for (const auto &kv : colptr_dict) {
    const auto &rel_type = kv.key();
    const auto &edge_type = to_edge_type.at(rel_type);
    const auto &src_node_type = get<0>(edge_type);
    const auto &dst_node_type = get<2>(edge_type);

    const auto *colptr_data = kv.value().data_ptr<int64_t>();
    const auto *row_data = row_dict.at(rel_type).data_ptr<int64_t>();

    const auto &dst_nodes = nodes_dict.at(dst_node_type);
    const auto &to_local_src_node = to_local_node_dict.at(src_node_type);

    vector<int64_t> rows, cols, edges;
    for (int64_t i = 0; i < (int64_t)dst_nodes.size(); i++) {
      const auto &w = dst_nodes[i];
      const auto &col_start = colptr_data[w], &col_end = colptr_data[w + 1];
      if (col_end - col_start > MAX_NEIGHBORS) {
        auto indices = choice(col_end - col_start, MAX_NEIGHBORS, false);
        auto *indices_data = indices.data_ptr<int64_t>();
        for (int64_t j = 0; j < indices.numel(); j++) {
          const auto &v = row_data[col_start + indices_data[j]];
          if (to_local_src_node.find(v) != to_local_src_node.end()) {
            rows.push_back(to_local_src_node.at(v));
            cols.push_back(i);
            edges.push_back(col_start + j);
          }
        }
      } else {
        for (int64_t j = col_start; j < col_end; j++) {
          const auto &v = row_data[j];
          if (to_local_src_node.find(v) != to_local_src_node.end()) {
            rows.push_back(to_local_src_node.at(v));
            cols.push_back(i);
            edges.push_back(j);
          }
        }
      }
    }
    if (rows.size() > 0) {
      out_row_dict.insert(rel_type, from_vector<int64_t>(rows));
      out_col_dict.insert(rel_type, from_vector<int64_t>(cols));
      out_edge_dict.insert(rel_type, from_vector<int64_t>(edges));
    }
  }

  // Generate tensor-valued output node dictionary (line 20):
  for (const auto &kv : nodes_dict) {
    const auto &node_type = kv.first;
    const auto &nodes = kv.second;
    if (!nodes.empty())
      out_node_dict.insert(node_type, from_vector<int64_t>(nodes));
  }

  return make_tuple(out_node_dict, out_row_dict, out_col_dict, out_edge_dict);
}

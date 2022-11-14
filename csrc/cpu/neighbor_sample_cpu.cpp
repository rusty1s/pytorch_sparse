#include "neighbor_sample_cpu.h"

#include "utils.h"

#ifdef _WIN32
#include <process.h>
#endif

using namespace std;

namespace {

typedef phmap::flat_hash_map<pair<int64_t, int64_t>, int64_t> temporarl_edge_dict;

template <bool replace, bool directed>
tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sample(const torch::Tensor &colptr, const torch::Tensor &row,
       const torch::Tensor &input_node, const vector<int64_t> num_neighbors) {

  // Initialize some data structures for the sampling process:
  vector<int64_t> samples;
  phmap::flat_hash_map<int64_t, int64_t> to_local_node;

  auto *colptr_data = colptr.data_ptr<int64_t>();
  auto *row_data = row.data_ptr<int64_t>();
  auto *input_node_data = input_node.data_ptr<int64_t>();

  for (int64_t i = 0; i < input_node.numel(); i++) {
    const auto &v = input_node_data[i];
    samples.push_back(v);
    to_local_node.insert({v, i});
  }

  vector<int64_t> rows, cols, edges;

  int64_t begin = 0, end = samples.size();
  for (int64_t ell = 0; ell < (int64_t)num_neighbors.size(); ell++) {
    const auto &num_samples = num_neighbors[ell];
    for (int64_t i = begin; i < end; i++) {
      const auto &w = samples[i];
      const auto &col_start = colptr_data[w];
      const auto &col_end = colptr_data[w + 1];
      const auto col_count = col_end - col_start;

      if (col_count == 0)
        continue;

      if ((num_samples < 0) || (!replace && (num_samples >= col_count))) {
        for (int64_t offset = col_start; offset < col_end; offset++) {
          const int64_t &v = row_data[offset];
          const auto res = to_local_node.insert({v, samples.size()});
          if (res.second)
            samples.push_back(v);
          if (directed) {
            cols.push_back(i);
            rows.push_back(res.first->second);
            edges.push_back(offset);
          }
        }
      } else if (replace) {
        for (int64_t j = 0; j < num_samples; j++) {
          const int64_t offset = col_start + uniform_randint(col_count);
          const int64_t &v = row_data[offset];
          const auto res = to_local_node.insert({v, samples.size()});
          if (res.second)
            samples.push_back(v);
          if (directed) {
            cols.push_back(i);
            rows.push_back(res.first->second);
            edges.push_back(offset);
          }
        }
      } else {
        unordered_set<int64_t> rnd_indices;
        for (int64_t j = col_count - num_samples; j < col_count; j++) {
          int64_t rnd = uniform_randint(j);
          if (!rnd_indices.insert(rnd).second) {
            rnd = j;
            rnd_indices.insert(j);
          }
          const int64_t offset = col_start + rnd;
          const int64_t &v = row_data[offset];
          const auto res = to_local_node.insert({v, samples.size()});
          if (res.second)
            samples.push_back(v);
          if (directed) {
            cols.push_back(i);
            rows.push_back(res.first->second);
            edges.push_back(offset);
          }
        }
      }
    }
    begin = end, end = samples.size();
  }

  if (!directed) {
    phmap::flat_hash_map<int64_t, int64_t>::iterator iter;
    for (int64_t i = 0; i < (int64_t)samples.size(); i++) {
      const auto &w = samples[i];
      const auto &col_start = colptr_data[w];
      const auto &col_end = colptr_data[w + 1];
      for (int64_t offset = col_start; offset < col_end; offset++) {
        const auto &v = row_data[offset];
        iter = to_local_node.find(v);
        if (iter != to_local_node.end()) {
          rows.push_back(iter->second);
          cols.push_back(i);
          edges.push_back(offset);
        }
      }
    }
  }

  return make_tuple(from_vector<int64_t>(samples), from_vector<int64_t>(rows),
                    from_vector<int64_t>(cols), from_vector<int64_t>(edges));
}

inline bool satisfy_time(const c10::Dict<node_t, torch::Tensor> &node_time_dict,
                         const node_t &src_node_type, int64_t dst_time,
                         int64_t src_node) {
  try {
    // Check whether src -> dst obeys the time constraint
    const torch::Tensor &src_node_time = node_time_dict.at(src_node_type);
    return src_node_time.data_ptr<int64_t>()[src_node] <= dst_time;
  } catch (const std::out_of_range& e) {
    // If no time is given, fall back to normal sampling
    return true;
  }
}

template <bool replace, bool directed, bool temporal>
tuple<c10::Dict<node_t, torch::Tensor>, c10::Dict<rel_t, torch::Tensor>,
      c10::Dict<rel_t, torch::Tensor>, c10::Dict<rel_t, torch::Tensor>>
hetero_sample(const vector<node_t> &node_types,
              const vector<edge_t> &edge_types,
              const c10::Dict<rel_t, torch::Tensor> &colptr_dict,
              const c10::Dict<rel_t, torch::Tensor> &row_dict,
              const c10::Dict<node_t, torch::Tensor> &input_node_dict,
              const c10::Dict<rel_t, vector<int64_t>> &num_neighbors_dict,
              const c10::Dict<node_t, torch::Tensor> &node_time_dict,
              const int64_t num_hops) {

  // Create a mapping to convert single string relations to edge type triplets:
  phmap::flat_hash_map<rel_t, edge_t> to_edge_type;
  for (const auto &k : edge_types)
    to_edge_type[get<0>(k) + "__" + get<1>(k) + "__" + get<2>(k)] = k;

  // Initialize some data structures for the sampling process:
  phmap::flat_hash_map<node_t, vector<int64_t>> samples_dict;
  phmap::flat_hash_map<node_t, vector<pair<int64_t, int64_t>>> temp_samples_dict;
  phmap::flat_hash_map<node_t, phmap::flat_hash_map<int64_t, int64_t>> to_local_node_dict;
  phmap::flat_hash_map<node_t, temporarl_edge_dict> temp_to_local_node_dict;
  phmap::flat_hash_map<node_t, vector<int64_t>> root_time_dict;
  for (const auto &node_type : node_types) {
    samples_dict[node_type];
    temp_samples_dict[node_type];
    to_local_node_dict[node_type];
    temp_to_local_node_dict[node_type];
    root_time_dict[node_type];
  }

  phmap::flat_hash_map<rel_t, vector<int64_t>> rows_dict, cols_dict, edges_dict;
  for (const auto &kv : colptr_dict) {
    const auto &rel_type = kv.key();
    rows_dict[rel_type];
    cols_dict[rel_type];
    edges_dict[rel_type];
  }

  // Add the input nodes to the output nodes:
  for (const auto &kv : input_node_dict) {
    const auto &node_type = kv.key();
    const torch::Tensor &input_node = kv.value();
    const auto *input_node_data = input_node.data_ptr<int64_t>();

    int64_t *node_time_data;
    if (temporal) {
      const torch::Tensor &node_time = node_time_dict.at(node_type);
      node_time_data = node_time.data_ptr<int64_t>();
    }

    auto &samples = samples_dict.at(node_type);
    auto &temp_samples = temp_samples_dict.at(node_type);
    auto &to_local_node = to_local_node_dict.at(node_type);
    auto &temp_to_local_node = temp_to_local_node_dict.at(node_type);
    auto &root_time = root_time_dict.at(node_type);
    for (int64_t i = 0; i < input_node.numel(); i++) {
      const auto &v = input_node_data[i];
      if (temporal) {
        temp_samples.push_back({v, i});
        temp_to_local_node.insert({{v, i}, i});
      } else {
        samples.push_back(v);
        to_local_node.insert({v, i});
      }
      if (temporal)
        root_time.push_back(node_time_data[v]);
    }
  }

  phmap::flat_hash_map<node_t, pair<int64_t, int64_t>> slice_dict;
  if (temporal) {
    for (const auto &kv : temp_samples_dict) {
      slice_dict[kv.first] = {0, kv.second.size()};
    }
  } else {
    for (const auto &kv : samples_dict)
      slice_dict[kv.first] = {0, kv.second.size()};
  }

  vector<rel_t> all_rel_types;
  for (const auto &kv : num_neighbors_dict) {
    all_rel_types.push_back(kv.key());
  }
  std::sort(all_rel_types.begin(), all_rel_types.end());

  for (int64_t ell = 0; ell < num_hops; ell++) {
    for (const auto &rel_type : all_rel_types) {
      const auto &edge_type = to_edge_type[rel_type];
      const auto &src_node_type = get<0>(edge_type);
      const auto &dst_node_type = get<2>(edge_type);
      const auto num_samples = num_neighbors_dict.at(rel_type)[ell];
      const auto &dst_samples = samples_dict.at(dst_node_type);
      const auto &temp_dst_samples = temp_samples_dict.at(dst_node_type);
      auto &src_samples = samples_dict.at(src_node_type);
      auto &temp_src_samples = temp_samples_dict.at(src_node_type);
      auto &to_local_src_node = to_local_node_dict.at(src_node_type);
      auto &temp_to_local_src_node = temp_to_local_node_dict.at(src_node_type);

      const torch::Tensor &colptr = colptr_dict.at(rel_type);
      const auto *colptr_data = colptr.data_ptr<int64_t>();
      const torch::Tensor &row = row_dict.at(rel_type);
      const auto *row_data = row.data_ptr<int64_t>();

      auto &rows = rows_dict.at(rel_type);
      auto &cols = cols_dict.at(rel_type);
      auto &edges = edges_dict.at(rel_type);

      // For temporal sampling, sampled nodes cannot have a timestamp greater
      // than the timestamp of the root nodes:
      const auto &dst_root_time = root_time_dict.at(dst_node_type);
      auto &src_root_time = root_time_dict.at(src_node_type);

      const auto &begin = slice_dict.at(dst_node_type).first;
      const auto &end = slice_dict.at(dst_node_type).second;
      for (int64_t i = begin; i < end; i++) {
        const auto &w = temporal ? temp_dst_samples[i].first : dst_samples[i];
        const int64_t root_w = temporal ? temp_dst_samples[i].second : -1;
        int64_t dst_time = 0;
        if (temporal)
          dst_time = dst_root_time[i];
        const auto &col_start = colptr_data[w];
        const auto &col_end = colptr_data[w + 1];
        const auto col_count = col_end - col_start;

        if (col_count == 0)
          continue;

        if ((num_samples < 0) || (!replace && (num_samples >= col_count))) {
          // Select all neighbors:
          for (int64_t offset = col_start; offset < col_end; offset++) {
            const int64_t &v = row_data[offset];
            if (temporal) {
              if (!satisfy_time(node_time_dict, src_node_type, dst_time, v))
                continue;
              // force disjoint of computation tree based on source batch idx.
              // note that the sampling always needs to have directed=True
              // for temporal case
              // to_local_src_node is not used for temporal / directed case
              const auto res = temp_to_local_src_node.insert({{v, root_w}, (int64_t)temp_src_samples.size()});
              if (res.second) {
                temp_src_samples.push_back({v, root_w});
                src_root_time.push_back(dst_time);
              }

              cols.push_back(i);
              rows.push_back(res.first->second);
              edges.push_back(offset);
            } else {
              const auto res = to_local_src_node.insert({v, src_samples.size()});
              if (res.second)
                src_samples.push_back(v);
              if (directed) {
                cols.push_back(i);
                rows.push_back(res.first->second);
                edges.push_back(offset);
              }
            }
          }
        } else if (replace) {
          // Sample with replacement:
          int64_t num_neighbors = 0;
          while (num_neighbors < num_samples) {
            const int64_t offset = col_start + uniform_randint(col_count);
            const int64_t &v = row_data[offset];
            if (temporal) {
              // TODO Infinity loop if no neighbor satisfies time constraint:
              if (!satisfy_time(node_time_dict, src_node_type, dst_time, v))
                continue;
              // force disjoint of computation tree based on source batch idx.
              // note that the sampling always needs to have directed=True
              // for temporal case
              const auto res = temp_to_local_src_node.insert({{v, root_w}, (int64_t)temp_src_samples.size()});
              if (res.second) {
                temp_src_samples.push_back({v, root_w});
                src_root_time.push_back(dst_time);
              }

              cols.push_back(i);
              rows.push_back(res.first->second);
              edges.push_back(offset);
            } else {
              const auto res = to_local_src_node.insert({v, src_samples.size()});
              if (res.second)
                src_samples.push_back(v);
              if (directed) {
                cols.push_back(i);
                rows.push_back(res.first->second);
                edges.push_back(offset);
              }
            }
            num_neighbors += 1;
          }
        } else {
          // Sample without replacement:
          unordered_set<int64_t> rnd_indices;
          for (int64_t j = col_count - num_samples; j < col_count; j++) {
            int64_t rnd = uniform_randint(j);
            if (!rnd_indices.insert(rnd).second) {
              rnd = j;
              rnd_indices.insert(j);
            }
            const int64_t offset = col_start + rnd;
            const int64_t &v = row_data[offset];
            if (temporal) {
              if (!satisfy_time(node_time_dict, src_node_type, dst_time, v))
                continue;
              // force disjoint of computation tree based on source batch idx.
              // note that the sampling always needs to have directed=True
              // for temporal case
              const auto res = temp_to_local_src_node.insert({{v, root_w}, (int64_t)temp_src_samples.size()});
              if (res.second) {
                temp_src_samples.push_back({v, root_w});
                src_root_time.push_back(dst_time);
              }

              cols.push_back(i);
              rows.push_back(res.first->second);
              edges.push_back(offset);
            } else {
              const auto res = to_local_src_node.insert({v, src_samples.size()});
              if (res.second)
                src_samples.push_back(v);
              if (directed) {
                cols.push_back(i);
                rows.push_back(res.first->second);
                edges.push_back(offset);
              }
            }
          }
        }
      }
    }

    if (temporal) {
      for (const auto &kv : temp_samples_dict) {
        slice_dict[kv.first] = {slice_dict.at(kv.first).second, kv.second.size()};
      }
    } else {
      for (const auto &kv : samples_dict)
        slice_dict[kv.first] = {slice_dict.at(kv.first).second, kv.second.size()};
    }
  }

  // Temporal sample disable undirected
  assert(!(temporal && !directed));
  if (!directed) { // Construct the subgraph among the sampled nodes:
    phmap::flat_hash_map<int64_t, int64_t>::iterator iter;
    for (const auto &kv : colptr_dict) {
      const auto &rel_type = kv.key();
      const auto &edge_type = to_edge_type[rel_type];
      const auto &src_node_type = get<0>(edge_type);
      const auto &dst_node_type = get<2>(edge_type);
      const auto &dst_samples = samples_dict.at(dst_node_type);
      auto &to_local_src_node = to_local_node_dict.at(src_node_type);

      const auto *colptr_data = ((torch::Tensor)kv.value()).data_ptr<int64_t>();
      const auto *row_data =
          ((torch::Tensor)row_dict.at(rel_type)).data_ptr<int64_t>();

      auto &rows = rows_dict.at(rel_type);
      auto &cols = cols_dict.at(rel_type);
      auto &edges = edges_dict.at(rel_type);

      for (int64_t i = 0; i < (int64_t)dst_samples.size(); i++) {
        const auto &w = dst_samples[i];
        const auto &col_start = colptr_data[w];
        const auto &col_end = colptr_data[w + 1];
        for (int64_t offset = col_start; offset < col_end; offset++) {
          const auto &v = row_data[offset];
          iter = to_local_src_node.find(v);
          if (iter != to_local_src_node.end()) {
            rows.push_back(iter->second);
            cols.push_back(i);
            edges.push_back(offset);
          }
        }
      }
    }
  }

  // Construct samples dictionary from temporal sample dictionary.
  if (temporal) {
    for (const auto &kv : temp_samples_dict) {
      const auto &node_type = kv.first;
      const auto &samples = kv.second;
      samples_dict[node_type].reserve(samples.size());
      for (const auto &v : samples) {
        samples_dict[node_type].push_back(v.first);
      }
    }
  }

  return make_tuple(from_vector<node_t, int64_t>(samples_dict),
                    from_vector<rel_t, int64_t>(rows_dict),
                    from_vector<rel_t, int64_t>(cols_dict),
                    from_vector<rel_t, int64_t>(edges_dict));
}

} // namespace

tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
neighbor_sample_cpu(const torch::Tensor &colptr, const torch::Tensor &row,
                    const torch::Tensor &input_node,
                    const vector<int64_t> num_neighbors, const bool replace,
                    const bool directed) {

  if (replace && directed) {
    return sample<true, true>(colptr, row, input_node, num_neighbors);
  } else if (replace && !directed) {
    return sample<true, false>(colptr, row, input_node, num_neighbors);
  } else if (!replace && directed) {
    return sample<false, true>(colptr, row, input_node, num_neighbors);
  } else {
    return sample<false, false>(colptr, row, input_node, num_neighbors);
  }
}

tuple<c10::Dict<node_t, torch::Tensor>, c10::Dict<rel_t, torch::Tensor>,
      c10::Dict<rel_t, torch::Tensor>, c10::Dict<rel_t, torch::Tensor>>
hetero_neighbor_sample_cpu(
    const vector<node_t> &node_types, const vector<edge_t> &edge_types,
    const c10::Dict<rel_t, torch::Tensor> &colptr_dict,
    const c10::Dict<rel_t, torch::Tensor> &row_dict,
    const c10::Dict<node_t, torch::Tensor> &input_node_dict,
    const c10::Dict<rel_t, vector<int64_t>> &num_neighbors_dict,
    const int64_t num_hops, const bool replace, const bool directed) {

  c10::Dict<node_t, torch::Tensor> node_time_dict; // Empty dictionary.

  if (replace && directed) {
    return hetero_sample<true, true, false>(
        node_types, edge_types, colptr_dict, row_dict, input_node_dict,
        num_neighbors_dict, node_time_dict, num_hops);
  } else if (replace && !directed) {
    return hetero_sample<true, false, false>(
        node_types, edge_types, colptr_dict, row_dict, input_node_dict,
        num_neighbors_dict, node_time_dict, num_hops);
  } else if (!replace && directed) {
    return hetero_sample<false, true, false>(
        node_types, edge_types, colptr_dict, row_dict, input_node_dict,
        num_neighbors_dict, node_time_dict, num_hops);
  } else {
    return hetero_sample<false, false, false>(
        node_types, edge_types, colptr_dict, row_dict, input_node_dict,
        num_neighbors_dict, node_time_dict, num_hops);
  }
}

tuple<c10::Dict<node_t, torch::Tensor>, c10::Dict<rel_t, torch::Tensor>,
      c10::Dict<rel_t, torch::Tensor>, c10::Dict<rel_t, torch::Tensor>>
hetero_temporal_neighbor_sample_cpu(
    const vector<node_t> &node_types, const vector<edge_t> &edge_types,
    const c10::Dict<rel_t, torch::Tensor> &colptr_dict,
    const c10::Dict<rel_t, torch::Tensor> &row_dict,
    const c10::Dict<node_t, torch::Tensor> &input_node_dict,
    const c10::Dict<rel_t, vector<int64_t>> &num_neighbors_dict,
    const c10::Dict<node_t, torch::Tensor> &node_time_dict,
    const int64_t num_hops, const bool replace, const bool directed) {
  AT_ASSERTM(directed, "Temporal sampling requires 'directed' sampling");
  if (replace) {
    // We assume that directed = True for temporal sampling
    // The current implementation uses disjoint computation trees
    // to tackle the case of the same node sampled having different
    // root time constraint.
    // In future, we could extend to directed = False case,
    // allowing additional edges within each computation tree.
    return hetero_sample<true, true, true>(
        node_types, edge_types, colptr_dict, row_dict, input_node_dict,
        num_neighbors_dict, node_time_dict, num_hops);
  } else {
    return hetero_sample<false, true, true>(
        node_types, edge_types, colptr_dict, row_dict, input_node_dict,
        num_neighbors_dict, node_time_dict, num_hops);
  }
}

#include "neighbor_sample_cpu.h"

#include "utils.h"

#ifdef _WIN32
#include <process.h>
#endif

using namespace std;

namespace {

template <bool replace, bool directed>
tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sample(const torch::Tensor &colptr, const torch::Tensor &row,
       const torch::Tensor &input_node, const vector<int64_t> num_neighbors) {

  // Initialize some data structures for the sampling process:
  vector<int64_t> samples;
  unordered_map<int64_t, int64_t> to_local_node;

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
    unordered_map<int64_t, int64_t>::iterator iter;
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

bool satisfy_time_constraint(const c10::Dict<node_t, torch::Tensor> &node_time_dict,
                             const std::string &src_node_type,
                             const int64_t &dst_time,
                             const int64_t &sampled_node) {
  // whether src -> dst obeys the time constraint
  try {
    const auto *src_time = node_time_dict.at(src_node_type).data_ptr<int64_t>();
    if (dst_time < src_time[sampled_node])
      return false;
    return true;
  }
  catch (int err) {
    // if the node type does not have timestamp, fall back to normal sampling
    return true;
  }
}


template <bool replace, bool directed>
tuple<c10::Dict<node_t, torch::Tensor>, c10::Dict<rel_t, torch::Tensor>,
      c10::Dict<rel_t, torch::Tensor>, c10::Dict<rel_t, torch::Tensor>>
hetero_sample_temporal(const vector<node_t> &node_types,
                       const vector<edge_t> &edge_types,
                       const c10::Dict<rel_t, torch::Tensor> &colptr_dict,
                       const c10::Dict<rel_t, torch::Tensor> &row_dict,
                       const c10::Dict<node_t, torch::Tensor> &input_node_dict,
                       const c10::Dict<rel_t, vector<int64_t>> &num_neighbors_dict,
                       const int64_t num_hops,
                       const c10::Dict<node_t, torch::Tensor> &node_time_dict) {
  bool is_temporal = (!node_time_dict.empty());

  // Create a mapping to convert single string relations to edge type triplets:
  unordered_map<rel_t, edge_t> to_edge_type;
  for (const auto &k : edge_types)
    to_edge_type[get<0>(k) + "__" + get<1>(k) + "__" + get<2>(k)] = k;

  // Initialize some data structures for the sampling process:
  unordered_map<rel_t, vector<int64_t>> rows_dict, cols_dict, edges_dict;
  for (const auto &kv : colptr_dict) {
    const auto &rel_type = kv.key();
    rows_dict[rel_type];
    cols_dict[rel_type];
    edges_dict[rel_type];
  }

  unordered_map<node_t, vector<int64_t>> samples_dict;
  unordered_map<node_t, unordered_map<int64_t, int64_t>> to_local_node_dict;
  // The timestamp of the center node whose neighborhood that the sampled node
  // belongs to. It maps node_type to empty vector in non-temporal sampling.
  unordered_map<node_t, vector<int64_t>> root_time_dict;
  for (const auto &node_type : node_types) {
    samples_dict[node_type];
    to_local_node_dict[node_type];
    root_time_dict[node_type];
  }

  // Add the input nodes to the output nodes:
  for (const auto &kv : input_node_dict) {
    const auto &node_type = kv.key();
    const torch::Tensor &input_node = kv.value();
    const auto *input_node_data = input_node.data_ptr<int64_t>();
    // dummy value. will be reset to root time if is_temporal==true
    const auto *node_time_data = input_node.data_ptr<int64_t>();
    // root_time[i] stores the timestamp of the computation tree root
    // of the node samples[i]
    if (is_temporal) {
      // node_time_data = node_time_dict.at(node_type).data_ptr<int64_t>();
      node_time_data = input_node.data_ptr<int64_t>()
    }

    auto &samples = samples_dict.at(node_type);
    auto &to_local_node = to_local_node_dict.at(node_type);
    auto &root_time = root_time_dict.at(node_type);
    for (int64_t i = 0; i < input_node.numel(); i++) {
      const auto &v = input_node_data[i];
      samples.push_back(v);
      to_local_node.insert({v, i});
      if (is_temporal) {
        root_time.push_back(node_time_data[v]);
      }
    }
  }

  unordered_map<node_t, pair<int64_t, int64_t>> slice_dict;
  for (const auto &kv : samples_dict)
    slice_dict[kv.first] = {0, kv.second.size()};

  for (int64_t ell = 0; ell < num_hops; ell++) {
    for (const auto &kv : num_neighbors_dict) {
      const auto &rel_type = kv.key();
      const auto &edge_type = to_edge_type[rel_type];
      const auto &src_node_type = get<0>(edge_type);
      const auto &dst_node_type = get<2>(edge_type);
      const auto num_samples = kv.value()[ell];
      const auto &dst_samples = samples_dict.at(dst_node_type);
      auto &src_samples = samples_dict.at(src_node_type);
      auto &to_local_src_node = to_local_node_dict.at(src_node_type);

      const auto *colptr_data =
          ((torch::Tensor)colptr_dict.at(rel_type)).data_ptr<int64_t>();
      const auto *row_data =
          ((torch::Tensor)row_dict.at(rel_type)).data_ptr<int64_t>();

      auto &rows = rows_dict.at(rel_type);
      auto &cols = cols_dict.at(rel_type);
      auto &edges = edges_dict.at(rel_type);

      const auto &begin = slice_dict.at(dst_node_type).first;
      const auto &end = slice_dict.at(dst_node_type).second;
      if (begin == end){
      }
      // for temporal sampling, sampled src node cannot have timestamp greater
      // than its corresponding dst_root_time
      const auto &dst_root_time = root_time_dict.at(dst_node_type);
      auto &src_root_time = root_time_dict.at(src_node_type);

      for (int64_t i = begin; i < end; i++) {
        const auto &w = dst_samples[i];
        const auto &dst_time = dst_root_time[i];
        const auto &col_start = colptr_data[w];
        const auto &col_end = colptr_data[w + 1];
        const auto col_count = col_end - col_start;

        if (col_count == 0)
          continue;

        if ((num_samples < 0) || (!replace && (num_samples >= col_count))) {
          // select all neighbors
          for (int64_t offset = col_start; offset < col_end; offset++) {
            const int64_t &v = row_data[offset];
            bool time_constraint = true;
            if (is_temporal) {
              time_constraint = satisfy_time_constraint(
                  node_time_dict, src_node_type, dst_time, v);
            }
            if (!time_constraint)
              continue;
            const auto res = to_local_src_node.insert({v, src_samples.size()});
            if (res.second) {
              src_samples.push_back(v);
              if (is_temporal)
                src_root_time.push_back(dst_time);
            }
            if (directed) {
              cols.push_back(i);
              rows.push_back(res.first->second);
              edges.push_back(offset);
            }
          }
        } else if (replace) {
          // sample with replacement
          int64_t num_neighbors = 0;
          while (num_neighbors < num_samples) {
            const int64_t offset = col_start + uniform_randint(col_count);
            const int64_t &v = row_data[offset];
            bool time_constraint = true;
            if (is_temporal) {
              time_constraint = satisfy_time_constraint(
                  node_time_dict, src_node_type, dst_time, v);
            }
            if (!time_constraint)
              continue;
            const auto res = to_local_src_node.insert({v, src_samples.size()});
            if (res.second) {
              src_samples.push_back(v);
              if (is_temporal)
                src_root_time.push_back(dst_time);
            }
            if (directed) {
              cols.push_back(i);
              rows.push_back(res.first->second);
              edges.push_back(offset);
            }
            num_neighbors += 1;
          }
        } else {
          // sample without replacement
          unordered_set<int64_t> rnd_indices;
          for (int64_t j = col_count - num_samples; j < col_count; j++) {
            int64_t rnd = uniform_randint(j);
            if (!rnd_indices.insert(rnd).second) {
              rnd = j;
              rnd_indices.insert(j);
            }
            const int64_t offset = col_start + rnd;
            const int64_t &v = row_data[offset];
            bool time_constraint = true;
            if (is_temporal) {
              time_constraint = satisfy_time_constraint(
                  node_time_dict, src_node_type, dst_time, v);
            }
            if (!time_constraint)
              continue;
            const auto res = to_local_src_node.insert({v, src_samples.size()});
            if (res.second) {
              src_samples.push_back(v);
              if (is_temporal)
                src_root_time.push_back(dst_time);
            }
            if (directed) {
              cols.push_back(i);
              rows.push_back(res.first->second);
              edges.push_back(offset);
            }
          }
        }
      }
    }

    for (const auto &kv : samples_dict) {
      slice_dict[kv.first] = {slice_dict.at(kv.first).second, kv.second.size()};
    }
  }

  if (!directed) { // Construct the subgraph among the sampled nodes:
    unordered_map<int64_t, int64_t>::iterator iter;
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

  return make_tuple(from_vector<node_t, int64_t>(samples_dict),
                    from_vector<rel_t, int64_t>(rows_dict),
                    from_vector<rel_t, int64_t>(cols_dict),
                    from_vector<rel_t, int64_t>(edges_dict));
}

template <bool replace, bool directed>
tuple<c10::Dict<node_t, torch::Tensor>, c10::Dict<rel_t, torch::Tensor>,
      c10::Dict<rel_t, torch::Tensor>, c10::Dict<rel_t, torch::Tensor>>
hetero_sample_random(const vector<node_t> &node_types,
              const vector<edge_t> &edge_types,
              const c10::Dict<rel_t, torch::Tensor> &colptr_dict,
              const c10::Dict<rel_t, torch::Tensor> &row_dict,
              const c10::Dict<node_t, torch::Tensor> &input_node_dict,
              const c10::Dict<rel_t, vector<int64_t>> &num_neighbors_dict,
              const int64_t num_hops) {
  c10::Dict<node_t, torch::Tensor> empty_dict;
  return hetero_sample_temporal<replace, directed>(node_types,
              edge_types,
              colptr_dict,
              row_dict,
              input_node_dict,
              num_neighbors_dict,
              num_hops,
              empty_dict);
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

  if (replace && directed) {
    return hetero_sample_random<true, true>(
        node_types, edge_types, colptr_dict,
        row_dict, input_node_dict,
        num_neighbors_dict, num_hops);
  } else if (replace && !directed) {
    return hetero_sample_random<true, false>(
        node_types, edge_types, colptr_dict,
        row_dict, input_node_dict,
        num_neighbors_dict, num_hops);
  } else if (!replace && directed) {
    return hetero_sample_random<false, true>(
        node_types, edge_types, colptr_dict,
        row_dict, input_node_dict,
        num_neighbors_dict, num_hops);
  } else {
    return hetero_sample_random<false, false>(
        node_types, edge_types, colptr_dict,
        row_dict, input_node_dict,
        num_neighbors_dict, num_hops);
  }
}

tuple<c10::Dict<node_t, torch::Tensor>, c10::Dict<rel_t, torch::Tensor>,
      c10::Dict<rel_t, torch::Tensor>, c10::Dict<rel_t, torch::Tensor>>
hetero_neighbor_temporal_sample_cpu(
    const vector<node_t> &node_types, const vector<edge_t> &edge_types,
    const c10::Dict<rel_t, torch::Tensor> &colptr_dict,
    const c10::Dict<rel_t, torch::Tensor> &row_dict,
    const c10::Dict<node_t, torch::Tensor> &input_node_dict,
    const c10::Dict<rel_t, vector<int64_t>> &num_neighbors_dict,
    const c10::Dict<node_t, torch::Tensor> &node_time_dict,
    const int64_t num_hops, const bool replace, const bool directed) {

  if (replace && directed) {
    return hetero_sample_temporal<true, true>(
        node_types, edge_types, colptr_dict,
        row_dict, input_node_dict,
        num_neighbors_dict, num_hops, node_time_dict);
  } else if (replace && !directed) {
    return hetero_sample_temporal<true, false>(
        node_types, edge_types, colptr_dict,
        row_dict, input_node_dict,
        num_neighbors_dict, num_hops, node_time_dict);
  } else if (!replace && directed) {
    return hetero_sample_temporal<false, true>(
        node_types, edge_types, colptr_dict,
        row_dict, input_node_dict,
        num_neighbors_dict, num_hops, node_time_dict);
  } else {
    return hetero_sample_temporal<false, false>(
        node_types, edge_types, colptr_dict,
        row_dict, input_node_dict,
        num_neighbors_dict, num_hops, node_time_dict);
  }
}
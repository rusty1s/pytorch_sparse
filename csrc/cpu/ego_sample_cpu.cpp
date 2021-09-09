#include "ego_sample_cpu.h"

#include <ATen/Parallel.h>

#include "utils.h"

#ifdef _WIN32
#include <process.h>
#endif

inline torch::Tensor vec2tensor(std::vector<int64_t> vec) {
  return torch::from_blob(vec.data(), {(int64_t)vec.size()}, at::kLong).clone();
}

// Returns `rowptr`, `col`, `n_id`, `e_id`, `ptr`, `root_n_id`
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
ego_k_hop_sample_adj_cpu(torch::Tensor rowptr, torch::Tensor col,
                         torch::Tensor idx, int64_t depth,
                         int64_t num_neighbors, bool replace) {

  srand(time(NULL) + 1000 * getpid()); // Initialize random seed.

  std::vector<torch::Tensor> out_rowptrs(idx.numel() + 1);
  std::vector<torch::Tensor> out_cols(idx.numel());
  std::vector<torch::Tensor> out_n_ids(idx.numel());
  std::vector<torch::Tensor> out_e_ids(idx.numel());
  auto out_root_n_id = torch::empty({idx.numel()}, at::kLong);
  out_rowptrs[0] = torch::zeros({1}, at::kLong);

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  auto idx_data = idx.data_ptr<int64_t>();
  auto out_root_n_id_data = out_root_n_id.data_ptr<int64_t>();

  at::parallel_for(0, idx.numel(), 1, [&](int64_t begin, int64_t end) {
    int64_t row_start, row_end, row_count, vec_start, vec_end, v, w;
    for (int64_t g = begin; g < end; g++) {
      std::set<int64_t> n_id_set;
      n_id_set.insert(idx_data[g]);
      std::vector<int64_t> n_ids;
      n_ids.push_back(idx_data[g]);

      vec_start = 0, vec_end = n_ids.size();
      for (int64_t d = 0; d < depth; d++) {
        for (int64_t i = vec_start; i < vec_end; i++) {
          v = n_ids[i];
          row_start = rowptr_data[v], row_end = rowptr_data[v + 1];
          row_count = row_end - row_start;

          if (row_count <= num_neighbors) {
            for (int64_t e = row_start; e < row_end; e++) {
              w = col_data[e];
              n_id_set.insert(w);
              n_ids.push_back(w);
            }
          } else if (replace) {
            for (int64_t j = 0; j < num_neighbors; j++) {
              w = col_data[row_start + (rand() % row_count)];
              n_id_set.insert(w);
              n_ids.push_back(w);
            }
          } else {
            std::unordered_set<int64_t> perm;
            for (int64_t j = row_count - num_neighbors; j < row_count; j++) {
              if (!perm.insert(rand() % j).second) {
                perm.insert(j);
              }
            }
            for (int64_t j : perm) {
              w = col_data[row_start + j];
              n_id_set.insert(w);
              n_ids.push_back(w);
            }
          }
        }
        vec_start = vec_end;
        vec_end = n_ids.size();
      }

      n_ids.clear();
      std::map<int64_t, int64_t> n_id_map;
      std::map<int64_t, int64_t>::iterator iter;

      int64_t i = 0;
      for (int64_t v : n_id_set) {
        n_ids.push_back(v);
        n_id_map[v] = i;
        i++;
      }

      out_root_n_id_data[g] = n_id_map[idx_data[g]];

      std::vector<int64_t> rowptrs, cols, e_ids;
      for (int64_t v : n_ids) {
        row_start = rowptr_data[v], row_end = rowptr_data[v + 1];
        for (int64_t e = row_start; e < row_end; e++) {
          w = col_data[e];
          iter = n_id_map.find(w);
          if (iter != n_id_map.end()) {
            cols.push_back(iter->second);
            e_ids.push_back(e);
          }
        }
        rowptrs.push_back(cols.size());
      }

      out_rowptrs[g + 1] = vec2tensor(rowptrs);
      out_cols[g] = vec2tensor(cols);
      out_n_ids[g] = vec2tensor(n_ids);
      out_e_ids[g] = vec2tensor(e_ids);
    }
  });

  auto out_ptr = torch::empty({idx.numel() + 1}, at::kLong);
  auto out_ptr_data = out_ptr.data_ptr<int64_t>();
  out_ptr_data[0] = 0;

  int64_t node_cumsum = 0, edge_cumsum = 0;
  for (int64_t g = 1; g < idx.numel(); g++) {
    node_cumsum += out_n_ids[g - 1].numel();
    edge_cumsum += out_cols[g - 1].numel();
    out_rowptrs[g + 1].add_(edge_cumsum);
    out_cols[g].add_(node_cumsum);
    out_ptr_data[g] = node_cumsum;
    out_root_n_id_data[g] += node_cumsum;
  }
  node_cumsum += out_n_ids[idx.numel() - 1].numel();
  out_ptr_data[idx.numel()] = node_cumsum;

  return std::make_tuple(torch::cat(out_rowptrs, 0), torch::cat(out_cols, 0),
                         torch::cat(out_n_ids, 0), torch::cat(out_e_ids, 0),
                         out_ptr, out_root_n_id);
}

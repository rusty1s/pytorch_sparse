#include "sample_cpu.h"

#include "utils.h"

// Returns `rowptr`, `col`, `n_id`, `e_id`,
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sample_adj_cpu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
               int64_t num_neighbors, bool replace) {
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  CHECK_CPU(idx);
  CHECK_INPUT(idx.dim() == 1);

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  auto idx_data = idx.data_ptr<int64_t>();

  auto out_rowptr = torch::empty(idx.numel() + 1, rowptr.options());
  auto out_rowptr_data = out_rowptr.data_ptr<int64_t>();
  out_rowptr_data[0] = 0;

  std::vector<std::multiset<int64_t>> cols;
  std::vector<int64_t> n_ids;
  std::unordered_map<int64_t, int64_t> n_id_map;
  std::vector<int64_t> e_ids;

  int64_t i;
  for (int64_t n = 0; n < idx.numel(); n++) {
    i = idx_data[n];
    cols.push_back(std::multiset<int64_t>());
    n_id_map[i] = n;
    n_ids.push_back(i);
  }

  int64_t n, c, e, row_start, row_end, row_count;

  if (num_neighbors < 0) { // No sampling ======================================

    for (int64_t i = 0; i < idx.numel(); i++) {
      n = idx_data[i];
      row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
      row_count = row_end - row_start;

      for (int64_t j = 0; j < row_count; j++) {
        e = row_start + j;
        c = col_data[e];

        if (n_id_map.count(c) == 0) {
          n_id_map[c] = n_ids.size();
          n_ids.push_back(c);
        }

        cols[i].insert(n_id_map[c]);
        e_ids.push_back(e);
      }
      out_rowptr_data[i + 1] = out_rowptr_data[i] + cols[i].size();
    }
  }

  else if (replace) { // Sample with replacement ===============================

    for (int64_t i = 0; i < idx.numel(); i++) {
      n = idx_data[i];
      row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
      row_count = row_end - row_start;

      for (int64_t j = 0; j < num_neighbors; j++) {
        e = row_start + rand() % row_count;
        c = col_data[e];

        if (n_id_map.count(c) == 0) {
          n_id_map[c] = n_ids.size();
          n_ids.push_back(c);
        }

        cols[i].insert(n_id_map[c]);
        e_ids.push_back(c);
      }
      out_rowptr_data[i + 1] = out_rowptr_data[i] + cols[i].size();
    }

  } else { // Sample without replacement via Robert Floyd algorithm ============

    for (int64_t i = 0; i < idx.numel(); i++) {
      n = idx_data[i];
      row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
      row_count = row_end - row_start;

      std::unordered_set<int64_t> perm;
      if (row_count <= num_neighbors) {
        for (int64_t j = 0; j < row_count; j++)
          perm.insert(j);
      } else { // See: https://www.nowherenearithaca.com/2013/05/
               //      robert-floyds-tiny-and-beautiful.html
        for (int64_t j = row_count - num_neighbors; j < row_count; j++) {
          if (!perm.insert(rand() % j).second)
            perm.insert(j);
        }
      }

      for (const int64_t &p : perm) {
        e = row_start + p;
        c = col_data[e];

        if (n_id_map.count(c) == 0) {
          n_id_map[c] = n_ids.size();
          n_ids.push_back(c);
        }

        cols[i].insert(n_id_map[c]);
        e_ids.push_back(c);
      }
      out_rowptr_data[i + 1] = out_rowptr_data[i] + cols[i].size();
    }
  }

  int64_t n_len = n_ids.size(), e_len = e_ids.size();
  auto n_id = torch::from_blob(n_ids.data(), {n_len}, col.options()).clone();
  auto e_id = torch::from_blob(e_ids.data(), {e_len}, col.options()).clone();

  auto out_col = torch::empty(e_len, col.options());
  auto out_col_data = out_col.data_ptr<int64_t>();

  i = 0;
  for (const std::multiset<int64_t> &col_set : cols) {
    for (const int64_t &c : col_set) {
      out_col_data[i] = c;
      i += 1;
    }
  }

  return std::make_tuple(out_rowptr, out_col, n_id, e_id);
}

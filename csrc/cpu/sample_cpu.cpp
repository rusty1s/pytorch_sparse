#include "sample_cpu.h"

#include "utils.h"

// Returns `rowptr`, `col`, `n_id`, `e_id`,
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sample_adj_cpu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor rowcount,
               torch::Tensor idx, int64_t num_neighbors, bool replace) {
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  CHECK_CPU(idx);
  CHECK_INPUT(idx.dim() == 1);

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  auto rowcount_data = rowcount.data_ptr<int64_t>();
  auto idx_data = idx.data_ptr<int64_t>();

  auto out_rowptr = torch::empty(idx.size(0) + 1, rowptr.options());
  auto out_rowptr_data = out_rowptr.data_ptr<int64_t>();
  out_rowptr_data[0] = 0;

  std::vector<int64_t> cols;
  std::vector<int64_t> n_ids;
  std::unordered_map<int64_t, int64_t> n_id_map;
  std::vector<int64_t> e_ids;

  int64_t i;
  for (int64_t n = 0; n < idx.size(0); n++) {
    i = idx_data[n];
    n_id_map[i] = n;
    n_ids.push_back(i);
  }

  if (num_neighbors < 0) { // No sampling ======================================

    int64_t r, c, e, offset = 0;
    for (int64_t i = 0; i < idx.size(0); i++) {
      r = idx_data[i];

      for (int64_t j = 0; j < rowcount_data[r]; j++) {
        e = rowptr_data[r] + j;
        c = col_data[e];

        if (n_id_map.count(c) == 0) {
          n_id_map[c] = n_ids.size();
          n_ids.push_back(c);
        }

        cols.push_back(n_id_map[c]);
        e_ids.push_back(e);
      }
      offset = cols.size();
      out_rowptr_data[i + 1] = offset;
    }
  }

  else if (replace) { // Sample with replacement ===============================

    int64_t r, c, e, offset = 0;
    for (int64_t i = 0; i < idx.size(0); i++) {
      r = idx_data[i];

      for (int64_t j = 0; j < num_neighbors; j++) {
        e = rowptr_data[r] + rand() % rowcount_data[r];
        c = col_data[e];

        if (n_id_map.count(c) == 0) {
          n_id_map[c] = n_ids.size();
          n_ids.push_back(c);
        }

        c = n_id_map[c];
        if (std::find(cols.begin() + offset, cols.end(), c) == cols.end()) {
          cols.push_back(c);
          e_ids.push_back(e);
        }
      }
      offset = cols.size();
      out_rowptr_data[i + 1] = offset;
    }

  } else { // Sample without replacement via Robert Floyd algorithm ============

    int64_t r, c, e, rc, offset = 0;
    for (int64_t i = 0; i < idx.size(0); i++) {
      r = idx_data[i];
      rc = rowcount_data[r];

      std::unordered_set<int64_t> perm;
      if (rc <= num_neighbors) {
        for (int64_t x = 0; x < rc; x++) {
          perm.insert(x);
        }
      } else {
        for (int64_t x = rc - std::min(rc, num_neighbors); x < rc; x++) {
          if (!perm.insert(rand() % x).second) {
            perm.insert(x);
          }
        }
      }

      for (const int64_t &p : perm) {
        e = rowptr_data[r] + p;
        c = col_data[e];

        if (n_id_map.count(c) == 0) {
          n_id_map[c] = n_ids.size();
          n_ids.push_back(c);
        }

        cols.push_back(n_id_map[c]);
        e_ids.push_back(e);
      }
      offset = cols.size();
      out_rowptr_data[i + 1] = offset;
    }
  }

  int64_t n_len = n_ids.size(), e_len = cols.size();
  col = torch::from_blob(cols.data(), {e_len}, col.options()).clone();
  auto n_id = torch::from_blob(n_ids.data(), {n_len}, col.options()).clone();
  auto e_id = torch::from_blob(e_ids.data(), {e_len}, col.options()).clone();

  return std::make_tuple(out_rowptr, col, n_id, e_id);
}

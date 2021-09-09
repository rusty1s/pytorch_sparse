#include "sample_cpu.h"

#include "utils.h"

#ifdef _WIN32
#include <process.h>
#endif

// Returns `rowptr`, `col`, `n_id`, `e_id`
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sample_adj_cpu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
               int64_t num_neighbors, bool replace) {
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  CHECK_CPU(idx);
  CHECK_INPUT(idx.dim() == 1);

  srand(time(NULL) + 1000 * getpid()); // Initialize random seed.

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  auto idx_data = idx.data_ptr<int64_t>();

  auto out_rowptr = torch::empty(idx.numel() + 1, rowptr.options());
  auto out_rowptr_data = out_rowptr.data_ptr<int64_t>();
  out_rowptr_data[0] = 0;

  std::vector<std::vector<std::tuple<int64_t, int64_t>>> cols; // col, e_id
  std::vector<int64_t> n_ids;
  std::unordered_map<int64_t, int64_t> n_id_map;

  int64_t i;
  for (int64_t n = 0; n < idx.numel(); n++) {
    i = idx_data[n];
    cols.push_back(std::vector<std::tuple<int64_t, int64_t>>());
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
        cols[i].push_back(std::make_tuple(n_id_map[c], e));
      }
      out_rowptr_data[i + 1] = out_rowptr_data[i] + cols[i].size();
    }
  }

  else if (replace) { // Sample with replacement ===============================

    for (int64_t i = 0; i < idx.numel(); i++) {
      n = idx_data[i];
      row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
      row_count = row_end - row_start;

      if (row_count > 0) {
        for (int64_t j = 0; j < num_neighbors; j++) {
          e = row_start + rand() % row_count;
          c = col_data[e];

          if (n_id_map.count(c) == 0) {
            n_id_map[c] = n_ids.size();
            n_ids.push_back(c);
          }
          cols[i].push_back(std::make_tuple(n_id_map[c], e));
        }
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
        cols[i].push_back(std::make_tuple(n_id_map[c], e));
      }
      out_rowptr_data[i + 1] = out_rowptr_data[i] + cols[i].size();
    }
  }

  int64_t N = n_ids.size();
  auto out_n_id = torch::from_blob(n_ids.data(), {N}, col.options()).clone();

  int64_t E = out_rowptr_data[idx.numel()];
  auto out_col = torch::empty(E, col.options());
  auto out_col_data = out_col.data_ptr<int64_t>();
  auto out_e_id = torch::empty(E, col.options());
  auto out_e_id_data = out_e_id.data_ptr<int64_t>();

  i = 0;
  for (std::vector<std::tuple<int64_t, int64_t>> &col_vec : cols) {
    std::sort(col_vec.begin(), col_vec.end(),
              [](const std::tuple<int64_t, int64_t> &a,
                 const std::tuple<int64_t, int64_t> &b) -> bool {
                return std::get<0>(a) < std::get<0>(b);
              });
    for (const std::tuple<int64_t, int64_t> &value : col_vec) {
      out_col_data[i] = std::get<0>(value);
      out_e_id_data[i] = std::get<1>(value);
      i += 1;
    }
  }

  return std::make_tuple(out_rowptr, out_col, out_n_id, out_e_id);
}

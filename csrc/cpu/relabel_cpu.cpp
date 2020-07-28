#include "relabel_cpu.h"

#include "utils.h"

std::tuple<torch::Tensor, torch::Tensor> relabel_cpu(torch::Tensor col,
                                                     torch::Tensor idx) {

  CHECK_CPU(col);
  CHECK_CPU(idx);
  CHECK_INPUT(idx.dim() == 1);

  auto col_data = col.data_ptr<int64_t>();
  auto idx_data = idx.data_ptr<int64_t>();

  std::vector<int64_t> cols;
  std::vector<int64_t> n_ids;
  std::unordered_map<int64_t, int64_t> n_id_map;

  int64_t i;
  for (int64_t n = 0; n < idx.size(0); n++) {
    i = idx_data[n];
    n_id_map[i] = n;
    n_ids.push_back(i);
  }

  int64_t c;
  for (int64_t e = 0; e < col.size(0); e++) {
    c = col_data[e];

    if (n_id_map.count(c) == 0) {
      n_id_map[c] = n_ids.size();
      n_ids.push_back(c);
    }

    cols.push_back(n_id_map[c]);
  }

  int64_t n_len = n_ids.size(), e_len = cols.size();
  auto out_col = torch::from_blob(cols.data(), {e_len}, col.options()).clone();
  auto out_idx = torch::from_blob(n_ids.data(), {n_len}, col.options()).clone();

  return std::make_tuple(out_col, out_idx);
}

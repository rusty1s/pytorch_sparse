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

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>,
           torch::Tensor>
relabel_one_hop_cpu(torch::Tensor rowptr, torch::Tensor col,
                    torch::optional<torch::Tensor> optional_value,
                    torch::Tensor idx, bool bipartite) {

  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  if (optional_value.has_value()) {
    CHECK_CPU(optional_value.value());
    CHECK_INPUT(optional_value.value().dim() == 1);
  }
  CHECK_CPU(idx);

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  auto idx_data = idx.data_ptr<int64_t>();

  std::vector<int64_t> n_ids;
  std::unordered_map<int64_t, int64_t> n_id_map;
  std::unordered_map<int64_t, int64_t>::iterator it;

  auto out_rowptr = torch::empty(idx.numel() + 1, rowptr.options());
  auto out_rowptr_data = out_rowptr.data_ptr<int64_t>();

  out_rowptr_data[0] = 0;
  int64_t v, w, c, row_start, row_end, offset = 0;
  for (int64_t i = 0; i < idx.numel(); i++) {
    v = idx_data[i];
    n_id_map[v] = i;
    offset += rowptr_data[v + 1] - rowptr_data[v];
    out_rowptr_data[i + 1] = offset;
  }

  auto out_col = torch::empty(offset, col.options());
  auto out_col_data = out_col.data_ptr<int64_t>();

  torch::optional<torch::Tensor> out_value = torch::nullopt;
  if (optional_value.has_value()) {
    out_value = torch::empty(offset, optional_value.value().options());

    AT_DISPATCH_ALL_TYPES(optional_value.value().scalar_type(), "relabel", [&] {
      auto value_data = optional_value.value().data_ptr<scalar_t>();
      auto out_value_data = out_value.value().data_ptr<scalar_t>();

      offset = 0;
      for (int64_t i = 0; i < idx.numel(); i++) {
        v = idx_data[i];
        row_start = rowptr_data[v], row_end = rowptr_data[v + 1];

        for (int64_t j = row_start; j < row_end; j++) {
          w = col_data[j];
          it = n_id_map.find(w);
          if (it == n_id_map.end()) {
            c = idx.numel() + n_ids.size();
            n_id_map[w] = c;
            n_ids.push_back(w);
            out_col_data[offset] = c;
          } else {
            out_col_data[offset] = it->second;
          }
          out_value_data[offset] = value_data[j];
          offset++;
        }
      }
    });

  } else {
    offset = 0;
    for (int64_t i = 0; i < idx.numel(); i++) {
      v = idx_data[i];
      row_start = rowptr_data[v], row_end = rowptr_data[v + 1];

      for (int64_t j = row_start; j < row_end; j++) {
        w = col_data[j];
        it = n_id_map.find(w);
        if (it == n_id_map.end()) {
          c = idx.numel() + n_ids.size();
          n_id_map[w] = c;
          n_ids.push_back(w);
          out_col_data[offset] = c;
        } else {
          out_col_data[offset] = it->second;
        }
        offset++;
      }
    }
  }

  if (!bipartite)
    out_rowptr = torch::cat(
        {out_rowptr, torch::full({(int64_t)n_ids.size()}, out_col.numel(),
                                 rowptr.options())});

  idx = torch::cat({idx, torch::from_blob(n_ids.data(), {(int64_t)n_ids.size()},
                                          idx.options())});

  return std::make_tuple(out_rowptr, out_col, out_value, idx);
}

#include "saint_cpu.h"

#include "utils.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
subgraph_cpu(torch::Tensor idx, torch::Tensor rowptr, torch::Tensor row,
             torch::Tensor col) {
  CHECK_CPU(idx);
  CHECK_CPU(rowptr);
  CHECK_CPU(col);

  CHECK_INPUT(idx.dim() == 1);
  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);

  auto assoc = torch::full({rowptr.size(0) - 1}, -1, idx.options());
  assoc.index_copy_(0, idx, torch::arange(idx.size(0), idx.options()));

  auto idx_data = idx.data_ptr<int64_t>();
  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  auto assoc_data = assoc.data_ptr<int64_t>();

  std::vector<int64_t> rows, cols, indices;

  int64_t v, w, w_new, row_start, row_end;
  for (int64_t v_new = 0; v_new < idx.size(0); v_new++) {
    v = idx_data[v_new];
    row_start = rowptr_data[v];
    row_end = rowptr_data[v + 1];

    for (int64_t j = row_start; j < row_end; j++) {
      w = col_data[j];
      w_new = assoc_data[w];
      if (w_new > -1) {
        rows.push_back(v_new);
        cols.push_back(w_new);
        indices.push_back(j);
      }
    }
  }

  int64_t length = rows.size();
  row = torch::from_blob(rows.data(), {length}, row.options()).clone();
  col = torch::from_blob(cols.data(), {length}, row.options()).clone();
  idx = torch::from_blob(indices.data(), {length}, row.options()).clone();

  return std::make_tuple(row, col, idx);
}

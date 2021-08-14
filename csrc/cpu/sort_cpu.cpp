#include "sort_cpu.h"

#include "utils.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sort_cpu(const torch::Tensor &row, const torch::Tensor &col,
         const int64_t num_rows, const bool compressed) {

  const auto row_data = row.data_ptr<int64_t>();
  const auto col_data = col.data_ptr<int64_t>();

  std::vector<std::vector<int64_t>> cols(num_rows);
  std::vector<std::vector<int64_t>> edges(num_rows);
  for (int64_t i = 0; i < row.numel(); i++) {
    const auto &v = row_data[i];
    const auto &w = col_data[i];
    cols.at(v).push_back(w);
    edges.at(v).push_back(i);
  }

  return std::make_tuple(row, row, row);
}

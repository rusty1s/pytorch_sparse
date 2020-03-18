#include "rw_cpu.h"

#include "utils.h"

torch::Tensor random_walk_cpu(torch::Tensor rowptr, torch::Tensor col,
                              torch::Tensor start, int64_t walk_length) {
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  CHECK_CPU(start);

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  CHECK_INPUT(start.dim() == 1);

  auto rand = torch::rand({start.size(0), walk_length},
                          start.options().dtype(torch::kFloat));

  auto L = walk_length + 1;
  auto out = torch::full({start.size(0), L}, -1, start.options());

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  auto start_data = start.data_ptr<int64_t>();
  auto rand_data = rand.data_ptr<float>();
  auto out_data = out.data_ptr<int64_t>();

  for (auto n = 0; n < start.size(0); n++) {
    auto cur = start_data[n];
    out_data[n * L] = cur;

    int64_t row_start, row_end;
    for (auto l = 0; l < walk_length; l++) {
      row_start = rowptr_data[cur];
      row_end = rowptr_data[cur + 1];

      cur = col_data[row_start + int64_t(rand_data[n * walk_length + l] *
                                         (row_end - row_start))];
      out_data[n * L + l + 1] = cur;
    }
  }

  return out;
}

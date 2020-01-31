#include "diag_cpu.h"

#include "utils.h"

torch::Tensor non_diag_mask_cpu(torch::Tensor row, torch::Tensor col, int64_t M,
                                int64_t N, int64_t k) {
  CHECK_CPU(row);
  CHECK_CPU(col);

  auto E = row.size(0);
  auto num_diag = k < 0 ? std::min(M + k, N) : std::min(M, N - k);

  auto row_data = row.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();

  auto mask = torch::zeros(E + num_diag, row.options().dtype(torch::kBool));
  auto mask_data = mask.data_ptr<bool>();

  int64_t r, c;
  if (k < 0) {
    for (int64_t i = 0; i < E; i++) {
      r = row_data[i], c = col_data[i];
      if (r + k < 0) {
        mask_data[i] = true;
      } else if (r + k >= N) {
        mask_data[i + num_diag] = true;
      } else if (r + k > c) {
        mask_data[i + r + k] = true;
      } else if (r + k < c) {
        mask_data[i + r + k + 1] = true;
      }
    }
  } else {
    for (int64_t i = 0; i < E; i++) {
      r = row_data[i], c = col_data[i];
      if (r + k >= N) {
        mask_data[i + num_diag] = true;
      } else if (r + k > c) {
        mask_data[i + r] = true;
      } else if (r + k < c) {
        mask_data[i + r + 1] = true;
      }
    }
  }

  return mask;
}

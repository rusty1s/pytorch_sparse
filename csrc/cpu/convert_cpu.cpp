#include "convert_cpu.h"

#include "utils.h"

torch::Tensor ind2ptr_cpu(torch::Tensor ind, int64_t M) {
  CHECK_CPU(ind);
  auto out = torch::empty(M + 1, ind.options());
  auto ind_data = ind.data_ptr<int64_t>();
  auto out_data = out.data_ptr<int64_t>();

  int64_t numel = ind.numel(), idx = ind_data[0], next_idx;
  for (auto i = 0; i <= idx; i++)
    out_data[i] = 0;

  for (auto i = 0; i < numel - 1; i++) {
    next_idx = ind_data[i + 1];
    for (auto j = idx; j < next_idx; j++)
      out_data[j + 1] = i + 1;
    idx = next_idx;
  }

  for (auto i = idx + 1; i < M + 1; i++)
    out_data[i] = numel;

  return out;
}

torch::Tensor ptr2ind_cpu(torch::Tensor ptr, int64_t E) {
  CHECK_CPU(ptr);
  auto out = torch::empty(E, ptr.options());
  auto ptr_data = ptr.data_ptr<int64_t>();
  auto out_data = out.data_ptr<int64_t>();

  int64_t idx = ptr_data[0], next_idx;
  for (auto i = 0; i < ptr.numel() - 1; i++) {
    next_idx = ptr_data[i + 1];
    for (auto e = idx; e < next_idx; e++)
      out_data[e] = i;
    idx = next_idx;
  }

  return out;
}

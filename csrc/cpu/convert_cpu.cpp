#include "convert_cpu.h"

#include <ATen/Parallel.h>

#include "utils.h"

torch::Tensor ind2ptr_cpu(torch::Tensor ind, int64_t M) {
  CHECK_CPU(ind);
  auto out = torch::empty(M + 1, ind.options());
  auto ind_data = ind.data_ptr<int64_t>();
  auto out_data = out.data_ptr<int64_t>();

  int64_t numel = ind.numel();

  if (numel == 0)
    return out.zero_();

  for (int64_t i = 0; i <= ind_data[0]; i++)
    out_data[i] = 0;

  int64_t grain_size = at::internal::GRAIN_SIZE;
  at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
    int64_t idx = ind_data[begin], next_idx;
    for (int64_t i = begin; i < std::min(end, numel - 1); i++) {
      next_idx = ind_data[i + 1];
      for (; idx < next_idx; idx++)
        out_data[idx + 1] = i + 1;
    }
  });

  for (int64_t i = ind_data[numel - 1] + 1; i < M + 1; i++)
    out_data[i] = numel;

  return out;
}

torch::Tensor ptr2ind_cpu(torch::Tensor ptr, int64_t E) {
  CHECK_CPU(ptr);
  auto out = torch::empty(E, ptr.options());
  auto ptr_data = ptr.data_ptr<int64_t>();
  auto out_data = out.data_ptr<int64_t>();

  int64_t numel = ptr.numel();

  int64_t grain_size = at::internal::GRAIN_SIZE;
  at::parallel_for(0, numel - 1, grain_size, [&](int64_t begin, int64_t end) {
    int64_t idx = ptr_data[begin], next_idx;
    for (int64_t i = begin; i < end; i++) {
      next_idx = ptr_data[i + 1];
      for (int64_t e = idx; e < next_idx; e++)
        out_data[e] = i;
      idx = next_idx;
    }
  });

  return out;
}

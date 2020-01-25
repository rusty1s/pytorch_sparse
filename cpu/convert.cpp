#include <torch/extension.h>

#include "compat.h"

#define CHECK_CPU(x) AT_ASSERTM(!x.type().is_cuda(), #x " must be CPU tensor")

at::Tensor ind2ptr(at::Tensor ind, int64_t M) {
  CHECK_CPU(ind);
  auto out = at::empty(M + 1, ind.options());
  auto ind_data = ind.DATA_PTR<int64_t>();
  auto out_data = out.DATA_PTR<int64_t>();

  int64_t numel = ind.numel(), idx = ind_data[0], next_idx;

  for (int64_t i = 0; i <= idx; i++)
    out_data[i] = 0;

  for (int64_t i = 0; i < numel - 1; i++) {
    next_idx = ind_data[i + 1];
    for (int64_t j = idx; j < next_idx; j++)
      out_data[j + 1] = i + 1;
    idx = next_idx;
  }

  for (int64_t i = idx + 1; i < M + 1; i++)
    out_data[i] = numel;

  return out;
}

at::Tensor ptr2ind(at::Tensor ptr, int64_t E) {
  CHECK_CPU(ptr);
  auto out = at::empty(E, ptr.options());
  auto ptr_data = ptr.DATA_PTR<int64_t>();
  auto out_data = out.DATA_PTR<int64_t>();

  int64_t idx = ptr_data[0], next_idx;
  for (int64_t i = 0; i < ptr.numel() - 1; i++) {
    next_idx = ptr_data[i + 1];
    for (int64_t e = idx; e < next_idx; e++)
      out_data[e] = i;
    idx = next_idx;
  }

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ind2ptr", &ind2ptr, "Ind2Ptr (CPU)");
  m.def("ptr2ind", &ptr2ind, "Ptr2Ind (CPU)");
}

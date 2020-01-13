#include <torch/extension.h>

#include "compat.h"

#define CHECK_CPU(x) AT_ASSERTM(!x.type().is_cuda(), #x " must be CPU tensor")

at::Tensor rowptr(at::Tensor row, int64_t size) {
  CHECK_CPU(row);
  AT_ASSERTM(row.dim() == 1, "Row needs to be one-dimensional");

  auto out = at::empty(size + 1, row.options());
  auto row_data = row.DATA_PTR<int64_t>();
  auto out_data = out.DATA_PTR<int64_t>();

  int64_t numel = row.numel(), idx = row_data[0], next_idx;
  for (int64_t i = 0; i <= idx; i++)
    out_data[i] = 0;

  for (int64_t i = 0; i < row.size(0) - 1; i++) {
    next_idx = row_data[i + 1];
    for (int64_t j = idx; j < next_idx; j++)
      out_data[j + 1] = i + 1;
    idx = next_idx;
  }

  for (int64_t i = idx + 1; i < size + 1; i++)
    out_data[i] = numel;

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rowptr", &rowptr, "Rowptr (CPU)");
}

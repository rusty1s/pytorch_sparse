#include <torch/extension.h>

#include "compat.h"

at::Tensor arange_interleave(at::Tensor start, at::Tensor repeat) {
  auto count = repeat.sum().DATA_PTR<int64_t>()[0];
  auto out = at::empty(count, start.options());

  auto repeat_data = repeat.DATA_PTR<int64_t>();

  AT_DISPATCH_ALL_TYPES(start.scalar_type(), "arange_interleave", [&] {
    auto start_data = start.DATA_PTR<scalar_t>();
    auto out_data = out.DATA_PTR<scalar_t>();

    int i = 0;
    for (int start_idx = 0; start_idx < start.size(0); start_idx++) {
      scalar_t init = start_data[start_idx];
      for (scalar_t rep_idx = 0; rep_idx < repeat_data[start_idx]; rep_idx++) {
        out_data[i] = init + rep_idx;
        i++;
      }
    }
  });

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("arange_interleave", &arange_interleave, "Arange Interleave (CPU)");
}

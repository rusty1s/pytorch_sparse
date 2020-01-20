#include <torch/extension.h>

#include "compat.h"

#define CHECK_CPU(x) AT_ASSERTM(!x.type().is_cuda(), #x " must be CPU tensor")

at::Tensor non_diag_mask(at::Tensor index, int64_t M, int64_t N, int64_t k) {
  CHECK_CPU(index);

  int64_t E = index.size(1);

  index = index.contiguous();
  auto index_data = index.DATA_PTR<int64_t>();

  int64_t num_diag = k < 0 ? std::min(M + k, N) : std::min(M, N - k);

  auto mask = at::zeros(E + num_diag, index.options().dtype(at::kBool));
  auto mask_data = mask.DATA_PTR<bool>();

  int64_t r, c;
  if (k < 0) {
    for (int64_t i = 0; i < E; i++) {
      r = index_data[i], c = index_data[i + E];
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
      r = index_data[i], c = index_data[i + E];
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("non_diag_mask", &non_diag_mask, "Non-Diagonal Mask (CPU)");
}

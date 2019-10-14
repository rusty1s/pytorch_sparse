#include <torch/extension.h>

#include "compat.h"

at::Tensor degree(at::Tensor row, int64_t num_nodes) {
  auto zero = at::zeros(num_nodes, row.options());
  auto one = at::ones(row.size(0), row.options());
  return zero.scatter_add_(0, row, one);
}

std::tuple<at::Tensor, at::Tensor> to_csr(at::Tensor row, at::Tensor col,
                                          int64_t num_nodes) {
  // Assert already coalesced input.
  row = degree(row, num_nodes).cumsum(0);
  row = at::cat({at::zeros(1, row.options()), row}, 0); // Prepend zero.
  return std::make_tuple(row, col);
}

at::Tensor spspmm_bw(at::Tensor index, at::Tensor indexA, at::Tensor valueA,
                     at::Tensor indexB, at::Tensor valueB, size_t rowA_max,
                     size_t rowB_max) {

  int64_t *index_data = index.DATA_PTR<int64_t>();
  auto value = at::zeros(index.size(1), valueA.options());

  at::Tensor rowA, colA;
  std::tie(rowA, colA) = to_csr(indexA[0], indexA[1], rowA_max);
  int64_t *rowA_data = rowA.DATA_PTR<int64_t>();
  int64_t *colA_data = colA.DATA_PTR<int64_t>();

  at::Tensor rowB, colB;
  std::tie(rowB, colB) = to_csr(indexB[0], indexB[1], rowB_max);
  int64_t *rowB_data = rowB.DATA_PTR<int64_t>();
  int64_t *colB_data = colB.DATA_PTR<int64_t>();

  AT_DISPATCH_FLOATING_TYPES(valueA.scalar_type(), "spspmm_bw", [&] {
    scalar_t *value_data = value.DATA_PTR<scalar_t>();
    scalar_t *valueA_data = valueA.DATA_PTR<scalar_t>();
    scalar_t *valueB_data = valueB.DATA_PTR<scalar_t>();

    for (int64_t e = 0; e < value.size(0); e++) {
      int64_t i = index_data[e], j = index_data[value.size(0) + e];

      for (ptrdiff_t dA = rowA_data[i]; dA < rowA_data[i + 1]; dA++) {
        int64_t cA = colA_data[dA];

        for (ptrdiff_t dB = rowB_data[j]; dB < rowB_data[j + 1]; dB++) {
          int64_t cB = colB_data[dB];

          if (cA == cB) {
            value_data[e] += valueA_data[dA] * valueB_data[dB];
          }

          if (cB >= cA) {
            break;
          }
        }
      }
    }
  });

  return value;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spspmm_bw", &spspmm_bw,
        "Sparse-Sparse Matrix Multiplication Backward (CPU)");
}

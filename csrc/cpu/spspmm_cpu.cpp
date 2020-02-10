#include "spspmm_cpu.h"

#include "utils.h"

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
spspmm_cpu(torch::Tensor rowptrA, torch::Tensor colA,
           torch::optional<torch::Tensor> optional_valueA,
           torch::Tensor rowptrB, torch::Tensor colB,
           torch::optional<torch::Tensor> optional_valueB, int64_t K,
           std::string reduce) {

  CHECK_CPU(rowptrA);
  CHECK_CPU(colA);
  if (optional_valueA.has_value())
    CHECK_CPU(optional_valueA.value());
  CHECK_CPU(rowptrB);
  CHECK_CPU(colB);
  if (optional_valueB.has_value())
    CHECK_CPU(optional_valueB.value());

  CHECK_INPUT(rowptrA.dim() == 1);
  CHECK_INPUT(colA.dim() == 1);
  if (optional_valueA.has_value()) {
    CHECK_INPUT(optional_valueA.value().dim() == 1);
    CHECK_INPUT(optional_valueA.value().size(0) == colA.size(0));
  }
  CHECK_INPUT(rowptrB.dim() == 1);
  CHECK_INPUT(colB.dim() == 1);
  if (optional_valueB.has_value()) {
    CHECK_INPUT(optional_valueB.value().dim() == 1);
    CHECK_INPUT(optional_valueB.value().size(0) == colB.size(0));
  }

  if (!optional_valueA.has_value() && optional_valueB.has_value())
    optional_valueA =
        torch::ones(colA.numel(), optional_valueB.value().options());

  if (!optional_valueB.has_value() && optional_valueA.has_value())
    optional_valueB =
        torch::ones(colB.numel(), optional_valueA.value().options());

  auto scalar_type = torch::ScalarType::Float;
  if (optional_valueA.has_value())
    scalar_type = optional_valueA.value().scalar_type();

  auto rowptrA_data = rowptrA.data_ptr<int64_t>();
  auto colA_data = colA.data_ptr<int64_t>();
  auto rowptrB_data = rowptrB.data_ptr<int64_t>();
  auto colB_data = colB.data_ptr<int64_t>();

  // Pass 1: Compute CSR row pointer.
  auto rowptrC = torch::empty_like(rowptrA);
  auto rowptrC_data = rowptrC.data_ptr<int64_t>();
  rowptrC_data[0] = 0;

  std::vector<int64_t> mask(K, -1);
  int64_t nnz = 0, row_nnz, rowA_start, rowA_end, rowB_start, rowB_end, cA, cB;
  for (auto n = 0; n < rowptrA.numel() - 1; n++) {
    row_nnz = 0;

    for (auto eA = rowptrA_data[n]; eA < rowptrA_data[n + 1]; eA++) {
      cA = colA_data[eA];
      for (auto eB = rowptrB_data[cA]; eB < rowptrB_data[cA + 1]; eB++) {
        cB = colB_data[eB];
        if (mask[cB] != n) {
          mask[cB] = n;
          row_nnz++;
        }
      }
    }

    nnz += row_nnz;
    rowptrC_data[n + 1] = nnz;
  }

  // Pass 2: Compute CSR entries.
  auto colC = torch::empty(nnz, rowptrC.options());
  auto colC_data = colC.data_ptr<int64_t>();

  torch::optional<torch::Tensor> optional_valueC = torch::nullopt;
  if (optional_valueA.has_value())
    optional_valueC = torch::empty(nnz, optional_valueA.value().options());

  AT_DISPATCH_ALL_TYPES(scalar_type, "spspmm", [&] {
    AT_DISPATCH_HAS_VALUE(optional_valueC, [&] {
      scalar_t *valA_data = nullptr, *valB_data = nullptr, *valC_data = nullptr;
      if (HAS_VALUE) {
        valA_data = optional_valueA.value().data_ptr<scalar_t>();
        valB_data = optional_valueB.value().data_ptr<scalar_t>();
        valC_data = optional_valueC.value().data_ptr<scalar_t>();
      }
      scalar_t valA;

      rowA_start = 0, nnz = 0;
      std::vector<scalar_t> vals(K, 0);
      for (auto n = 1; n < rowptrA.numel(); n++) {
        rowA_end = rowptrA_data[n];

        for (auto eA = rowA_start; eA < rowA_end; eA++) {
          cA = colA_data[eA];
          if (HAS_VALUE)
            valA = valA_data[eA];

          rowB_start = rowptrB_data[cA], rowB_end = rowptrB_data[cA + 1];
          for (auto eB = rowB_start; eB < rowB_end; eB++) {
            cB = colB_data[eB];
            if (HAS_VALUE)
              vals[cB] += valA * valB_data[eB];
            else
              vals[cB] += 1;
          }
        }

        for (auto k = 0; k < K; k++) {
          if (vals[k] != 0) {
            colC_data[nnz] = k;
            if (HAS_VALUE)
              valC_data[nnz] = vals[k];
            nnz++;
          }
          vals[k] = (scalar_t)0;
        }

        rowA_start = rowA_end;
      }
    });
  });

  return std::make_tuple(rowptrC, colC, optional_valueC);
}

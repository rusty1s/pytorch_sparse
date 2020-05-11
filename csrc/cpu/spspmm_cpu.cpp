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

  auto rowptrC = torch::empty_like(rowptrA);
  auto rowptrC_data = rowptrC.data_ptr<int64_t>();
  rowptrC_data[0] = 0;

  torch::Tensor colC;
  torch::optional<torch::Tensor> optional_valueC = torch::nullopt;

  AT_DISPATCH_ALL_TYPES(scalar_type, "spspmm", [&] {
    AT_DISPATCH_HAS_VALUE(optional_valueA, [&] {
      scalar_t *valA_data = nullptr, *valB_data = nullptr;
      if (HAS_VALUE) {
        valA_data = optional_valueA.value().data_ptr<scalar_t>();
        valB_data = optional_valueB.value().data_ptr<scalar_t>();
      }

      int64_t nnz = 0, cA, cB;
      std::vector<scalar_t> tmp_vals(K, 0);
      std::vector<int64_t> cols;
      std::vector<scalar_t> vals;

      for (auto rA = 0; rA < rowptrA.numel() - 1; rA++) {
        for (auto eA = rowptrA_data[rA]; eA < rowptrA_data[rA + 1]; eA++) {
          cA = colA_data[eA];
          for (auto eB = rowptrB_data[cA]; eB < rowptrB_data[cA + 1]; eB++) {
            cB = colB_data[eB];

            if (HAS_VALUE)
              tmp_vals[cB] += valA_data[eA] * valB_data[eB];
            else
              tmp_vals[cB]++;
          }
        }

        for (auto k = 0; k < K; k++) {
          if (tmp_vals[k] != 0) {
            cols.push_back(k);
            if (HAS_VALUE)
              vals.push_back(tmp_vals[k]);
            nnz++;
          }
          tmp_vals[k] = (scalar_t)0;
        }
        rowptrC_data[rA + 1] = nnz;
      }

      colC = torch::from_blob(cols.data(), {nnz}, colA.options()).clone();
      if (HAS_VALUE) {
        optional_valueC = torch::from_blob(vals.data(), {nnz},
                                           optional_valueA.value().options());
        optional_valueC = optional_valueC.value().clone();
      }
    });
  });

  return std::make_tuple(rowptrC, colC, optional_valueC);
}

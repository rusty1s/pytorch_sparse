#include "spmm_cpu.h"

#include <ATen/Parallel.h>

#include "reducer.h"
#include "utils.h"

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_cpu(torch::Tensor rowptr, torch::Tensor col,
         torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
         std::string reduce) {
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  if (optional_value.has_value())
    CHECK_CPU(optional_value.value());
  CHECK_CPU(mat);

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  if (optional_value.has_value()) {
    CHECK_INPUT(optional_value.value().dim() == 1);
    CHECK_INPUT(optional_value.value().size(0) == col.size(0));
  }
  CHECK_INPUT(mat.dim() >= 2);

  mat = mat.contiguous();

  auto sizes = mat.sizes().vec();
  sizes[mat.dim() - 2] = rowptr.numel() - 1;
  auto out = torch::empty(sizes, mat.options());

  torch::optional<torch::Tensor> arg_out = torch::nullopt;
  int64_t *arg_out_data = nullptr;
  if (reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX) {
    arg_out = torch::full_like(out, col.numel(), rowptr.options());
    arg_out_data = arg_out.value().data_ptr<int64_t>();
  }

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();

  auto M = rowptr.numel() - 1;
  auto N = mat.size(-2);
  auto K = mat.size(-1);
  auto B = mat.numel() / (N * K);

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, mat.scalar_type(), "_", [&] {
    scalar_t *value_data = nullptr;
    auto mat_data = mat.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      AT_DISPATCH_HAS_VALUE(optional_value, [&] {
        if (HAS_VALUE) {
          value_data = optional_value.value().data_ptr<scalar_t>();
        }

        int64_t grain_size = at::internal::GRAIN_SIZE /
                             (K * std::max(col.numel() / M, (int64_t)1));
        at::parallel_for(0, B * M, grain_size, [&](int64_t begin, int64_t end) {
          scalar_t val;
          std::vector<scalar_t> vals(K);
          int64_t row_start, row_end, b, m, c;
          std::vector<int64_t> args(K);

          for (auto i = begin; i < end; i++) {
            b = i / M, m = i % M;

            row_start = rowptr_data[m], row_end = rowptr_data[m + 1];

            for (auto k = 0; k < K; k++)
              vals[k] = Reducer<scalar_t, REDUCE>::init();

            auto offset = b * N * K;
            for (auto e = row_start; e < row_end; e++) {
              c = col_data[e];
              if (HAS_VALUE)
                val = value_data[e];
              for (auto k = 0; k < K; k++) {
                if (HAS_VALUE)
                  Reducer<scalar_t, REDUCE>::update(
                      &vals[k], val * mat_data[offset + c * K + k], &args[k],
                      e);
                else
                  Reducer<scalar_t, REDUCE>::update(
                      &vals[k], mat_data[offset + c * K + k], &args[k], e);
              }
            }
            offset = b * M * K + m * K;
            for (auto k = 0; k < K; k++)
              Reducer<scalar_t, REDUCE>::write(out_data + offset + k, vals[k],
                                               arg_out_data + offset + k,
                                               args[k], row_end - row_start);
          }
        });
      });
    });
  });

  return std::make_tuple(out, arg_out);
}

torch::Tensor spmm_value_bw_cpu(torch::Tensor row, torch::Tensor rowptr,
                                torch::Tensor col, torch::Tensor mat,
                                torch::Tensor grad, std::string reduce) {
  CHECK_CPU(row);
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  CHECK_CPU(mat);
  CHECK_CPU(grad);

  mat = mat.contiguous();
  grad = grad.contiguous();

  auto M = grad.size(-2);
  auto N = mat.size(-2);
  auto E = row.numel();
  auto K = mat.size(-1);
  auto B = mat.numel() / (N * K);

  auto out = torch::zeros(row.numel(), grad.options());

  auto row_data = row.data_ptr<int64_t>();
  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, mat.scalar_type(), "_", [&] {
    auto mat_data = mat.data_ptr<scalar_t>();
    auto grad_data = grad.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    scalar_t val;
    int64_t row, col;
    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      for (int b = 0; b < B; b++) {
        for (int e = 0; e < E; e++) {
          row = row_data[e], col = col_data[e], val = (scalar_t)0;
          for (int k = 0; k < K; k++) {
            val += mat_data[b * N * K + col * K + k] *
                   grad_data[b * M * K + row * K + k];
          }
          if (REDUCE == MEAN) {
            int row_start = rowptr_data[row], row_end = rowptr_data[row + 1];
            val /= (scalar_t)std::max(row_end - row_start, 1);
          }
          out_data[e] += val;
        }
      }
    });
  });

  return out;
}

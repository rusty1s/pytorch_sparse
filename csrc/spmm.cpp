#include <torch/script.h>

#include "cpu/spmm_cpu.h"

#ifdef WITH_CUDA
#include "cuda/spmm_cuda.h"
#endif

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_fw(torch::Tensor rowptr, torch::Tensor col,
        torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
        std::string reduce) {
  if (rowptr.device().is_cuda()) {
#ifdef WITH_CUDA
    return spmm_cuda(rowptr, col, optional_value, mat, reduce);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return spmm_cpu(rowptr, col, optional_value, mat, reduce);
  }
}

torch::Tensor spmm_value_bw(torch::Tensor row, torch::Tensor rowptr,
                            torch::Tensor col, torch::Tensor mat,
                            torch::Tensor grad, std::string reduce) {
  if (row.device().is_cuda()) {
#ifdef WITH_CUDA
    return spmm_value_bw_cuda(row, rowptr, col, mat, grad, reduce);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return spmm_value_bw_cpu(row, rowptr, col, mat, grad, reduce);
  }
}

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class SPMMSum : public torch::autograd::Function<SPMMSum> {
public:
  static variable_list forward(AutogradContext *ctx,
                               torch::optional<Variable> opt_row,
                               Variable rowptr, Variable col, Variable value,
                               torch::optional<Variable> opt_colptr,
                               torch::optional<Variable> opt_csr2csc,
                               Variable mat) {
    auto row = opt_row.has_value() ? opt_row.value() : torch::Tensor();
    auto colptr = opt_colptr.has_value() ? opt_colptr.value() : torch::Tensor();
    auto csr2csc =
        opt_csr2csc.has_value() ? opt_csr2csc.value() : torch::Tensor();

    torch::optional<torch::Tensor> opt_value = torch::nullopt;
    if (value.numel() > 0)
      opt_value = value;

    auto out = std::get<0>(spmm_fw(rowptr, col, opt_value, mat, "sum"));
    ctx->save_for_backward({row, rowptr, col, value, colptr, csr2csc, mat});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto row = saved[0], rowptr = saved[1], col = saved[2], value = saved[3],
         colptr = saved[4], csr2csc = saved[5], mat = saved[6];

    auto grad_value = Variable();
    if (value.numel() > 0 &&
        torch::autograd::any_variable_requires_grad({value})) {
      grad_value = spmm_value_bw(row, rowptr, col, mat, grad_out, "sum");
    }

    auto grad_mat = Variable();
    if (torch::autograd::any_variable_requires_grad({mat})) {
      torch::optional<torch::Tensor> opt_value = torch::nullopt;
      if (value.numel() > 0)
        opt_value = value.index_select(0, csr2csc);

      grad_mat = std::get<0>(spmm_fw(colptr, row.index_select(0, csr2csc),
                                     opt_value, grad_out, "sum"));
    }

    return {Variable(), Variable(), Variable(), grad_value,
            Variable(), Variable(), grad_mat};
  }
};

class SPMMMean : public torch::autograd::Function<SPMMMean> {
public:
  static variable_list forward(AutogradContext *ctx,
                               torch::optional<Variable> opt_row,
                               Variable rowptr, Variable col, Variable value,
                               torch::optional<Variable> opt_rowcount,
                               torch::optional<Variable> opt_colptr,
                               torch::optional<Variable> opt_csr2csc,
                               Variable mat) {
    auto row = opt_row.has_value() ? opt_row.value() : torch::Tensor();
    auto rowcount =
        opt_rowcount.has_value() ? opt_rowcount.value() : torch::Tensor();
    auto colptr = opt_colptr.has_value() ? opt_colptr.value() : torch::Tensor();
    auto csr2csc =
        opt_csr2csc.has_value() ? opt_csr2csc.value() : torch::Tensor();

    torch::optional<torch::Tensor> opt_value = torch::nullopt;
    if (value.numel() > 0)
      opt_value = value;

    auto out = std::get<0>(spmm_fw(rowptr, col, opt_value, mat, "mean"));
    ctx->save_for_backward(
        {row, rowptr, col, value, rowcount, colptr, csr2csc, mat});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto row = saved[0], rowptr = saved[1], col = saved[2], value = saved[3],
         rowcount = saved[4], colptr = saved[5], csr2csc = saved[6],
         mat = saved[7];

    auto grad_value = Variable();
    if (value.numel() > 0 &&
        torch::autograd::any_variable_requires_grad({value})) {
      grad_value = spmm_value_bw(row, rowptr, col, mat, grad_out, "mean");
    }

    auto grad_mat = Variable();
    if (torch::autograd::any_variable_requires_grad({mat})) {
      row = row.index_select(0, csr2csc);
      rowcount = rowcount.toType(mat.scalar_type()).index_select(0, row);
      rowcount.clamp_(1);

      if (value.numel() > 0)
        rowcount = value.index_select(0, csr2csc).div(rowcount);
      else
        rowcount.pow_(-1);

      grad_mat = std::get<0>(spmm_fw(colptr, row, rowcount, grad_out, "sum"));
    }

    return {Variable(), Variable(), Variable(), grad_value,
            Variable(), Variable(), Variable(), grad_mat};
  }
};

torch::Tensor spmm_sum(torch::optional<torch::Tensor> opt_row,
                       torch::Tensor rowptr, torch::Tensor col,
                       torch::optional<torch::Tensor> opt_value,
                       torch::optional<torch::Tensor> opt_colptr,
                       torch::optional<torch::Tensor> opt_csr2csc,
                       torch::Tensor mat) {
  // Since we cannot return an *optional* gradient, we need to convert
  // `opt_value` to an empty sized tensor first :(
  auto value = opt_value.has_value() ? opt_value.value() : torch::Tensor();
  return SPMMSum::apply(opt_row, rowptr, col, value, opt_colptr, opt_csr2csc,
                        mat)[0];
}

torch::Tensor spmm_mean(torch::optional<torch::Tensor> opt_row,
                        torch::Tensor rowptr, torch::Tensor col,
                        torch::optional<torch::Tensor> opt_value,
                        torch::optional<torch::Tensor> opt_rowcount,
                        torch::optional<torch::Tensor> opt_colptr,
                        torch::optional<torch::Tensor> opt_csr2csc,
                        torch::Tensor mat) {
  auto value = opt_value.has_value() ? opt_value.value() : torch::Tensor();
  return SPMMMean::apply(opt_row, rowptr, col, value, opt_rowcount, opt_colptr,
                         opt_csr2csc, mat)[0];
}

static auto registry = torch::RegisterOperators()
                           .op("torch_sparse::spmm_sum", &spmm_sum)
                           .op("torch_sparse::spmm_mean", &spmm_mean);

#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>

#include "cpu/spmm_cpu.h"

#ifdef WITH_CUDA
#include "cuda/spmm_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__spmm_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__spmm_cpu(void) { return NULL; }
#endif
#endif
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
                               Variable mat, bool has_value) {

    if (has_value && torch::autograd::any_variable_requires_grad({value})) {
      AT_ASSERTM(opt_row.has_value(), "Argument `row` is missing");
    }

    if (torch::autograd::any_variable_requires_grad({mat})) {
      AT_ASSERTM(opt_row.has_value(), "Argument `row` is missing");
      AT_ASSERTM(opt_colptr.has_value(), "Argument `colptr` is missing");
      AT_ASSERTM(opt_csr2csc.has_value(), "Argument `csr2csc` is missing");
    }

    auto row = opt_row.has_value() ? opt_row.value() : col;
    auto colptr = opt_colptr.has_value() ? opt_colptr.value() : col;
    auto csr2csc = opt_csr2csc.has_value() ? opt_csr2csc.value() : col;

    torch::optional<torch::Tensor> opt_value = torch::nullopt;
    if (has_value)
      opt_value = value;

    auto out = std::get<0>(spmm_fw(rowptr, col, opt_value, mat, "sum"));
    ctx->saved_data["has_value"] = has_value;
    ctx->save_for_backward({row, rowptr, col, value, colptr, csr2csc, mat});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto has_value = ctx->saved_data["has_value"].toBool();
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto row = saved[0], rowptr = saved[1], col = saved[2], value = saved[3],
         colptr = saved[4], csr2csc = saved[5], mat = saved[6];

    auto grad_value = Variable();
    if (has_value > 0 && torch::autograd::any_variable_requires_grad({value})) {
      grad_value = spmm_value_bw(row, rowptr, col, mat, grad_out, "sum");
    }

    auto grad_mat = Variable();
    if (torch::autograd::any_variable_requires_grad({mat})) {
      torch::optional<torch::Tensor> opt_value = torch::nullopt;
      if (has_value)
        opt_value = value.view({-1, 1}).index_select(0, csr2csc).view(-1);

      grad_mat = std::get<0>(spmm_fw(colptr, row.index_select(0, csr2csc),
                                     opt_value, grad_out, "sum"));
    }

    return {Variable(), Variable(), Variable(), grad_value,
            Variable(), Variable(), grad_mat,   Variable()};
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
                               Variable mat, bool has_value) {

    if (has_value && torch::autograd::any_variable_requires_grad({value})) {
      AT_ASSERTM(opt_row.has_value(), "Argument `row` is missing");
    }

    if (torch::autograd::any_variable_requires_grad({mat})) {
      AT_ASSERTM(opt_row.has_value(), "Argument `row` is missing");
      AT_ASSERTM(opt_rowcount.has_value(), "Argument `rowcount` is missing");
      AT_ASSERTM(opt_colptr.has_value(), "Argument `colptr` is missing");
      AT_ASSERTM(opt_csr2csc.has_value(), "Argument `csr2csc` is missing");
    }

    auto row = opt_row.has_value() ? opt_row.value() : col;
    auto rowcount = opt_rowcount.has_value() ? opt_rowcount.value() : col;
    auto colptr = opt_colptr.has_value() ? opt_colptr.value() : col;
    auto csr2csc = opt_csr2csc.has_value() ? opt_csr2csc.value() : col;

    torch::optional<torch::Tensor> opt_value = torch::nullopt;
    if (has_value)
      opt_value = value;

    auto out = std::get<0>(spmm_fw(rowptr, col, opt_value, mat, "mean"));
    ctx->saved_data["has_value"] = has_value;
    ctx->save_for_backward(
        {row, rowptr, col, value, rowcount, colptr, csr2csc, mat});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto has_value = ctx->saved_data["has_value"].toBool();
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto row = saved[0], rowptr = saved[1], col = saved[2], value = saved[3],
         rowcount = saved[4], colptr = saved[5], csr2csc = saved[6],
         mat = saved[7];

    auto grad_value = Variable();
    if (has_value > 0 && torch::autograd::any_variable_requires_grad({value})) {
      grad_value = spmm_value_bw(row, rowptr, col, mat, grad_out, "mean");
    }

    auto grad_mat = Variable();
    if (torch::autograd::any_variable_requires_grad({mat})) {
      row = row.index_select(0, csr2csc);
      rowcount = rowcount.index_select(0, row).toType(mat.scalar_type());
      rowcount.masked_fill_(rowcount < 1, 1);

      if (has_value > 0)
        rowcount =
            value.view({-1, 1}).index_select(0, csr2csc).view(-1).div(rowcount);
      else
        rowcount.pow_(-1);

      grad_mat = std::get<0>(spmm_fw(colptr, row, rowcount, grad_out, "sum"));
    }

    return {Variable(), Variable(), Variable(), grad_value, Variable(),
            Variable(), Variable(), grad_mat,   Variable()};
  }
};

class SPMMMin : public torch::autograd::Function<SPMMMin> {
public:
  static variable_list forward(AutogradContext *ctx, Variable rowptr,
                               Variable col, Variable value, Variable mat,
                               bool has_value) {

    torch::optional<torch::Tensor> opt_value = torch::nullopt;
    if (has_value)
      opt_value = value;

    auto result = spmm_fw(rowptr, col, opt_value, mat, "min");
    auto out = std::get<0>(result);
    auto arg_out = std::get<1>(result).value();
    ctx->saved_data["has_value"] = has_value;
    ctx->save_for_backward({col, value, mat, arg_out});
    ctx->mark_non_differentiable({arg_out});
    return {out, arg_out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto has_value = ctx->saved_data["has_value"].toBool();
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto col = saved[0], value = saved[1], mat = saved[2], arg_out = saved[3];

    auto invalid_arg_mask = arg_out == col.size(0);
    arg_out = arg_out.masked_fill(invalid_arg_mask, 0);

    auto grad_value = Variable();
    if (has_value > 0 && torch::autograd::any_variable_requires_grad({value})) {
      auto ind = col.index_select(0, arg_out.flatten()).view_as(arg_out);
      auto out = mat.gather(-2, ind);
      out.mul_(grad_out);
      out.masked_fill_(invalid_arg_mask, 0);

      grad_value = torch::zeros_like(value);
      grad_value.scatter_add_(0, arg_out.flatten(), out.flatten());
    }

    auto grad_mat = Variable();
    if (torch::autograd::any_variable_requires_grad({mat})) {
      if (has_value > 0) {
        value = value.view({-1, 1})
                    .index_select(0, arg_out.flatten())
                    .view_as(arg_out)
                    .mul_(grad_out);
      } else
        value = grad_out;

      value.masked_fill_(invalid_arg_mask, 0);
      auto ind = col.index_select(0, arg_out.flatten()).view_as(arg_out);

      grad_mat = torch::zeros_like(mat);
      grad_mat.scatter_add_(-2, ind, value);
    }

    return {Variable(), Variable(), grad_value, grad_mat, Variable()};
  }
};

class SPMMMax : public torch::autograd::Function<SPMMMax> {
public:
  static variable_list forward(AutogradContext *ctx, Variable rowptr,
                               Variable col, Variable value, Variable mat,
                               bool has_value) {

    torch::optional<torch::Tensor> opt_value = torch::nullopt;
    if (has_value)
      opt_value = value;

    auto result = spmm_fw(rowptr, col, opt_value, mat, "max");
    auto out = std::get<0>(result);
    auto arg_out = std::get<1>(result).value();
    ctx->saved_data["has_value"] = has_value;
    ctx->save_for_backward({col, value, mat, arg_out});
    ctx->mark_non_differentiable({arg_out});
    return {out, arg_out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto has_value = ctx->saved_data["has_value"].toBool();
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto col = saved[0], value = saved[1], mat = saved[2], arg_out = saved[3];

    auto invalid_arg_mask = arg_out == col.size(0);
    arg_out = arg_out.masked_fill(invalid_arg_mask, 0);

    auto grad_value = Variable();
    if (has_value > 0 && torch::autograd::any_variable_requires_grad({value})) {
      auto ind = col.index_select(0, arg_out.flatten()).view_as(arg_out);
      auto out = mat.gather(-2, ind);
      out.mul_(grad_out);
      out.masked_fill_(invalid_arg_mask, 0);

      grad_value = torch::zeros_like(value);
      grad_value.scatter_add_(0, arg_out.flatten(), out.flatten());
    }

    auto grad_mat = Variable();
    if (torch::autograd::any_variable_requires_grad({mat})) {
      if (has_value > 0) {
        value = value.view({-1, 1})
                    .index_select(0, arg_out.flatten())
                    .view_as(arg_out)
                    .mul_(grad_out);
      } else
        value = grad_out;

      value.masked_fill_(invalid_arg_mask, 0);
      auto ind = col.index_select(0, arg_out.flatten()).view_as(arg_out);

      grad_mat = torch::zeros_like(mat);
      grad_mat.scatter_add_(-2, ind, value);
    }

    return {Variable(), Variable(), grad_value, grad_mat, Variable()};
  }
};

SPARSE_API torch::Tensor spmm_sum(torch::optional<torch::Tensor> opt_row,
                       torch::Tensor rowptr, torch::Tensor col,
                       torch::optional<torch::Tensor> opt_value,
                       torch::optional<torch::Tensor> opt_colptr,
                       torch::optional<torch::Tensor> opt_csr2csc,
                       torch::Tensor mat) {
  auto value = opt_value.has_value() ? opt_value.value() : col;
  return SPMMSum::apply(opt_row, rowptr, col, value, opt_colptr, opt_csr2csc,
                        mat, opt_value.has_value())[0];
}

SPARSE_API torch::Tensor spmm_mean(torch::optional<torch::Tensor> opt_row,
                        torch::Tensor rowptr, torch::Tensor col,
                        torch::optional<torch::Tensor> opt_value,
                        torch::optional<torch::Tensor> opt_rowcount,
                        torch::optional<torch::Tensor> opt_colptr,
                        torch::optional<torch::Tensor> opt_csr2csc,
                        torch::Tensor mat) {
  auto value = opt_value.has_value() ? opt_value.value() : col;
  return SPMMMean::apply(opt_row, rowptr, col, value, opt_rowcount, opt_colptr,
                         opt_csr2csc, mat, opt_value.has_value())[0];
}

SPARSE_API std::tuple<torch::Tensor, torch::Tensor>
spmm_min(torch::Tensor rowptr, torch::Tensor col,
         torch::optional<torch::Tensor> opt_value, torch::Tensor mat) {
  auto value = opt_value.has_value() ? opt_value.value() : col;
  auto result = SPMMMin::apply(rowptr, col, value, mat, opt_value.has_value());
  return std::make_tuple(result[0], result[1]);
}

SPARSE_API std::tuple<torch::Tensor, torch::Tensor>
spmm_max(torch::Tensor rowptr, torch::Tensor col,
         torch::optional<torch::Tensor> opt_value, torch::Tensor mat) {
  auto value = opt_value.has_value() ? opt_value.value() : col;
  auto result = SPMMMax::apply(rowptr, col, value, mat, opt_value.has_value());
  return std::make_tuple(result[0], result[1]);
}

static auto registry = torch::RegisterOperators()
                           .op("torch_sparse::spmm_sum", &spmm_sum)
                           .op("torch_sparse::spmm_mean", &spmm_mean)
                           .op("torch_sparse::spmm_min", &spmm_min)
                           .op("torch_sparse::spmm_max", &spmm_max);

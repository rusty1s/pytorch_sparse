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
  if (rowptr.device().is_cuda()) {
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
                               torch::optional<Variable> optional_row,
                               Variable rowptr, Variable col, Variable value,
                               torch::optional<Variable> optional_colptr,
                               torch::optional<Variable> optional_csr2csc,
                               Variable mat) {
    torch::Tensor row;
    if (optional_row.has_value())
      row = optional_row.value();
    torch::optional<torch::Tensor> optional_value = torch::nullopt;
    if (value.numel() > 0)
      optional_value = value;
    torch::Tensor colptr;
    if (optional_colptr.has_value())
      colptr = optional_colptr.value();
    torch::Tensor csr2csc;
    if (optional_csr2csc.has_value())
      csr2csc = optional_csr2csc.value();

    auto out = std::get<0>(spmm_fw(rowptr, col, optional_value, mat, "sum"));
    ctx->save_for_backward({row, rowptr, col, value, colptr, csr2csc, mat});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto row = saved[0];
    auto rowptr = saved[1];
    auto col = saved[2];
    auto value = saved[3];
    torch::optional<torch::Tensor> optional_value = torch::nullopt;
    if (value.numel() > 0)
      optional_value = value;
    auto colptr = saved[4];
    auto csr2csc = saved[5];
    auto mat = saved[6];

    auto grad_value = Variable();
    if (optional_value.has_value() &&
        torch::autograd::any_variable_requires_grad({value})) {
      grad_value = spmm_value_bw(row, rowptr, col, mat, grad_out, "sum");
    }

    auto grad_mat = Variable();
    if (torch::autograd::any_variable_requires_grad({mat})) {
      if (optional_value.has_value())
        optional_value = optional_value.value().index_select(0, csr2csc);
      grad_mat = torch::zeros_like(mat);
      grad_mat = std::get<0>(spmm_fw(colptr, row.index_select(0, csr2csc),
                                     optional_value, grad_out, "sum"));
    }

    return {Variable(), Variable(), Variable(), grad_value,
            Variable(), Variable(), grad_mat};
  }
};

torch::Tensor spmm_sum(torch::optional<torch::Tensor> optional_row,
                       torch::Tensor rowptr, torch::Tensor col,
                       torch::optional<torch::Tensor> optional_value,
                       torch::optional<torch::Tensor> optional_colptr,
                       torch::optional<torch::Tensor> optional_csr2csc,
                       torch::Tensor mat) {
  // Since we cannot return an *optional* gradient, we need to convert
  // `optional_value` to an empty sized tensor first :(
  auto value = torch::Tensor();
  if (optional_value.has_value())
    value = optional_value.value();
  return SPMMSum::apply(optional_row, rowptr, col, value, optional_colptr,
                        optional_csr2csc, mat)[0];
}

static auto registry =
    torch::RegisterOperators().op("torch_sparse::spmm_sum", &spmm_sum);

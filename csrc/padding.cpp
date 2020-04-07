#include <Python.h>
#include <torch/script.h>

#include "cpu/padding_cpu.h"

#ifdef WITH_CUDA
#include "cuda/padding_cuda.h"
#endif

#ifdef _WIN32
PyMODINIT_FUNC PyInit__padding(void) { return NULL; }
#endif

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<int64_t>, std::vector<int64_t>>
padded_index(torch::Tensor rowptr, torch::Tensor col, torch::Tensor rowcount,
             torch::Tensor binptr) {
  if (rowptr.device().is_cuda()) {
#ifdef WITH_CUDA
    return padded_index_cuda(rowptr, col, rowcount, binptr);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return padded_index_cpu(rowptr, col, rowcount, binptr);
  }
}

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class PaddedIndexSelect : public torch::autograd::Function<PaddedIndexSelect> {
public:
  static variable_list forward(AutogradContext *ctx, Variable src,
                               Variable index, Variable fill_value) {
    ctx->saved_data["N"] = src.size(0);

    torch::Tensor out;
    if (src.device().is_cuda()) {
#ifdef WITH_CUDA
      out = padded_index_select_cuda(src, index, fill_value);
#else
      AT_ERROR("Not compiled with CUDA support");
#endif
    } else {
      out = padded_index_select_cpu(src, index, fill_value);
    }
    ctx->save_for_backward({index});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto index = saved[0];
    auto N = ctx->saved_data["N"].toInt();
    torch::Tensor grad_in;
    if (grad_out.device().is_cuda()) {
#ifdef WITH_CUDA
      grad_in = padded_index_scatter_cuda(grad_out, index, N);
#else
      AT_ERROR("Not compiled with CUDA support");
#endif
    } else {
      grad_in = padded_index_scatter_cpu(grad_out, index, N);
    }
    return {grad_in, Variable(), Variable()};
  }
};

torch::Tensor padded_index_select(torch::Tensor src, torch::Tensor index,
                                  torch::Tensor fill_value) {
  return PaddedIndexSelect::apply(src, index, fill_value)[0];
}

static auto registry =
    torch::RegisterOperators()
        .op("torch_sparse::padded_index", &padded_index)
        .op("torch_sparse::padded_index_select", &padded_index_select);

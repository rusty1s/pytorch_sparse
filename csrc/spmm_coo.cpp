#include <torch/script.h>
#include <vector>

#include "cpu/spmm_coo_cpu.h"
#include "cuda/spmm_coo_cuda.h"

inline std::vector<int64_t> list2vec(const c10::List<int64_t> list)
{
    std::vector<int64_t> result;
    result.reserve(list.size());
    for (size_t i = 0; i < list.size(); i++)
        result.push_back(list[i]);
    return result;
}

torch::Tensor broadcast(torch::Tensor src, torch::Tensor other, int64_t dim)
{
    if (src.dim() == 1)
        for (auto i = 0; i < dim; i++)
            src = src.unsqueeze(0);
    for (auto i = src.dim(); i < other.dim(); i++)
        src = src.unsqueeze(-1);
    src = src.expand(other.sizes().vec());
    return src;
}

std::tuple<torch::Tensor, torch::Tensor> spmm_coo_fw(
    const torch::Tensor row,
    const torch::Tensor col,
    const torch::Tensor mat,
    int64_t dim_size,
    std::string reduce)
{
    if(row.device().is_cuda()){
        return spmm_coo_cuda(row, col, mat, dim_size, reduce);
    }
    else{
        return spmm_coo_cpu(row, col, mat, dim_size, reduce);
    }
        //AT_ERROR("Row Tensor not in GPU!");
}

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class SPMMMax : public torch::autograd::Function<SPMMMax>
{
public:
    static variable_list forward(AutogradContext *ctx, Variable row,
                                 Variable col,
                                 Variable mat,
                                 int64_t dim_size)
    {
        ctx->saved_data["mat_shape"] = mat.sizes();
        auto result = spmm_coo_fw(row, col, mat, dim_size, "max");
        auto out = std::get<0>(result);
        auto arg_out = std::get<1>(result);
        ctx->save_for_backward({arg_out});
        ctx->mark_non_differentiable({arg_out});
        return {out, arg_out};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outs)
    {
        auto grad_out = grad_outs[0];
        auto arg_out = ctx->get_saved_variables()[0];
        auto mat_shape = list2vec(ctx->saved_data["mat_shape"].toIntList());
        mat_shape[0] += 1;
        auto grad_in = torch::zeros(mat_shape, grad_out.options());
        grad_in.scatter_(0, arg_out, grad_out, "add");
        grad_in = grad_in.narrow(0, 0, mat_shape[0] - 1);
        return {grad_in, Variable(), Variable(), Variable()};
    }
};

class SPMMSum : public torch::autograd::Function<SPMMSum>
{
public:
    static variable_list forward(AutogradContext *ctx, Variable row,
                                 Variable col,
                                 Variable mat,
                                 int64_t dim_size)
    {
        ctx->saved_data["mat_shape"] = mat.sizes();
        auto result = spmm_coo_fw(row, col, mat, dim_size, "sum");
        auto out = std::get<0>(result);
        ctx->save_for_backward({row, col});
        ctx->mark_non_differentiable({row, col});
        return {out};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outs)
    {
        auto grad_out = grad_outs[0];
        auto row = ctx->get_saved_variables()[0];
        auto col = ctx->get_saved_variables()[1];
        auto mat_shape = list2vec(ctx->saved_data["mat_shape"].toIntList());
        auto result = spmm_coo_fw(col, row, grad_out, mat_shape[0], "sum");
        auto grad_in = std::get<0>(result);
        return {grad_in, Variable(), Variable(), Variable()};
    }
};

class SPMMMean : public torch::autograd::Function<SPMMMean>
{
public:
    static variable_list forward(AutogradContext *ctx, Variable row,
                                 Variable col,
                                 Variable mat,
                                 int64_t dim_size)
    {
        ctx->saved_data["mat_shape"] = mat.sizes();
        auto result = spmm_coo_fw(row, col, mat, dim_size, "sum");
        auto out = std::get<0>(result);
        // compute degree of elements in result tensor
        auto ones = torch::ones(size(mat,0), mat.options());
        result = spmm_coo_fw(row, col, ones, dim_size, "sum");
        auto degree = std::get<0>(result);
        degree.masked_fill_(degree < 1, 1);
        // divide result tensor by degree
        degree = broadcast(degree, out, 0);
        if (out.is_floating_point())
            out.true_divide_(degree);
        else
            out.div_(degree, "floor");
        ctx->save_for_backward({row, col, degree});
        ctx->mark_non_differentiable({row, col, degree});
        return {out};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outs)
    {
        auto grad_out = grad_outs[0].clone();
        auto saved = ctx->get_saved_variables();
        auto row = saved[0];
        auto col = saved[1];
        auto degree = saved[2];
        auto mat_shape = list2vec(ctx->saved_data["mat_shape"].toIntList());
        grad_out.true_divide_(degree);
        auto result = spmm_coo_fw(col, row, grad_out, mat_shape[0], "sum");
        auto grad_in = std::get<0>(result);
        return {grad_in, Variable(), Variable(), Variable(), Variable()};
    }
};

std::tuple<torch::Tensor, torch::Tensor> spmm_coo_max(const torch::Tensor row,
                                                      const torch::Tensor col,
                                                      const torch::Tensor mat,
                                                      int64_t dim_size)
{
    auto result = SPMMMax::apply(row, col, mat, dim_size);

    return std::make_tuple(result[0], result[1]);
}

torch::Tensor spmm_coo_sum(const torch::Tensor row,
                           const torch::Tensor col,
                           const torch::Tensor mat,
                           int64_t dim_size)
{

    return SPMMSum::apply(row, col, mat, dim_size)[0];
}

torch::Tensor spmm_coo_mean(const torch::Tensor row,
                            const torch::Tensor col,
                            const torch::Tensor mat,
                            int64_t dim_size)
{

    return SPMMMean::apply(row, col, mat, dim_size)[0];
}

static auto registry = torch::RegisterOperators()
                           .op("torch_sparse::spmm_coo_sum", &spmm_coo_sum)
                           .op("torch_sparse::spmm_coo_mean", &spmm_coo_mean)
                           .op("torch_sparse::spmm_coo_max", &spmm_coo_max);

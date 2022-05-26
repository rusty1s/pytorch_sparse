#include "spmm_coo_cpu.h"
#include "utils.h"
#include "reducer.h"

std::tuple<torch::Tensor, torch::Tensor> spmm_coo_cpu(
    const torch::Tensor row,
    const torch::Tensor col,
    torch::Tensor mat,
    int64_t dim_size,
    std::string reduce)
{
    // check input
    CHECK_CPU(row);
    CHECK_CPU(col);
    CHECK_CPU(mat);
    CHECK_INPUT(col.dim() == 1);
    CHECK_INPUT(row.dim() == 1);
    CHECK_INPUT(row.size(0) == col.size(0));
    mat = mat.contiguous();

    // variables for loops
    size_t hidden_dim = 1;
    if (mat.dim() == 2)
        hidden_dim = size(mat, 1);
    size_t edge_count = row.numel();    

    // create out and arg_out Tensor with given out_dim
    auto res_dims = mat.sizes().vec();
    res_dims[0] = dim_size;
    torch::Tensor res = torch::empty(res_dims, mat.options());
    torch::Tensor arg_out = torch::empty(0, row.options());
    int64_t *arg_out_data = nullptr;
    if (reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX)
    {
        arg_out = torch::full_like(res, mat.size(0), row.options());
        arg_out_data = arg_out.data_ptr<int64_t>();
    }

    auto row_data = row.data_ptr<int64_t>();
    auto col_data = col.data_ptr<int64_t>();

    // sparse matrix multiplication
    AT_DISPATCH_ALL_TYPES(mat.scalar_type(), "_", [&]{
        auto mat_data = mat.data_ptr<scalar_t>();
        auto res_data = res.data_ptr<scalar_t>();
        
        AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
            res.fill_(Reducer<scalar_t, REDUCE>::init());
            for (size_t e = 0; e < edge_count; e++)
            {
                for (size_t h = 0; h < hidden_dim; h++)
                {
                   Reducer<scalar_t, REDUCE>::update(
                    res_data + col_data[e]*hidden_dim + h, 
                    mat_data[row_data[e]*hidden_dim + h],
                    arg_out_data + col_data[e]*hidden_dim + h, row_data[e]); 
                }
            }
    
            res.masked_fill_(res == Reducer<scalar_t, REDUCE>::init(), (scalar_t)0);
            
        }); 
    });

    return std::make_tuple(res, arg_out);
}
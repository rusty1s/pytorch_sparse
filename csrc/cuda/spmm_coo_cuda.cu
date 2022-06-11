#include "spmm_coo_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "reducer.cuh"
#include "utils.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

template <typename scalar_t, ReductionType REDUCE, bool HAS_VALUE>
__global__ void spmm_coo_kernel(
    const scalar_t* __restrict__ mat,
    const int64_t* __restrict__ row, 
    const int64_t* __restrict__ col,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ res,
    int64_t* __restrict__ arg_out,
    size_t hidden_dim,
    size_t N)
{
    
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(thread_id < N){
        int edge_index = thread_id / hidden_dim;
        int hidden_dim_index = thread_id % hidden_dim;

        int edge_start = __ldg(row + edge_index);
        int edge_end = __ldg(col + edge_index);
        scalar_t write_val = __ldg(mat + edge_start*hidden_dim + hidden_dim_index);
        int res_index = edge_end*hidden_dim + hidden_dim_index;
        
        if(HAS_VALUE)
            write_val *= __ldg(value + edge_index);
          
        Reducer<scalar_t, REDUCE>::atomic_write(
                res + res_index, 
                write_val);
        
        //compute arg out tensor
        if(REDUCE == MIN || REDUCE == MAX){
            __syncthreads();
            if(HAS_VALUE)
               edge_start = edge_index;

            if(res[res_index] == write_val)
                arg_out[res_index] = edge_start;
        }
    }
}

std::tuple<torch::Tensor,torch::optional<torch::Tensor>> spmm_coo_cuda(
    const torch::Tensor row, 
    const torch::Tensor col,
    const torch::optional<torch::Tensor> optional_value, 
    torch::Tensor mat,
    int64_t dim_size,
    std::string reduce)
{
    //check input
    CHECK_CUDA(row);
    CHECK_CUDA(col);
    CHECK_CUDA(mat);
    CHECK_INPUT(row.size(0) == col.size(0)); 
    if(optional_value.has_value()){
        CHECK_CUDA(optional_value.value());
        CHECK_INPUT(optional_value.value().dim() == 1);
        CHECK_INPUT(optional_value.value().size(0) == col.size(0));
    }
    mat = mat.contiguous();
    
    // variables for loops
    size_t hidden_dim = 1;
    if(mat.dim() == 2)
        hidden_dim = size(mat, 1);
    size_t N = row.numel()*hidden_dim;

    //create out and arg_out Tensor
    auto res_dims = mat.sizes().vec();
    res_dims[0] = dim_size;
    torch::Tensor res = torch::empty(res_dims, mat.options());
    torch::optional<torch::Tensor> arg_out = torch::nullopt;
    int64_t *arg_out_data = nullptr;
    if(reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX){
        arg_out = torch::full_like(res,col.size(0),row.options());
        arg_out_data = arg_out.value().data_ptr<int64_t>();
    }

    auto row_data = row.data_ptr<int64_t>();
    auto col_data = col.data_ptr<int64_t>();
   
    // sparse matrix multiplication
    AT_DISPATCH_ALL_TYPES(mat.scalar_type(), "_", [&] {
        auto mat_data = mat.data_ptr<scalar_t>();
        auto res_data = res.data_ptr<scalar_t>();
        
        AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
            res.fill_(Reducer<scalar_t, REDUCE>::init());

            if(optional_value.has_value()){
                auto value_data = optional_value.value().data_ptr<scalar_t>();
                spmm_coo_kernel<scalar_t, REDUCE, true><<<BLOCKS(N), THREADS>>>(
                    mat_data,
                    row_data,
                    col_data,
                    value_data,
                    res_data,
                    arg_out_data,
                    hidden_dim,
                    N);     
            }else{
                spmm_coo_kernel<scalar_t, REDUCE, false><<<BLOCKS(N), THREADS>>>(
                    mat_data,
                    row_data,
                    col_data,
                    nullptr,
                    res_data,
                    arg_out_data,
                    hidden_dim,
                    N);     
            }

            res.masked_fill_(res == Reducer<scalar_t, REDUCE>::init(), (scalar_t)0);
        });     
    });

    checkCuda(cudaGetLastError());
    
    return std::make_tuple(res,arg_out);   
}
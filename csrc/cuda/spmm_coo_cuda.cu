#include "spmm_coo_cuda.h"

//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <stdio.h>
//#include <assert.h>
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

template <typename scalar_t, ReductionType REDUCE>
__global__ void spmm_coo_kernel(
    const scalar_t* __restrict__ mat,
    const int64_t* __restrict__ row, 
    const int64_t* __restrict__ col,
    scalar_t* __restrict__ res,
    size_t hidden_dim,
    size_t N)
{
    
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(thread_id < N){
        int edge_index = thread_id / hidden_dim;
        int hidden_dim_index = thread_id % hidden_dim;      

        Reducer<scalar_t, REDUCE>::atomic_write(
           res + col[edge_index]*hidden_dim + hidden_dim_index, 
            mat[ row[edge_index]*hidden_dim + hidden_dim_index]);  
    }
}

template <typename scalar_t>
__global__ void spmm_coo_arg_kernel(
    const scalar_t* __restrict__ mat,
    const int64_t* __restrict__ row, 
    const int64_t* __restrict__ col,
    scalar_t* __restrict__ res,
    int64_t* __restrict__ arg_out,
    size_t hidden_dim,
    size_t N)
{
    
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_id < N){
        int edge_index = thread_id / hidden_dim;
        int hidden_dim_index = thread_id % hidden_dim;

        if(res[col[edge_index]*hidden_dim + hidden_dim_index] == mat[row[edge_index]*hidden_dim + hidden_dim_index]){
            arg_out[col[edge_index]*hidden_dim + hidden_dim_index] = row[edge_index];
        }
    }
}

std::tuple<torch::Tensor,torch::Tensor> spmm_coo_cuda(
    const torch::Tensor row, 
    const torch::Tensor col, 
    torch::Tensor mat,
    int64_t dim_size,
    std::string reduce)
{
    //check input
    CHECK_CUDA(row);
    CHECK_CUDA(col);
    CHECK_CUDA(mat);
    CHECK_INPUT(row.size(0) == col.size(0));
    mat = mat.contiguous();
    
    // variables for loops
    size_t hidden_dim = 1;
    if(mat.dim() == 2)
        hidden_dim = size(mat, 1);
    size_t N = row.numel()*hidden_dim;

    //create out and arg_out Tensor with given out_dim
    auto res_dims = mat.sizes().vec();
    res_dims[0] = dim_size;
    torch::Tensor res = torch::empty(res_dims, mat.options());
    torch::Tensor arg_out = torch::empty(0, row.options());
    int64_t *arg_out_data = nullptr;
    if (reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX) {
        arg_out = torch::full_like(res,mat.size(0),row.options());
        arg_out_data = arg_out.data_ptr<int64_t>();
      }

    auto row_data = row.data_ptr<int64_t>();
    auto col_data = col.data_ptr<int64_t>();
   
    // sparse matrix multiplication
    AT_DISPATCH_ALL_TYPES(mat.scalar_type(), "_", [&] {
        auto mat_data = mat.data_ptr<scalar_t>();
        auto res_data = res.data_ptr<scalar_t>();
        
        AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
            res.fill_(Reducer<scalar_t, REDUCE>::init());

            spmm_coo_kernel<scalar_t, REDUCE><<<BLOCKS(N), THREADS>>>(
                mat_data,
                row_data,
                col_data,
                res_data,
                hidden_dim,
                N);
    
            res.masked_fill_(res == Reducer<scalar_t, REDUCE>::init(), (scalar_t)0);
            if (REDUCE == MIN || REDUCE == MAX){
                spmm_coo_arg_kernel<scalar_t><<<BLOCKS(N), THREADS>>>(
                    mat_data,
                    row_data,
                    col_data,
                    res_data,
                    arg_out_data,
                    hidden_dim,
                    N);
            }      
        });     
    });

    checkCuda(cudaGetLastError());
    
    return std::make_tuple(res,arg_out);   
}
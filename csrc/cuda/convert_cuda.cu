#include "convert_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"

#define THREADS 256

__global__ void ind2ptr_kernel(const int64_t *ind_data, int64_t *out_data,
                               int64_t M, int64_t numel) {

  int64_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (thread_idx == 0) {
    for (int64_t i = 0; i <= ind_data[0]; i++)
      out_data[i] = 0;
  } else if (thread_idx < numel) {
    for (int64_t i = ind_data[thread_idx - 1]; i < ind_data[thread_idx]; i++)
      out_data[i + 1] = thread_idx;
  } else if (thread_idx == numel) {
    for (int64_t i = ind_data[numel - 1] + 1; i < M + 1; i++)
      out_data[i] = numel;
  }
}

torch::Tensor ind2ptr_cuda(torch::Tensor ind, int64_t M) {
  CHECK_CUDA(ind);
  cudaSetDevice(ind.get_device());

  auto out = torch::empty(M + 1, ind.options());

  if (ind.numel() == 0)
    return out.zero_();

  auto ind_data = ind.data_ptr<int64_t>();
  auto out_data = out.data_ptr<int64_t>();
  auto stream = at::cuda::getCurrentCUDAStream();
  ind2ptr_kernel<<<(ind.numel() + 2 + THREADS - 1) / THREADS, THREADS, 0,
                   stream>>>(ind_data, out_data, M, ind.numel());
  return out;
}

__global__ void ptr2ind_kernel(const int64_t *ptr_data, int64_t *out_data,
                               int64_t E, int64_t numel) {

  int64_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (thread_idx < numel) {
    int64_t idx = ptr_data[thread_idx], next_idx = ptr_data[thread_idx + 1];
    for (int64_t i = idx; i < next_idx; i++) {
      out_data[i] = thread_idx;
    }
  }
}

torch::Tensor ptr2ind_cuda(torch::Tensor ptr, int64_t E) {
  CHECK_CUDA(ptr);
  cudaSetDevice(ptr.get_device());

  auto out = torch::empty(E, ptr.options());
  auto ptr_data = ptr.data_ptr<int64_t>();
  auto out_data = out.data_ptr<int64_t>();
  auto stream = at::cuda::getCurrentCUDAStream();
  ptr2ind_kernel<<<(ptr.numel() - 1 + THREADS - 1) / THREADS, THREADS, 0,
                   stream>>>(ptr_data, out_data, E, ptr.numel() - 1);
  return out;
}

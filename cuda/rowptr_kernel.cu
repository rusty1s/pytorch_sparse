#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "compat.cuh"

#define THREADS 1024

__global__ void rowptr_kernel(const int64_t *row_data, int64_t *out_data,
                              int64_t M, int64_t numel) {

  int64_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (thread_idx == 0) {
    for (int64_t i = 0; i <= row_data[0]; i++)
      out_data[i] = 0;
  } else if (thread_idx < numel) {
    for (int64_t i = row_data[thread_idx - 1]; i < row_data[thread_idx]; i++)
      out_data[i + 1] = thread_idx;
  } else if (thread_idx == numel) {
    for (int64_t i = row_data[numel - 1] + 1; i < M + 1; i++)
      out_data[i] = numel;
  }
}

at::Tensor rowptr_cuda(at::Tensor row, int64_t M) {
  AT_ASSERTM(row.dim() == 1, "Row needs to be one-dimensional");

  auto out = at::empty(M + 1, row.options());
  auto row_data = row.DATA_PTR<int64_t>();
  auto out_data = out.DATA_PTR<int64_t>();

  auto stream = at::cuda::getCurrentCUDAStream();
  rowptr_kernel<<<(row.numel() + 2 + THREADS - 1) / THREADS, THREADS, 0,
                  stream>>>(row_data, out_data, M, row.numel());

  return out;
}

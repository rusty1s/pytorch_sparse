#include "diag_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"

#define THREADS 1024

__global__ void non_diag_mask_kernel(const int64_t *row_data,
                                     const int64_t *col_data, bool *out_data,
                                     int64_t N, int64_t k, int64_t num_diag,
                                     int64_t numel) {

  int64_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (thread_idx < numel) {
    int64_t r = row_data[thread_idx], c = col_data[thread_idx];

    if (k < 0) {
      if (r + k < 0) {
        out_data[thread_idx] = true;
      } else if (r + k >= N) {
        out_data[thread_idx + num_diag] = true;
      } else if (r + k > c) {
        out_data[thread_idx + r + k] = true;
      } else if (r + k < c) {
        out_data[thread_idx + r + k + 1] = true;
      }

    } else {
      if (r + k >= N) {
        out_data[thread_idx + num_diag] = true;
      } else if (r + k > c) {
        out_data[thread_idx + r] = true;
      } else if (r + k < c) {
        out_data[thread_idx + r + 1] = true;
      }
    }
  }
}

torch::Tensor non_diag_mask_cuda(torch::Tensor row, torch::Tensor col,
                                 int64_t M, int64_t N, int64_t k) {
  CHECK_CUDA(row);
  CHECK_CUDA(col);
  cudaSetDevice(row.get_device());

  auto E = row.size(0);
  auto num_diag = k < 0 ? std::min(M + k, N) : std::min(M, N - k);

  auto row_data = row.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();

  auto mask = torch::zeros(E + num_diag, row.options().dtype(torch::kBool));
  auto mask_data = mask.data_ptr<bool>();

  if (E == 0)
    return mask;

  auto stream = at::cuda::getCurrentCUDAStream();
  non_diag_mask_kernel<<<(E + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
      row_data, col_data, mask_data, N, k, num_diag, E);

  return mask;
}

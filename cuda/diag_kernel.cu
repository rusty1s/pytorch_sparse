#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "compat.cuh"

#define THREADS 1024

__global__ void non_diag_mask_kernel(const int64_t *index_data, bool *out_data,
                                     int64_t N, int64_t k, int64_t num_diag,
                                     int64_t numel) {

  int64_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (thread_idx < numel) {
    int64_t r = index_data[thread_idx], c = index_data[thread_idx + numel];

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

at::Tensor non_diag_mask_cuda(at::Tensor index, int64_t M, int64_t N,
                              int64_t k) {
  int64_t E = index.size(1);

  index = index.contiguous();
  auto index_data = index.DATA_PTR<int64_t>();

  int64_t num_diag = k < 0 ? std::min(M + k, N) : std::min(M, N - k);

  auto mask = at::zeros(E + num_diag, index.options().dtype(at::kBool));
  auto mask_data = mask.DATA_PTR<bool>();

  auto stream = at::cuda::getCurrentCUDAStream();
  non_diag_mask_kernel<<<(E + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
      index_data, mask_data, N, k, num_diag, E);

  return mask;
}

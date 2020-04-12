#include "rw_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

__global__ void uniform_random_walk_kernel(const int64_t *rowptr,
                                           const int64_t *col,
                                           const int64_t *start,
                                           const float *rand, int64_t *out,
                                           int64_t walk_length, int64_t numel) {
  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_idx < numel) {
    int64_t cur = start[thread_idx];
    out[thread_idx] = cur;

    int64_t row_start, row_end;
    for (int64_t l = 0; l < walk_length; l++) {
      row_start = rowptr[cur], row_end = rowptr[cur + 1];
      cur = col[row_start +
                int64_t(rand[l * numel + thread_idx] * (row_end - row_start))];
      out[(l + 1) * numel + thread_idx] = cur;
    }
  }
}

torch::Tensor random_walk_cuda(torch::Tensor rowptr, torch::Tensor col,
                               torch::Tensor start, int64_t walk_length) {
  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  CHECK_CUDA(start);
  cudaSetDevice(rowptr.get_device());

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  CHECK_INPUT(start.dim() == 1);

  auto rand = torch::rand({walk_length, start.size(0)},
                          start.options().dtype(torch::kFloat));
  auto out = torch::full({walk_length + 1, start.size(0)}, -1, start.options());

  auto stream = at::cuda::getCurrentCUDAStream();
  uniform_random_walk_kernel<<<BLOCKS(start.numel()), THREADS, 0, stream>>>(
      rowptr.data_ptr<int64_t>(), col.data_ptr<int64_t>(),
      start.data_ptr<int64_t>(), rand.data_ptr<float>(),
      out.data_ptr<int64_t>(), walk_length, start.numel());

  return out.t().contiguous();
}

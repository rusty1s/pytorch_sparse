#include <ATen/ATen.h>

#include "compat.cuh"

#define THREADS 32 * 16

// Paper: Design Principles for Sparse Matrix Multiplication on the GPU
// Code:  https://github.com/owensgroup/merge-spmm
template <typename scalar_t, size_t Y_SIZE>
__global__ void
spmm_row_kernel(const int64_t *rowptr_data, const int64_t *col_data,
                const scalar_t *val_data, const scalar_t *mat_data,
                scalar_t *out_data, size_t N, size_t M, size_t K) {

  // We ignore blockIdx.y here, because threads across blockIdx.y operate on the
  // same row.
  int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

  int warp_idx = thread_idx >> 5;       // thread_id / 32
  int lane_idx = thread_idx & (32 - 1); // thread_id % 32
  int row = warp_idx;                   // Each warp processes exactly one row.

  // Compute the column index of `mat` in which the thread is operating.
  int mat_col_idx = lane_idx + (blockIdx.y << 5);

  // Compute the output index given in row-major order.
  int out_idx = row * K + lane_idx + (blockIdx.y << 5);

  // Helper arrays for warp communication.
  int mat_row_all[Y_SIZE];
  scalar_t val_all[Y_SIZE];

  int leftover = K - (blockIdx.y << 5);

  if (row < N) {
    int row_start = __ldg(rowptr_data + row);
    int row_end = __ldg(rowptr_data + row + 1);

    // Iterate over all col indices in parallel.
    for (int col_idx = row_start + lane_idx; col_idx < row_end; col_idx += 32) {
      int mat_row = __ldg(col_data + col_idx) * K;
      int val = __ldg(val_data + col_idx);
      scalar_t sum = (scalar_t)0;

      for (int i = 0; i < 32; i += Y_SIZE) {
#pragma unroll
        for (int j = 0; j < Y_SIZE; j++) {
          // Warp communication with *all* threads (mask = 0xffffffff).
          // TODO: Compute real bit mask via `__ballot_sync()`.
          mat_row_all[j] = __shfl_sync(0xffffffff, mat_row, i + j);
          val_all[j] = __shfl_sync(0xffffffff, val, i + j);
        }
#pragma unroll
        for (int j = 0; j < Y_SIZE; j++) {
          if (lane_idx < leftover) {
            // Coalesced memory access into `mat`.
            sum += val_all[j] * __ldg(mat_data + mat_row_all[j] + mat_col_idx);
          }
        }
      }
      if (lane_idx < leftover) {
        out_data[out_idx] = sum;
      }
    }
  }
}

at::Tensor spmm_cuda(at::Tensor rowptr, at::Tensor col, at::Tensor val,
                     at::Tensor mat) {
  // TODO: Set device

  auto N = rowptr.numel() - 1;
  auto M = mat.size(0);
  auto K = mat.size(1);

  auto out = at::empty({N, K}, mat.options());

  auto rowptr_data = rowptr.DATA_PTR<int64_t>();
  auto col_data = col.DATA_PTR<int64_t>();
  auto val_data = val.DATA_PTR<float>();
  auto mat_data = mat.DATA_PTR<float>();
  auto out_data = out.DATA_PTR<float>();

  auto block_dim = dim3(THREADS);
  auto grid_dim = dim3((N + THREADS - 1) / THREADS, (K + 32 - 1) / 32);

  spmm_row_kernel<float, 32><<<grid_dim, block_dim, 0 /*, cuda_stream */>>>(
      rowptr_data, col_data, val_data, mat_data, out_data, N, M, K);

  return out;
}

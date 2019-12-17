#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "compat.cuh"

#define Y_SIZE 32
#define THREADS 256

// Paper: Design Principles for Sparse Matrix Multiplication on the GPU
// Code:  https://github.com/owensgroup/merge-spmm
template <typename scalar_t>
__global__ void
spmm_row_kernel(const int64_t *rowptr_data, const int64_t *col_data,
                const scalar_t *val_data, const scalar_t *mat_data,
                scalar_t *out_data, size_t N, size_t K) {

  // We ignore blockIdx.y here, because threads across blockIdx.y operate on the
  // same row.
  int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

  int warp_idx = thread_idx >> 5;       // thread_id / 32
  int lane_idx = thread_idx & (32 - 1); // thread_id % 32
  int row = warp_idx;                   // Each warp processes exactly one row.

  // Compute the column index of `mat` in which the thread is operating.
  int mat_col_idx = lane_idx + (blockIdx.y << 5);

  // Compute the output index (row-major order).
  int out_idx = row * K + lane_idx + (blockIdx.y << 5);

  // Helper arrays for warp communication.
  int mat_row_all[Y_SIZE];
  scalar_t val_all[Y_SIZE];

  int leftover = K - (blockIdx.y << 5);

  if (row < N) {
    int row_start = __ldg(rowptr_data + row);
    int row_end = __ldg(rowptr_data + row + 1);
    int col_idx = row_start + lane_idx;

    int mat_row = -1;
    scalar_t val = (scalar_t)0;
    scalar_t sum = (scalar_t)0;

    // Iterate over all col indices in parallel with 32 threads.
    for (int c = row_start; c < row_end; c += 32) {

      if (col_idx < row_end) {
        // Coalesced memory access into `col` and `val`.
        mat_row = __ldg(col_data + col_idx) * K;
        val = __ldg(val_data + col_idx);
      } else {
        mat_row = 0;
        val = (scalar_t)0;
      }
      col_idx += 32;

#pragma unroll
      for (int i = 0; i < 32; i += Y_SIZE) {
#pragma unroll
        for (int j = 0; j < Y_SIZE; j++) {
          // Communication between *all* threads in a warp.
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
    }
    if (lane_idx < leftover) {
      // Coalesced memory access into `out`.
      out_data[out_idx] = sum;
    }
  }
}

at::Tensor spmm_cuda(at::Tensor rowptr, at::Tensor col, at::Tensor val,
                     at::Tensor mat) {
  auto N = rowptr.numel() - 1;
  auto K = mat.size(1);
  auto out = at::empty({N, K}, mat.options());

  auto rowptr_data = rowptr.DATA_PTR<int64_t>();
  auto col_data = col.DATA_PTR<int64_t>();

  auto block_dim = dim3(THREADS);
  auto grid_dim = dim3((32 * N + THREADS - 1) / THREADS, (K + 31) / 32);

  AT_DISPATCH_ALL_TYPES(mat.scalar_type(), "spmm_kernel", [&] {
    auto val_data = val.DATA_PTR<scalar_t>();
    auto mat_data = mat.DATA_PTR<scalar_t>();
    auto out_data = out.DATA_PTR<scalar_t>();

    spmm_row_kernel<scalar_t>
        <<<grid_dim, block_dim, 0, at::cuda::getCurrentCUDAStream()>>>(
            rowptr_data, col_data, val_data, mat_data, out_data, N, K);
  });

  return out;
}

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "compat.cuh"

#define THREADS 256

#define ADD 0
#define MEAN 1
#define MIN 2
#define MAX 3

// Paper: Design Principles for Sparse Matrix Multiplication on the GPU
// Code:  https://github.com/owensgroup/merge-spmm
template <typename scalar_t, int64_t REDUCE, bool HAS_VAL>
__global__ void spmm_kernel(const int64_t *rowptr_data, const int64_t *col_data,
                            const scalar_t *val_data, const scalar_t *mat_data,
                            scalar_t *out_data, int64_t *arg_out_data, size_t N,
                            size_t K) {

  // We ignore blockIdx.y here, because threads
  // across `blockIdx.y` are treated equally.
  int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

  int row = thread_idx >> 5;            // thread_id / 32
  int lane_idx = thread_idx & (32 - 1); // thread_id % 32

  // Compute the column index of `mat` in which the thread is operating.
  int mat_col_idx = lane_idx + (blockIdx.y << 5);

  // Compute the output index (row-major order).
  int out_idx = row * K + lane_idx + (blockIdx.y << 5);

  // Helper arrays for warp communication.
  int mat_rows[32];
  scalar_t vals[32];

  // Do not aggregate/write across the Y-axis (lane_idx < leftover).
  int leftover = K - (blockIdx.y << 5);

  if (row < N) {
    int row_start = __ldg(rowptr_data + row);
    int row_end = __ldg(rowptr_data + row + 1);
    int col_idx = row_start + lane_idx;

    int mat_row;
    scalar_t val, result;
    int64_t arg_result = -1;

    // Dependent on `reduce`, we need to initialize `result` accordingly.
    if (REDUCE == ADD)
      result = (scalar_t)0;
    else if (REDUCE == MEAN)
      result = (scalar_t)0;
    else if (REDUCE == MIN)
      result = std::numeric_limits<scalar_t>::max();
    else if (REDUCE == MAX)
      result = std::numeric_limits<scalar_t>::min();

    // Iterate over all col indices in parallel within a warp.
    for (int c = row_start; c < row_end; c += 32) {

      if (col_idx < row_end) {
        // Coalesced memory access into `col` and `val`.
        mat_row = __ldg(col_data + col_idx) * K;
        val = HAS_VAL ? __ldg(val_data + col_idx) : (scalar_t)1;
      } else {
        mat_row = 0;
        val = (scalar_t)0;
      }
      col_idx += 32;

#pragma unroll
      for (int i = 0; i < 32; i++) {
        // Communication between all threads in a warp.
        mat_rows[i] = __shfl_sync(0xffffffff, mat_row, i);
        vals[i] = __shfl_sync(0xffffffff, val, i);
      }

#pragma unroll
      for (int i = 0; i < 32; i++) {
        if (lane_idx < leftover && vals[i] != 0) {
          // Coalesced memory access into `mat`.
          val = vals[i] * __ldg(mat_data + mat_rows[i] + mat_col_idx);

          // Aggregate results along row.
          if (REDUCE == ADD)
            result += val;
          else if (REDUCE == MEAN)
            result += val;
          else if (REDUCE == MIN) {
            if (val < result) {
              result = val;
              arg_result = row_start + i;
            }
          } else if (REDUCE == MAX) {
            if (val > result) {
              result = val;
              arg_result = row_start + i;
            }
          }
        }
      }
    }

    if (lane_idx < leftover) {
      // Coalesced write into `out` (dependent on `reduce`).
      if (REDUCE == ADD)
        out_data[out_idx] = result;
      else if (REDUCE == MEAN)
        out_data[out_idx] = result / scalar_t(row_end - row_start);
      else if (REDUCE == MIN) {
        arg_out_data[out_idx] = arg_result;
        if (result == std::numeric_limits<scalar_t>::max())
          out_data[out_idx] = (scalar_t)0;
        else
          out_data[out_idx] = result;
      } else if (REDUCE == MAX) {
        arg_out_data[out_idx] = arg_result;
        if (result == std::numeric_limits<scalar_t>::min())
          out_data[out_idx] = (scalar_t)0;
        else
          out_data[out_idx] = result;
      }
    }
  }
}

at::Tensor spmm_cuda(at::Tensor rowptr, at::Tensor col,
                     at::optional<at::Tensor> val, at::Tensor mat,
                     std::string reduce) {

  auto N = rowptr.size(0) - 1;
  auto K = mat.size(1);
  auto out = at::empty({N, K}, mat.options());

  auto rowptr_data = rowptr.DATA_PTR<int64_t>();
  auto col_data = col.DATA_PTR<int64_t>();

  auto block = dim3(THREADS);
  auto grid = dim3((32 * N + THREADS - 1) / THREADS, (K + 31) / 32);

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES(mat.scalar_type(), "spmm_kernel", [&] {
    auto mat_data = mat.DATA_PTR<scalar_t>();
    auto out_data = out.DATA_PTR<scalar_t>();

    if (val.has_value()) {
      auto val_data = val.value().DATA_PTR<scalar_t>();
      if (reduce == "add")
        spmm_kernel<scalar_t, ADD, true><<<grid, block, 0, stream>>>(
            rowptr_data, col_data, val_data, mat_data, out_data, nullptr, N, K);
      else if (reduce == "mean")
        spmm_kernel<scalar_t, MEAN, true><<<grid, block, 0, stream>>>(
            rowptr_data, col_data, val_data, mat_data, out_data, nullptr, N, K);
    } else {
      if (reduce == "add")
        spmm_kernel<scalar_t, ADD, false><<<grid, block, 0, stream>>>(
            rowptr_data, col_data, nullptr, mat_data, out_data, nullptr, N, K);
      else if (reduce == "mean")
        spmm_kernel<scalar_t, MEAN, false><<<grid, block, 0, stream>>>(
            rowptr_data, col_data, nullptr, mat_data, out_data, nullptr, N, K);
    }
  });

  return out;
}

std::tuple<at::Tensor, at::Tensor>
spmm_arg_cuda(at::Tensor rowptr, at::Tensor col, at::optional<at::Tensor> val,
              at::Tensor mat, std::string reduce) {

  auto N = rowptr.size(0) - 1;
  auto K = mat.size(1);
  auto out = at::empty({N, K}, mat.options());
  auto arg_out = at::empty({N, K}, rowptr.options());

  auto rowptr_data = rowptr.DATA_PTR<int64_t>();
  auto col_data = col.DATA_PTR<int64_t>();
  auto arg_out_data = arg_out.DATA_PTR<int64_t>();

  auto block = dim3(THREADS);
  auto grid = dim3((32 * N + THREADS - 1) / THREADS, (K + 31) / 32);

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES(mat.scalar_type(), "spmm_kernel", [&] {
    auto mat_data = mat.DATA_PTR<scalar_t>();
    auto out_data = out.DATA_PTR<scalar_t>();

    if (val.has_value()) {
      auto val_data = val.value().DATA_PTR<scalar_t>();
      if (reduce == "min")
        spmm_kernel<scalar_t, MIN, true><<<grid, block, 0, stream>>>(
            rowptr_data, col_data, val_data, mat_data, out_data, arg_out_data,
            N, K);
      else if (reduce == "max")
        spmm_kernel<scalar_t, MAX, true><<<grid, block, 0, stream>>>(
            rowptr_data, col_data, val_data, mat_data, out_data, arg_out_data,
            N, K);
    } else {
      if (reduce == "min")
        spmm_kernel<scalar_t, MIN, false><<<grid, block, 0, stream>>>(
            rowptr_data, col_data, nullptr, mat_data, out_data, arg_out_data, N,
            K);
      else if (reduce == "max")
        spmm_kernel<scalar_t, MAX, false><<<grid, block, 0, stream>>>(
            rowptr_data, col_data, nullptr, mat_data, out_data, arg_out_data, N,
            K);
    }
  });

  return std::make_tuple(out, arg_out);
}

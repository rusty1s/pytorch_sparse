#include "spmm_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "reducer.cuh"
#include "utils.cuh"

#define THREADS 256
#define FULL_MASK 0xffffffff

// Paper: Design Principles for Sparse Matrix Multiplication on the GPU
// Code:  https://github.com/owensgroup/merge-spmm
template <typename scalar_t, ReductionType REDUCE, bool HAS_VALUE>
__global__ void spmm_kernel(const int64_t *rowptr_data, const int64_t *col_data,
                            const scalar_t *value_data,
                            const scalar_t *mat_data, scalar_t *out_data,
                            int64_t *arg_out_data, int B, int M, int N, int K) {

  // We ignore blockIdx.y here, because threads
  // across `blockIdx.y` are treated equally.
  int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

  int row = thread_idx >> 5;            // thread_idx / 32
  int lane_idx = thread_idx & (32 - 1); // thread_idx % 32
  int batch_idx = row / M;

  // Compute the column index of `mat` in which the thread is operating.
  int mat_col_idx = lane_idx + (blockIdx.y << 5);

  // Compute the output index (row-major order).
  int out_idx = row * K + mat_col_idx;

  // Helper arrays for warp communication.
  int mat_row, mat_rows[32];
  scalar_t val, vals[HAS_VALUE ? 32 : 1];

  // Do not aggregate/write across the Y-axis (lane_idx < leftover).
  int leftover = K - (blockIdx.y << 5);

  if (batch_idx < B) {
    int row_start = __ldg(rowptr_data + (row % M));
    int row_end = __ldg(rowptr_data + (row % M) + 1);
    int col_idx = row_start + lane_idx;

    scalar_t result = Reducer<scalar_t, REDUCE>::init();
    int64_t arg;

    // Iterate over all `col` indices in parallel within a warp.
    for (int c = row_start; c < row_end; c += 32) {

      if (col_idx < row_end) {
        // Coalesced memory access into `col` and `val`.
        mat_row = __ldg(col_data + col_idx) * K;
        if (HAS_VALUE)
          val = __ldg(value_data + col_idx);
      } else {
        mat_row = -1;
        if (HAS_VALUE)
          val = (scalar_t)0;
      }
      col_idx += 32;

#pragma unroll
      for (int i = 0; i < 32; i++) {
        // Communication between all threads in a warp.
        mat_rows[i] = __shfl_sync(FULL_MASK, mat_row, i);
        if (HAS_VALUE)
          vals[i] = __shfl_sync(FULL_MASK, val, i);
      }

#pragma unroll
      for (int i = 0; i < 32; i++) {
        if (lane_idx < leftover && mat_rows[i] != -1) {
          // Coalesced memory access into `mat`.
          val = __ldg(mat_data + batch_idx * N * K + mat_rows[i] + mat_col_idx);
          if (HAS_VALUE)
            val = vals[i] * val;
          Reducer<scalar_t, REDUCE>::update(&result, val, &arg, c + i);
        }
      }
    }

    if (lane_idx < leftover) {
      // Coalesced write into `out`.
      Reducer<scalar_t, REDUCE>::write(out_data + out_idx, result,
                                       arg_out_data + out_idx, arg,
                                       row_end - row_start);
    }
  }
}

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_cuda(torch::Tensor rowptr, torch::Tensor col,
          torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
          std::string reduce) {

  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  if (optional_value.has_value())
    CHECK_CUDA(optional_value.value());
  CHECK_CUDA(mat);
  cudaSetDevice(rowptr.get_device());

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  if (optional_value.has_value()) {
    CHECK_INPUT(optional_value.value().dim() == 1);
    CHECK_INPUT(optional_value.value().size(0) == col.size(0));
  }
  CHECK_INPUT(mat.dim() >= 2);

  mat = mat.contiguous();

  auto sizes = mat.sizes().vec();
  sizes[mat.dim() - 2] = rowptr.numel() - 1;
  auto out = torch::empty(sizes, mat.options());

  torch::optional<torch::Tensor> arg_out = torch::nullopt;
  int64_t *arg_out_data = nullptr;
  if (reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX) {
    arg_out = torch::full_like(out, col.numel(), rowptr.options());
    arg_out_data = arg_out.value().data_ptr<int64_t>();
  }

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();

  auto M = rowptr.numel() - 1;
  auto N = mat.size(-2);
  auto K = mat.size(-1);
  auto B = mat.numel() / (N * K);
  auto BLOCKS = dim3((32 * B * M + THREADS - 1) / THREADS, (K + 31) / 32);

  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, mat.scalar_type(), "_", [&] {
    auto mat_data = mat.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      if (optional_value.has_value()) {
        auto value_data = optional_value.value().data_ptr<scalar_t>();
        spmm_kernel<scalar_t, REDUCE, true><<<BLOCKS, THREADS, 0, stream>>>(
            rowptr_data, col_data, value_data, mat_data, out_data, arg_out_data,
            B, M, N, K);
      } else {
        spmm_kernel<scalar_t, REDUCE, false><<<BLOCKS, THREADS, 0, stream>>>(
            rowptr_data, col_data, nullptr, mat_data, out_data, arg_out_data, B,
            M, N, K);
      }
    });
  });

  return std::make_tuple(out, arg_out);
}

template <typename scalar_t, ReductionType REDUCE>
__global__ void
spmm_value_bw_kernel(const int64_t *row_data, const int64_t *rowptr_data,
                     const int64_t *col_data, const scalar_t *mat_data,
                     const scalar_t *grad_data, scalar_t *out_data, int B,
                     int M, int N, int E, int K) {
  int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

  int index_idx = (thread_idx >> 5);    // thread_idx / 32
  int lane_idx = thread_idx & (32 - 1); // thread_idx % 32

  if (index_idx < E) {
    int row = __ldg(row_data + index_idx);
    int col = __ldg(col_data + index_idx);

    scalar_t val = (scalar_t)0;
    for (int b = 0; b < B; b++) {
      for (int k = lane_idx; k < K; k += 32) {
        val += mat_data[b * N * K + col * K + k] *
               grad_data[b * M * K + row * K + k];
      }
    }

#pragma unroll
    for (int i = 32 / 2; i > 0; i /= 2) { // Parallel reduction inside a warp.
      val += __shfl_down_sync(FULL_MASK, val, i);
    }

    if (lane_idx == 0) {
      if (REDUCE == MEAN) {
        int row_start = __ldg(rowptr_data + row);
        int row_end = __ldg(rowptr_data + row + 1);
        val /= (scalar_t)max(row_end - row_start, 1);
      }
      out_data[index_idx] = val;
    }
  }
}

torch::Tensor spmm_value_bw_cuda(torch::Tensor row, torch::Tensor rowptr,
                                 torch::Tensor col, torch::Tensor mat,
                                 torch::Tensor grad, std::string reduce) {
  CHECK_CUDA(row);
  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  CHECK_CUDA(mat);
  CHECK_CUDA(grad);
  cudaSetDevice(row.get_device());

  mat = mat.contiguous();
  grad = grad.contiguous();

  auto M = grad.size(-2);
  auto N = mat.size(-2);
  auto E = row.numel();
  auto K = mat.size(-1);
  auto B = mat.numel() / (N * K);
  auto BLOCKS = dim3((E * 32 + THREADS - 1) / THREADS);

  auto out = torch::zeros(row.numel(), grad.options());

  auto row_data = row.data_ptr<int64_t>();
  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();

  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, mat.scalar_type(), "_", [&] {
    auto mat_data = mat.data_ptr<scalar_t>();
    auto grad_data = grad.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      spmm_value_bw_kernel<scalar_t, REDUCE><<<BLOCKS, THREADS, 0, stream>>>(
          row_data, rowptr_data, col_data, mat_data, grad_data, out_data, B, M,
          N, E, K);
    });
  });

  return out;
}

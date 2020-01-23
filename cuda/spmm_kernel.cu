#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "compat.cuh"

#define THREADS 256
#define FULL_MASK 0xffffffff

enum ReductionType { SUM, MEAN, MIN, MAX };

const std::map<std::string, ReductionType> reduce2REDUCE = {
    {"sum", SUM}, {"add", SUM}, {"mean", MEAN}, {"min", MIN}, {"max", MAX},
};

#define AT_DISPATCH_REDUCTION_TYPES(reduce, ...)                               \
  [&] {                                                                        \
    switch (reduce2REDUCE.at(reduce)) {                                        \
    case SUM: {                                                                \
      const ReductionType REDUCE = SUM;                                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case MEAN: {                                                               \
      const ReductionType REDUCE = MEAN;                                       \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case MIN: {                                                                \
      const ReductionType REDUCE = MIN;                                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case MAX: {                                                                \
      const ReductionType REDUCE = MAX;                                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    }                                                                          \
  }()

template <typename scalar_t, ReductionType REDUCE> struct Reducer {
  static inline __host__ __device__ scalar_t init() {
    if (REDUCE == MIN) {
      return std::numeric_limits<scalar_t>::max();
    } else if (REDUCE == MAX) {
      return std::numeric_limits<scalar_t>::lowest();
    } else {
      return (scalar_t)0;
    }
  }

  static inline __host__ __device__ void update(scalar_t *val, scalar_t new_val,
                                                int64_t *arg, int64_t new_arg) {
    if (REDUCE == SUM || REDUCE == MEAN) {
      *val = *val + new_val;
    } else if ((REDUCE == MIN && new_val < *val) ||
               (REDUCE == MAX && new_val > *val)) {
      *val = new_val;
      *arg = new_arg;
    }
  }

  static inline __host__ __device__ void write(scalar_t *address, scalar_t val,
                                               int64_t *arg_address,
                                               int64_t arg, int count) {
    if (REDUCE == SUM) {
      *address = val;
    } else if (REDUCE == MEAN) {
      *address = val / (scalar_t)max(count, 1);
    } else if (REDUCE == MIN || REDUCE == MAX) {
      if (count > 0) {
        *address = val;
        *arg_address = arg;
      } else {
        *address = (scalar_t)0;
      }
    }
  }
};

// Paper: Design Principles for Sparse Matrix Multiplication on the GPU
// Code:  https://github.com/owensgroup/merge-spmm
template <typename scalar_t, ReductionType REDUCE, bool HAS_VAL>
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
  scalar_t val, vals[HAS_VAL ? 32 : 1];
  int bla, blas[32];

  // Do not aggregate/write across the Y-axis (lane_idx < leftover).
  int leftover = K - (blockIdx.y << 5);

  if (row < B * M) {
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
        bla = col_idx;
        if (HAS_VAL)
          val = __ldg(value_data + col_idx);
      } else {
        mat_row = -1;
        bla = -1;
        if (HAS_VAL)
          val = (scalar_t)0;
      }
      col_idx += 32;

#pragma unroll
      for (int i = 0; i < 32; i++) {
        // Communication between all threads in a warp.
        mat_rows[i] = __shfl_sync(FULL_MASK, mat_row, i);
        blas[i] = __shfl_sync(FULL_MASK, bla, i);
        if (HAS_VAL)
          vals[i] = __shfl_sync(FULL_MASK, val, i);
      }

#pragma unroll
      for (int i = 0; i < 32; i++) {
        if (lane_idx < leftover && mat_rows[i] != -1) {
          // Coalesced memory access into `mat`.
          val = __ldg(mat_data + batch_idx * N * K + mat_rows[i] + mat_col_idx);
          if (HAS_VAL)
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

std::tuple<at::Tensor, at::optional<at::Tensor>>
spmm_cuda(at::Tensor rowptr, at::Tensor col, at::optional<at::Tensor> value_opt,
          at::Tensor mat, std::string reduce) {

  AT_ASSERTM(rowptr.dim() == 1, "Input mismatch");
  AT_ASSERTM(col.dim() == 1, "Input mismatch");
  if (value_opt.has_value())
    AT_ASSERTM(value_opt.value().dim() == 1);
  AT_ASSERTM(mat.dim() >= 2, "Input mismatch");

  mat = mat.contiguous();

  auto sizes = mat.sizes().vec();
  sizes[mat.dim() - 2] = rowptr.numel() - 1;
  auto out = at::empty(sizes, mat.options());

  at::optional<at::Tensor> arg_out = at::nullopt;
  int64_t *arg_out_data = nullptr;
  if (reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX) {
    arg_out = at::full_like(out, col.numel(), rowptr.options());
    arg_out_data = arg_out.value().DATA_PTR<int64_t>();
  }

  auto rowptr_data = rowptr.DATA_PTR<int64_t>();
  auto col_data = col.DATA_PTR<int64_t>();

  auto M = rowptr.numel() - 1;
  auto N = mat.size(-2);
  auto K = mat.size(-1);
  auto B = mat.numel() / (N * K);
  auto BLOCKS = dim3((32 * B * M + THREADS - 1) / THREADS, (K + 31) / 32);

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES(mat.scalar_type(), "spmm_kernel", [&] {
    auto mat_data = mat.DATA_PTR<scalar_t>();
    auto out_data = out.DATA_PTR<scalar_t>();

    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      if (value_opt.has_value()) {
        auto value_data = value_opt.value().DATA_PTR<scalar_t>();
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

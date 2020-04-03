#include "degree_padding_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

__global__ void sizes_kernel(const int64_t *__restrict__ sorted_rowcount,
                             const int64_t *__restrict__ binptr,
                             int64_t *__restrict__ size,
                             int64_t *__restrict__ length,
                             const int64_t num_bins, const int64_t numel) {
  for (int64_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
       thread_idx < numel - 1; thread_idx += gridDim.x * blockDim.x) {

    int64_t deg1 = sorted_rowcount[thread_idx];
    int64_t deg2 = sorted_rowcount[thread_idx + 1];

    if (deg1 != deg2) {
      for (int64_t b = 1; b <= num_bins; b++) {
        if (deg1 < __ldg(binptr + b) && deg2 >= __ldg(binptr + b)) {
          size[b] = thread_idx + 1;
          length[b - 1] = deg1;
        }
      }
    }

    if (thread_idx + 1 == numel - 1) {
      size[num_bins] = numel;
      length[num_bins - 1] = deg2;
    }
  }
}

std::tuple<std::vector<torch::Tensor>, std::vector<int64_t>>
bin_assignment_cuda(torch::Tensor rowcount, torch::Tensor binptr) {
  CHECK_CUDA(rowcount);
  CHECK_CUDA(binptr);
  CHECK_INPUT(rowcount.dim() == 1);
  CHECK_INPUT(binptr.dim() == 1);

  cudaSetDevice(rowcount.get_device());
  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t mpc = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  torch::Tensor sorted_rowcount, perm;
  std::tie(sorted_rowcount, perm) = rowcount.sort();

  auto size = torch::zeros({binptr.numel()}, binptr.options());
  auto length = torch::zeros({binptr.numel() - 1}, binptr.options());

  sizes_kernel<<<std::min(BLOCKS(rowcount.numel() - 1), mpc * 8), THREADS, 0,
                 stream>>>(sorted_rowcount.data_ptr<int64_t>(),
                           binptr.data_ptr<int64_t>(), size.data_ptr<int64_t>(),
                           length.data_ptr<int64_t>(), length.numel(),
                           rowcount.numel());

  size = size.cpu();
  size = size.narrow(0, 1, length.numel()) - size.narrow(0, 0, length.numel());
  auto sizes = at::IntArrayRef(size.data_ptr<int64_t>(), size.numel());

  length = length.cpu();
  int64_t *length_data = length.data_ptr<int64_t>();
  std::vector<int64_t> lengths(length.numel());
  std::copy(length_data, length_data + length.numel(), lengths.begin());

  return std::make_tuple(perm.split_with_sizes(sizes), lengths);
}

__global__ void padded_mask_select_kernel(
    const int64_t *__restrict__ rowptr, const int64_t *__restrict__ col,
    const int64_t *__restrict__ index, int64_t *__restrict__ out_idx,
    bool *__restrict__ mask, const int64_t length, const int64_t numel) {

  int64_t lane_idx, row_idx, row_start, row_end, col_idx;
  for (int64_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
       thread_idx < numel; thread_idx += gridDim.x * blockDim.x) {
    lane_idx = thread_idx % length;
    row_idx = index[thread_idx / length];
    row_start = rowptr[row_idx];
    row_end = rowptr[row_idx + 1];
    col_idx = -1;
    if (lane_idx < row_end - row_start) {
      col_idx = col[row_start + lane_idx];
    }

    out_idx[thread_idx] = col_idx;
    mask[thread_idx] = col_idx == -1;
  }
}

template <typename scalar_t>
__global__ void
padded_index_select_kernel(const scalar_t *__restrict__ src,
                           const int64_t *__restrict__ index,
                           scalar_t *__restrict__ out, scalar_t fill_value,
                           const int64_t dim, const int64_t numel) {

  int64_t index_idx, dim_idx, col;
  for (int64_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
       thread_idx < numel; thread_idx += gridDim.x * blockDim.x) {
    index_idx = thread_idx / dim;
    dim_idx = thread_idx % dim;
    col = __ldg(index + index_idx);
    if (col >= 0) {
      fill_value = src[col * dim + dim_idx];
    }

    out[thread_idx] = fill_value;
  }
}

std::tuple<torch::Tensor, torch::Tensor>
padded_index_select_cuda(torch::Tensor src, torch::Tensor rowptr,
                         torch::Tensor col, torch::Tensor index, int64_t length,
                         torch::Tensor fill_value) {
  CHECK_CUDA(src);
  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  CHECK_CUDA(index);
  CHECK_INPUT(src.dim() == 2);
  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  CHECK_INPUT(index.dim() == 1);
  CHECK_INPUT(fill_value.numel() == 1);

  cudaSetDevice(src.get_device());
  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t mpc = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  auto out_idx = torch::empty({index.size(0), length}, index.options());
  auto out = torch::empty({index.size(0), length, src.size(-1)}, src.options());
  auto mask = torch::empty({index.size(0), length, 1},
                           src.options().dtype(torch::kBool));

  padded_mask_select_kernel<<<
      std::min((out_idx.numel() + THREADS - 1) / THREADS, mpc * 8), THREADS, 0,
      stream>>>(rowptr.data_ptr<int64_t>(), col.data_ptr<int64_t>(),
                index.data_ptr<int64_t>(), out_idx.data_ptr<int64_t>(),
                mask.data_ptr<bool>(), length, out_idx.numel());

  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "padded_index_select_kernel", [&] {
    scalar_t *fill;
    if (fill_value.is_cuda()) {
      fill = (scalar_t *)malloc(sizeof(scalar_t));
      cudaMemcpy(fill, fill_value.data_ptr<scalar_t>(), sizeof(scalar_t),
                 cudaMemcpyDeviceToHost);
    } else {
      fill = fill_value.data_ptr<scalar_t>();
    }

    padded_index_select_kernel<scalar_t>
        <<<std::min((out.numel() + THREADS - 1) / THREADS, mpc * 8), THREADS, 0,
           stream>>>(src.data_ptr<scalar_t>(), out_idx.data_ptr<int64_t>(),
                     out.data_ptr<scalar_t>(), fill[0], src.size(-1),
                     out.numel());
  });

  return std::make_tuple(out, mask);
}

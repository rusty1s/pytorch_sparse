#include "degree_padding_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

__global__ void bin_kernel(const int64_t *rowcount, const int64_t *bin_strategy,
                           int64_t *bin, int64_t *one_hot, int64_t num_bins,
                           int64_t numel) {
  int64_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (thread_idx < numel) {
    auto count = rowcount[thread_idx];

    int64_t b = -1;
    for (int64_t i = 0; i < num_bins; i++) {
      if (count >= __ldg(bin_strategy + 2 * i) &&
          count <= __ldg(bin_strategy + 2 * i + 1)) {
        b = i;
        break;
      }
    }

    bin[thread_idx] = b;
    if (b >= 0) {
      one_hot[b * numel + thread_idx] = 1;
    }
  }
}

__global__ void index_kernel(const int64_t *bin, const int64_t *cumsum,
                             const int64_t *nodes_per_bin, int64_t *index,
                             int64_t num_bins, int64_t numel) {
  int64_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (thread_idx < numel) {
    auto b = bin[thread_idx];
    if (b >= 0) {
      auto idx = cumsum[b * numel + thread_idx] - 1;
      for (int64_t i = 0; i < b; i++) {
        idx += __ldg(nodes_per_bin + i);
      }
      index[idx] = thread_idx;
    }
  }
}

std::vector<torch::Tensor> bin_assignment_cuda(torch::Tensor rowcount,
                                               torch::Tensor bin_strategy) {
  CHECK_CUDA(rowcount);
  CHECK_CUDA(bin_strategy);
  CHECK_INPUT(rowcount.dim() == 1);
  CHECK_INPUT(bin_strategy.dim() == 2 && bin_strategy.size(1) == 2);
  cudaSetDevice(rowcount.get_device());

  int64_t num_bins = bin_strategy.size(0);
  auto bin = torch::empty({rowcount.numel()}, rowcount.options());
  auto one_hot = torch::zeros({num_bins, rowcount.numel()}, rowcount.options());

  auto stream = at::cuda::getCurrentCUDAStream();
  bin_kernel<<<BLOCKS(rowcount.numel()), THREADS, 0, stream>>>(
      rowcount.data_ptr<int64_t>(), bin_strategy.data_ptr<int64_t>(),
      bin.data_ptr<int64_t>(), one_hot.data_ptr<int64_t>(), num_bins,
      rowcount.numel());

  auto cumsum = one_hot.cumsum(1);
  auto d_nodes_per_bin = cumsum.select(1, rowcount.numel() - 1).contiguous();
  auto h_nodes_per_bin = d_nodes_per_bin.cpu();

  auto h_size = h_nodes_per_bin.sum().data_ptr<int64_t>()[0];
  auto index = torch::empty({h_size}, rowcount.options());

  index_kernel<<<BLOCKS(bin.numel()), THREADS, 0, stream>>>(
      bin.data_ptr<int64_t>(), cumsum.data_ptr<int64_t>(),
      d_nodes_per_bin.data_ptr<int64_t>(), index.data_ptr<int64_t>(), num_bins,
      rowcount.numel());

  auto sizes = at::IntArrayRef(h_nodes_per_bin.data_ptr<int64_t>(), num_bins);
  return index.split_with_sizes(sizes);
}

__global__ void padded_mask_select_kernel(const int64_t *rowptr,
                                          const int64_t *col,
                                          const int64_t *index,
                                          int64_t *out_idx, bool *mask,
                                          int64_t length, int64_t numel) {

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
__global__ void padded_index_select_kernel(const scalar_t *src,
                                           const int64_t *index, scalar_t *out,
                                           scalar_t fill_value, int64_t dim,
                                           int64_t numel) {

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

  auto out_idx = torch::empty({index.size(0), length}, index.options());
  auto mask = torch::empty({index.size(0), length, 1},
                           src.options().dtype(torch::kBool));

  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t mpc = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  padded_mask_select_kernel<<<
      std::min((out_idx.numel() + THREADS - 1) / THREADS, mpc * 8), THREADS, 0,
      stream>>>(rowptr.data_ptr<int64_t>(), col.data_ptr<int64_t>(),
                index.data_ptr<int64_t>(), out_idx.data_ptr<int64_t>(),
                mask.data_ptr<bool>(), length, out_idx.numel());

  auto out = torch::empty({index.size(0), length, src.size(-1)}, src.options());
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

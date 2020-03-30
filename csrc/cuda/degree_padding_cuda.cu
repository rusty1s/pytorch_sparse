#include "degree_padding_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"

#define THREADS 256
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

template <typename scalar_t, int64_t TB>
__global__ void
padded_index_select_kernel(const scalar_t *src, const int64_t *rowptr,
                           const int64_t *col, const int64_t *index,
                           scalar_t *out, bool *mask, int64_t length,
                           int64_t dim, int64_t numel) {

  int64_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  auto dim_idx = thread_idx % dim;
  auto lane_idx = (thread_idx / dim) % TB;
  auto index_idx = thread_idx / (TB * dim);

  if (thread_idx < numel) {
    auto row_idx = __ldg(index + index_idx);
    auto row_start = __ldg(rowptr + row_idx);
    auto row_end = __ldg(rowptr + row_idx + 1);

    for (int64_t c = lane_idx; c < row_end - row_start; c += TB) {
      auto x = src[__ldg(col + row_start + c) * dim + dim_idx];
      out[index_idx * dim * length + c * dim + dim_idx] = x;
      // mask[index_idx * dim * length + c * dim + dim_idx] = true;
    }
  }
}

#define TB 4

std::tuple<torch::Tensor, torch::Tensor>
padded_index_select_cuda(torch::Tensor src, torch::Tensor rowptr,
                         torch::Tensor col, torch::Tensor index,
                         int64_t length) {
  CHECK_CUDA(src);
  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  CHECK_CUDA(index);
  CHECK_INPUT(src.dim() == 2);
  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  CHECK_INPUT(index.dim() == 1);
  cudaSetDevice(src.get_device());

  auto out = torch::zeros({index.size(0), length, src.size(-1)}, src.options());
  auto mask =
      torch::zeros({index.size(0), length}, src.options().dtype(torch::kBool));

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "padded_index_select_kernel", [&] {
    padded_index_select_kernel<scalar_t, TB>
        <<<BLOCKS(index.numel() * src.size(-1) * TB), THREADS, 0, stream>>>(
            src.data_ptr<scalar_t>(), rowptr.data_ptr<int64_t>(),
            col.data_ptr<int64_t>(), index.data_ptr<int64_t>(),
            out.data_ptr<scalar_t>(), mask.data_ptr<bool>(), length,
            src.size(-1), index.numel() * src.size(-1) * TB);
  });

  return std::make_tuple(out, mask);
}

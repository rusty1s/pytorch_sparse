#include "padding_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "atomics.cuh"
#include "utils.cuh"

#define THREADS 1024
#define FULL_MASK 0xffffffff
#define BLOCKS(N) (N + THREADS - 1) / THREADS

__global__ void bin_kernel(const int64_t *__restrict__ rowcount,
                           const int64_t *__restrict__ binptr,
                           int64_t *__restrict__ bin, int64_t *__restrict__ idx,
                           int *__restrict__ node_size,
                           int *__restrict__ max_deg, const size_t B,
                           const size_t N) {

  for (ptrdiff_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
       thread_idx < N; thread_idx += gridDim.x * blockDim.x) {

    int bin_idx = -1, deg = rowcount[thread_idx];
    for (ptrdiff_t b = 1; b <= B; b++) {
      if (deg < __ldg(binptr + b)) {
        bin_idx = b - 1;
        break;
      }
    }

    if (bin_idx == -1) {
      bin_idx = B - 1;
    }

    int old = atomicAdd(node_size + bin_idx, 1);
    atomicMax(max_deg + bin_idx, deg);

    bin[thread_idx] = bin_idx;
    idx[thread_idx] = old;
  }
}

__global__ void info_kernel(const int *__restrict__ node_size,
                            const int *__restrict__ max_deg,
                            int *__restrict__ edge_size,
                            int *__restrict__ node_offset,
                            int *__restrict__ edge_offset, const size_t B) {

  int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  int bin_idx = thread_idx / 32;
  int lane_idx = thread_idx % 32;

  if (bin_idx <= B) { // Computes `node_offset` and `edge_offset`.
    int node_tmp = 0;
    int edge_tmp = 0;

    for (int i = lane_idx; i < bin_idx; i += 32) {
      node_tmp += node_size[i];
      edge_tmp += node_size[i] * max_deg[i];
    }

    for (int i = 32 / 2; i > 0; i /= 2) {
      node_tmp += __shfl_down_sync(FULL_MASK, node_tmp, i);
      edge_tmp += __shfl_down_sync(FULL_MASK, edge_tmp, i);
    }

    if (lane_idx == 0) {
      node_offset[bin_idx] = node_tmp;
      edge_offset[bin_idx] = edge_tmp;
    }
  } else if (bin_idx == B + 1) { // Computes `edge_size`.
    for (int i = lane_idx; i < B; i += 32) {
      edge_size[i] = node_size[i] * max_deg[i];
    }
  }
}

__global__ void node_perm_kernel(const int64_t *__restrict__ bin,
                                 const int64_t *__restrict__ idx,
                                 const int *__restrict__ node_offset,
                                 int64_t *__restrict__ out, const size_t N) {

  for (ptrdiff_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
       thread_idx < N; thread_idx += gridDim.x * blockDim.x) {
    out[__ldg(node_offset + bin[thread_idx]) + idx[thread_idx]] = thread_idx;
  }
}

template <int TB>
__global__ void padded_index_kernel(
    const int64_t *__restrict__ rowptr, const int64_t *__restrict__ col,
    const int64_t *__restrict__ rowcount, const int64_t *__restrict__ bin,
    const int64_t *__restrict__ idx, const int *__restrict__ max_deg,
    const int *__restrict__ edge_offset, int64_t *__restrict__ row_perm,
    int64_t *__restrict__ col_perm, bool *__restrict__ edge_mask,
    const size_t B, const size_t N) {

  for (ptrdiff_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
       thread_idx < TB * N; thread_idx += gridDim.x * blockDim.x) {

    int row_idx = thread_idx / TB;
    int lane_idx = thread_idx % TB;

    int64_t bin_idx = bin[row_idx];
    int len = __ldg(max_deg + bin_idx);
    int off = __ldg(edge_offset + bin_idx) + len * idx[row_idx];

    int64_t row_start = rowptr[row_idx], deg = rowcount[row_idx];
    int64_t row_tmp, col_tmp;
    for (int i = lane_idx; i < len; i += TB) {
      row_tmp = -1, col_tmp = -1;
      if (i < deg) {
        row_tmp = row_idx;
        col_tmp = col[row_start + i];
      }
      row_perm[off + i] = row_tmp;
      col_perm[off + i] = col_tmp;
      edge_mask[off + i] = row_tmp == -1;
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<int64_t>, std::vector<int64_t>>
padded_index_cuda(torch::Tensor rowptr, torch::Tensor col,
                  torch::Tensor rowcount, torch::Tensor binptr) {
  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  CHECK_CUDA(rowcount);
  CHECK_CUDA(binptr);
  CHECK_INPUT(rowptr.numel() == rowcount.numel() + 1);

  cudaSetDevice(rowcount.get_device());
  auto stream = at::cuda::getCurrentCUDAStream();
  size_t mpc = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  size_t B = binptr.numel() - 1;
  size_t N = rowcount.numel();

  auto bin = torch::empty(N, col.options());
  auto idx = torch::empty(N, col.options());

  auto d_info = torch::zeros(5 * B + 2, col.options().dtype(torch::kInt));
  auto d_node_size = d_info.narrow(0, 0, B);
  auto d_edge_size = d_info.narrow(0, B, B);
  auto d_max_deg = d_info.narrow(0, 2 * B, B);
  auto d_node_offset = d_info.narrow(0, 3 * B, B + 1);
  auto d_edge_offset = d_info.narrow(0, 4 * B + 1, B + 1);

  bin_kernel<<<std::min(BLOCKS(N), mpc * 8), THREADS, 0, stream>>>(
      rowcount.data_ptr<int64_t>(), binptr.data_ptr<int64_t>(),
      bin.data_ptr<int64_t>(), idx.data_ptr<int64_t>(),
      d_node_size.data_ptr<int>(), d_max_deg.data_ptr<int>(), B, N);

  info_kernel<<<BLOCKS(32 * (B + 2)), THREADS, 0, stream>>>(
      d_node_size.data_ptr<int>(), d_max_deg.data_ptr<int>(),
      d_edge_size.data_ptr<int>(), d_node_offset.data_ptr<int>(),
      d_edge_offset.data_ptr<int>(), B);

  auto node_perm = torch::empty(N, col.options());

  node_perm_kernel<<<std::min(BLOCKS(N), mpc * 8), THREADS, 0, stream>>>(
      bin.data_ptr<int64_t>(), idx.data_ptr<int64_t>(),
      d_node_offset.data_ptr<int>(), node_perm.data_ptr<int64_t>(), N);

  auto h_info = torch::empty(
      d_info.numel(), d_info.options().device(torch::kCPU).pinned_memory(true));
  cudaMemcpy(h_info.data_ptr<int>(), d_info.data_ptr<int>(),
             d_info.numel() * sizeof(int), cudaMemcpyDeviceToHost);

  size_t E = h_info.data_ptr<int>()[5 * B + 1];
  auto row_perm = torch::empty(E, col.options());
  auto col_perm = torch::empty(E, col.options());
  auto edge_mask = torch::empty(E, col.options().dtype(torch::kBool));

  padded_index_kernel<8>
      <<<std::min(BLOCKS(N * 8), mpc * 8), THREADS, 0, stream>>>(
          rowptr.data_ptr<int64_t>(), col.data_ptr<int64_t>(),
          rowcount.data_ptr<int64_t>(), bin.data_ptr<int64_t>(),
          idx.data_ptr<int64_t>(), d_max_deg.data_ptr<int>(),
          d_edge_offset.data_ptr<int>(), row_perm.data_ptr<int64_t>(),
          col_perm.data_ptr<int64_t>(), edge_mask.data_ptr<bool>(), B, N);

  h_info = h_info.to(torch::kLong);
  auto h_info_data = h_info.data_ptr<int64_t>();
  std::vector<int64_t> node_sizes(h_info_data, h_info_data + B);
  std::vector<int64_t> edge_sizes(h_info_data + B, h_info_data + 2 * B);

  return std::make_tuple(node_perm, row_perm, col_perm, edge_mask, node_sizes,
                         edge_sizes);
}

template <typename scalar_t>
__global__ void padded_index_select_kernel(const scalar_t *__restrict__ src,
                                           const int64_t *__restrict__ index,
                                           scalar_t *__restrict__ out,
                                           const scalar_t fill_value,
                                           const size_t E, const size_t F) {

  for (ptrdiff_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
       thread_idx < E * F; thread_idx += gridDim.x * blockDim.x) {

    int64_t row_idx = thread_idx / F;
    int64_t lane_idx = thread_idx % F;
    int64_t index_idx = __ldg(index + row_idx);

    scalar_t tmp = fill_value;
    if (index_idx != -1) {
      tmp = src[index_idx * F + lane_idx];
    }

    out[thread_idx] = tmp;
  }
}

torch::Tensor padded_index_select_cuda(torch::Tensor src, torch::Tensor index,
                                       torch::Tensor fill_value) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  CHECK_INPUT(src.dim() == 2);
  CHECK_INPUT(index.dim() == 1);

  cudaSetDevice(src.get_device());
  auto stream = at::cuda::getCurrentCUDAStream();
  size_t mpc = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  size_t E = index.numel();
  size_t F = src.size(-1);

  auto out = torch::empty({(int)E, (int)F}, src.options());

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
        <<<std::min(BLOCKS(E * F), mpc * 8), THREADS, 0, stream>>>(
            src.data_ptr<scalar_t>(), index.data_ptr<int64_t>(),
            out.data_ptr<scalar_t>(), fill[0], E, F);
  });

  return out;
}

template <typename scalar_t>
__global__ void padded_index_scatter_kernel(const scalar_t *__restrict__ src,
                                            const int64_t *__restrict__ index,
                                            scalar_t *__restrict__ out,
                                            const size_t E, const size_t F) {

  for (ptrdiff_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
       thread_idx < E * F; thread_idx += gridDim.x * blockDim.x) {

    int64_t index_idx = __ldg(index + thread_idx / F);
    if (index_idx != -1) {
      atomAdd(out + index_idx * F + thread_idx % F, src[thread_idx]);
    }
  }
}

torch::Tensor padded_index_scatter_cuda(torch::Tensor src, torch::Tensor index,
                                        int64_t N) {
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  CHECK_INPUT(src.dim() == 2);
  CHECK_INPUT(index.dim() == 1);
  CHECK_INPUT(src.size(0) == index.size(0));

  cudaSetDevice(src.get_device());
  auto stream = at::cuda::getCurrentCUDAStream();
  size_t mpc = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  size_t E = index.numel();
  size_t F = src.size(-1);

  auto out = torch::zeros({N, (int)F}, src.options());

  AT_DISPATCH_FLOATING_TYPES(
      src.scalar_type(), "padded_index_scatter_kernel", [&] {
        padded_index_scatter_kernel<scalar_t>
            <<<std::min(BLOCKS(E * F), mpc * 8), THREADS, 0, stream>>>(
                src.data_ptr<scalar_t>(), index.data_ptr<int64_t>(),
                out.data_ptr<scalar_t>(), E, F);
      });

  return out;
}

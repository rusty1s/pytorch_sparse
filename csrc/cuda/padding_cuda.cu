#include "padding_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"

#define THREADS 1024
#define FULL_MASK 0xffffffff
#define BLOCKS(N) (N + THREADS - 1) / THREADS

__global__ void bin_kernel(const int64_t *__restrict__ rowcount,
                           const int64_t *__restrict__ binptr,
                           int64_t *__restrict__ bin, int64_t *__restrict__ idx,
                           int *__restrict__ size, int *__restrict__ length,
                           const size_t B, const size_t N) {
  for (ptrdiff_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
       thread_idx < N; thread_idx += gridDim.x * blockDim.x) {

    int bin_idx = -1, deg = rowcount[thread_idx];
    for (ptrdiff_t b = 1; b <= B; b++) {
      if (deg < __ldg(binptr + b)) {
        bin_idx = b - 1;
        break;
      }
    }

    if (bin_idx == -1)
      bin_idx = B - 1;

    int old = atomicAdd(size + bin_idx, 1);
    atomicMax(length + bin_idx, deg);

    bin[thread_idx] = bin_idx;
    idx[thread_idx] = old;
  }
}

__global__ void offset_kernel(const int *__restrict__ size,
                              const int *__restrict__ length,
                              int *__restrict__ offset, const size_t B) {

  int bin_idx = threadIdx.x / 32;
  int lane_idx = threadIdx.x % 32;

  if (bin_idx <= B) {
    int tmp = 0;
    for (int i = lane_idx; i < bin_idx; i += 32) {
      tmp += size[i] * length[i];
    }

    for (int i = 32 / 2; i > 0; i /= 2) {
      tmp += __shfl_down_sync(FULL_MASK, tmp, i);
    }

    if (lane_idx == 0)
      offset[bin_idx] = tmp;
  }
}

template <int TB>
__global__ void padded_index_kernel(
    const int64_t *__restrict__ rowptr, const int64_t *__restrict__ rowcount,
    const int64_t *__restrict__ bin, const int64_t *__restrict__ idx,
    int64_t *__restrict__ out, bool *__restrict__ mask,
    const int *__restrict__ length, const int *__restrict__ offset,
    const size_t B, const size_t N) {

  for (ptrdiff_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
       thread_idx < TB * N; thread_idx += gridDim.x * blockDim.x) {

    int row_idx = thread_idx / TB;
    int lane_idx = thread_idx % TB;

    int64_t bin_idx = bin[row_idx];
    int len = __ldg(length + bin_idx);
    int off = __ldg(offset + bin_idx) + len * idx[row_idx];

    int64_t row_start = rowptr[row_idx], deg = rowcount[row_idx];
    int64_t tmp;
    for (int i = lane_idx; i < len; i += TB) {
      tmp = -1;
      if (i < deg)
        tmp = row_start + i;
      out[off + i] = tmp;
      mask[off + i] = tmp == -1;
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
padded_index_cuda(torch::Tensor rowptr, torch::Tensor rowcount,
                  torch::Tensor binptr) {
  // TODO: Add checks

  cudaSetDevice(rowcount.get_device());
  auto stream = at::cuda::getCurrentCUDAStream();
  size_t mpc = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  size_t B = binptr.numel() - 1;
  size_t N = rowcount.numel();

  auto bin = torch::empty(N, rowptr.options());
  auto idx = torch::empty(N, rowptr.options());

  auto tmp = torch::zeros(B + B + B + 1, rowcount.options().dtype(torch::kInt));
  auto size = tmp.narrow(0, 0, B);
  auto length = tmp.narrow(0, B, B);
  auto offset = tmp.narrow(0, 2 * B, B + 1);

  bin_kernel<<<std::min(BLOCKS(N), mpc * 8), THREADS, 0, stream>>>(
      rowcount.data_ptr<int64_t>(), binptr.data_ptr<int64_t>(),
      bin.data_ptr<int64_t>(), idx.data_ptr<int64_t>(), size.data_ptr<int>(),
      length.data_ptr<int>(), B, N);

  offset_kernel<<<BLOCKS(32 * (B + 1)), THREADS, 0, stream>>>(
      size.data_ptr<int>(), length.data_ptr<int>(), offset.data_ptr<int>(), B);

  auto h_tmp = torch::empty(
      {tmp.numel()}, tmp.options().device(torch::kCPU).pinned_memory(true));
  cudaMemcpy(h_tmp.data_ptr<int>(), tmp.data_ptr<int>(),
             tmp.numel() * sizeof(int), cudaMemcpyDeviceToHost);

  auto out = torch::empty({h_tmp.data_ptr<int>()[3 * B]}, rowptr.options());
  auto mask = torch::empty({out.numel()}, rowptr.options().dtype(torch::kBool));

  padded_index_kernel<8>
      <<<std::min(BLOCKS(N * 8), mpc * 8), THREADS, 0, stream>>>(
          rowptr.data_ptr<int64_t>(), rowcount.data_ptr<int64_t>(),
          bin.data_ptr<int64_t>(), idx.data_ptr<int64_t>(),
          out.data_ptr<int64_t>(), mask.data_ptr<bool>(),
          length.data_ptr<int>(), offset.data_ptr<int>(), B, N);

  return std::make_tuple(out, mask, h_tmp.narrow(0, 0, B),
                         h_tmp.narrow(0, B, B), h_tmp.narrow(0, 2 * B, B + 1));
}

template <typename scalar_t>
__global__ void padded_index_select_kernel(const scalar_t *__restrict__ src,
                                           const int64_t *__restrict__ col,
                                           const int64_t *__restrict__ index,
                                           scalar_t *__restrict__ out,
                                           const scalar_t fill_value,
                                           const size_t F, const size_t E) {

  for (ptrdiff_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
       thread_idx < E * F; thread_idx += gridDim.x * blockDim.x) {

    int64_t row_idx = thread_idx / F;
    int64_t lane_idx = thread_idx % F;
    int64_t index_idx = __ldg(index + row_idx);

    scalar_t tmp = fill_value;
    if (index_idx != -1) {
      tmp = src[__ldg(col + index_idx) * F + lane_idx];
    }

    out[thread_idx] = tmp;
  }
}

torch::Tensor padded_index_select_cuda(torch::Tensor src, torch::Tensor col,
                                       torch::Tensor index,
                                       torch::Tensor fill_value) {
  // TODO: Add checks

  cudaSetDevice(src.get_device());
  auto stream = at::cuda::getCurrentCUDAStream();
  size_t mpc = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  size_t E = index.numel();
  size_t F = src.size(-1);

  auto out = torch::empty(E * F, src.options());

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
            src.data_ptr<scalar_t>(), col.data_ptr<int64_t>(),
            index.data_ptr<int64_t>(), out.data_ptr<scalar_t>(), fill[0], F, E);
  });

  return out;
}

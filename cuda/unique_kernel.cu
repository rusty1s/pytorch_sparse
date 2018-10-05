#include <ATen/ATen.h>

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t>
__global__ void unique_cuda_kernel(scalar_t *__restrict__ src, uint8_t *mask,
                                   size_t numel) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (ptrdiff_t i = index; i < numel; i += stride) {
    if (i == 0 || src[i] != src[i - 1]) {
      mask[i] = 1;
    }
  }
}

std::tuple<at::Tensor, at::Tensor> unique_cuda(at::Tensor src) {
  at::Tensor perm;
  std::tie(src, perm) = src.sort();

  auto mask = at::zeros(src.numel(), src.type().toScalarType(at::kByte));
  AT_DISPATCH_ALL_TYPES(src.type(), "grid_cuda_kernel", [&] {
    unique_cuda_kernel<scalar_t><<<BLOCKS(src.numel()), THREADS>>>(
        src.data<scalar_t>(), mask.data<uint8_t>(), src.numel());
  });

  src = src.masked_select(mask);
  perm = perm.masked_select(mask);

  return std::make_tuple(src, perm);
}

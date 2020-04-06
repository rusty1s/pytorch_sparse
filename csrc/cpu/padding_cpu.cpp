#include "padding_cpu.h"

#include "utils.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<int64_t>, std::vector<int64_t>>
padded_index_cpu(torch::Tensor rowptr, torch::Tensor col,
                 torch::Tensor rowcount, torch::Tensor binptr) {
  std::vector<int64_t> bla = {1};
  return std::make_tuple(col, col, col, col, bla, bla);
}

torch::Tensor padded_index_select_cpu(torch::Tensor src, torch::Tensor index,
                                      torch::Tensor fill_value) {
  CHECK_CPU(src);
  CHECK_CPU(index);
  CHECK_INPUT(src.dim() == 2);
  CHECK_INPUT(index.dim() == 1);

  auto mask = index == -1;

  auto out = src.index_select(0, index.masked_fill(mask, 0));
  out.masked_fill_(mask.view({-1, 1}).expand_as(out), fill_value);

  return out;
}

torch::Tensor padded_index_scatter_cpu(torch::Tensor src, torch::Tensor index,
                                       int64_t N) {
  CHECK_CPU(src);
  CHECK_CPU(index);
  CHECK_INPUT(src.dim() == 2);
  CHECK_INPUT(index.dim() == 1);
  CHECK_INPUT(src.size(0) == index.size(0));

  auto mask = index == -1;
  index = index.masked_fill(mask, N);

  auto out = torch::zeros({N + 1, src.size(-1)}, src.options());
  out.scatter_add_(0, index.view({-1, 1}).expand_as(src), src);
  out = out.narrow(0, 0, N);

  return out;
}

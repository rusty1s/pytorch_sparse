#include "padding_cpu.h"

#include "utils.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<int64_t>, std::vector<int64_t>>
padded_index_cpu(torch::Tensor rowptr, torch::Tensor col,
                 torch::Tensor rowcount, torch::Tensor binptr) {
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  CHECK_CPU(rowcount);
  CHECK_CPU(binptr);
  CHECK_INPUT(rowptr.numel() == rowcount.numel() + 1);

  ptrdiff_t B = binptr.numel() - 1;
  ptrdiff_t N = rowcount.numel();

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  auto rowcount_data = rowcount.data_ptr<int64_t>();
  auto binptr_data = binptr.data_ptr<int64_t>();

  auto bin = torch::empty(N, col.options());
  auto bin_data = bin.data_ptr<int64_t>();
  auto idx = torch::empty(N, col.options());
  auto idx_data = idx.data_ptr<int64_t>();

  std::vector<int64_t> node_sizes(B), edge_sizes(B), max_degs(B),
      node_offsets(B + 1), edge_offsets(B + 1);

  int64_t deg, bin_idx = -1;
  for (ptrdiff_t n = 0; n < N; n++) {
    deg = rowcount_data[n];

    for (ptrdiff_t b = 1; b <= B; b++) {
      if (deg < binptr_data[b]) {
        bin_idx = b - 1;
        break;
      }
    }

    if (bin_idx == -1) {
      bin_idx = B - 1;
    }

    bin_data[n] = bin_idx;
    idx_data[n] = node_sizes[bin_idx];

    node_sizes[bin_idx] += 1;
    max_degs[bin_idx] = std::max(max_degs[bin_idx], deg);
  }

  for (ptrdiff_t b = 0; b < B; b++) {
    edge_sizes[b] = node_sizes[b] * max_degs[b];
    node_offsets[b + 1] = node_offsets[b] + node_sizes[b];
    edge_offsets[b + 1] = edge_offsets[b] + edge_sizes[b];
  }

  auto node_perm = torch::empty(N, col.options());
  auto node_perm_data = node_perm.data_ptr<int64_t>();

  auto E = edge_offsets[B];
  auto row_perm = torch::empty(E, col.options());
  auto row_perm_data = row_perm.data_ptr<int64_t>();
  auto col_perm = torch::empty(E, col.options());
  auto col_perm_data = col_perm.data_ptr<int64_t>();
  auto edge_mask = torch::empty(E, col.options().dtype(torch::kBool));
  auto edge_mask_data = edge_mask.data_ptr<bool>();

  int64_t row_start = rowptr_data[0], row_end, edge_offset, offset;
  for (ptrdiff_t n = 0; n < N; n++) {
    bin_idx = bin_data[n];
    offset = idx_data[n];

    node_perm_data[node_offsets[bin_idx] + offset] = n;

    row_end = rowptr_data[n + 1];
    edge_offset = edge_offsets[bin_idx] + offset * max_degs[bin_idx];
    for (ptrdiff_t e = 0; e < row_end - row_start; e++) {
      row_perm_data[edge_offset + e] = n;
      col_perm_data[edge_offset + e] = col_data[row_start + e];
      edge_mask_data[edge_offset + e] = false;
    }

    for (ptrdiff_t e = row_end - row_start; e < max_degs[bin_data[n]]; e++) {
      row_perm_data[edge_offset + e] = -1;
      col_perm_data[edge_offset + e] = -1;
      edge_mask_data[edge_offset + e] = true;
    }

    row_start = row_end;
  }

  return std::make_tuple(node_perm, row_perm, col_perm, edge_mask, node_sizes,
                         edge_sizes);
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

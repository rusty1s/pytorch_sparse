from torch_scatter import scatter_add


def spmm(index, value, m, matrix):
    """Matrix product of sparse matrix with dense matrix."""

    row, col = index
    matrix = matrix if matrix.dim() > 1 else matrix.unsqueeze(-1)

    out = matrix[col]
    out = out * value.unsqueeze(-1)
    out = scatter_add(out, row, dim=0, dim_size=m)

    return out

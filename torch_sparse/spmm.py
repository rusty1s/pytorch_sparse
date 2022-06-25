from torch import Tensor
from torch_scatter import scatter_add


def spmm(index: Tensor, value: Tensor, m: int, n: int,
         matrix: Tensor) -> Tensor:
    """Matrix product of sparse matrix with dense matrix.

    Args:
        index (:class:`LongTensor`): The index tensor of sparse matrix.
        value (:class:`Tensor`): The value tensor of sparse matrix, either of
            floating-point or integer type. Does not work for boolean and
            complex number data types.
        m (int): The first dimension of sparse matrix.
        n (int): The second dimension of sparse matrix.
        matrix (:class:`Tensor`): The dense matrix of same type as
            :obj:`value`.

    :rtype: :class:`Tensor`
    """

    assert n == matrix.size(-2)

    row, col = index[0], index[1]
    matrix = matrix if matrix.dim() > 1 else matrix.unsqueeze(-1)

    out = matrix.index_select(-2, col)
    out = out * value.unsqueeze(-1)
    out = scatter_add(out, row, dim=-2, dim_size=m)

    return out

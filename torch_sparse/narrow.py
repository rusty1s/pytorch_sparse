import torch
from torch_sparse.tensor import SparseTensor


def narrow(src, dim, start, length):
    if dim  == 0:
        col, rowptr, value = src.csr()
        rowptr = rowptr.narrow(0, start=start, length=length)

        row_start, row_end = rowptr[0]
        row_length = rowptr[-1] - row_start

        col = col.narrow(0, row_start, row_length)
        row = self._row.narrow(0, row_start, row_length)




    elif dim == 0:

    else:


    pass


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    row = torch.tensor([0, 0, 1, 1], device=device)
    col = torch.tensor([1, 2, 0, 2], device=device)
    sparse_mat = SparseTensor(torch.stack([row, col], dim=0))
    print(sparse_mat)
    print(sparse_mat.to_dense())

import os
import wget
import time
import errno
import argparse
import os.path as osp

import torch
import torch_sparse  # noqa
from scipy.io import loadmat

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='/tmp/test_ptr2ind')
args = parser.parse_args()

matrices = [
    ('DIMACS10', 'citationCiteseer'),
    ('SNAP', 'web-Stanford'),
    ('Janna', 'StocF-1465'),
    ('GHS_psdef', 'ldoor'),
]


def get_torch_sparse_coo_tensor(root, group, name):
    path = osp.join(root, f'{name}.mat')
    if not osp.exists(path):
        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno != errno.EEXIST and osp.isdir(path):
                raise e
        url = f'https://sparse.tamu.edu/mat/{group}/{name}.mat'
        print(f'Downloading {group}/{name}:')
        wget.download(url, path)
    matrix = loadmat(path)['Problem'][0][0][2].tocoo()
    row = torch.from_numpy(matrix.row).to(torch.long)
    col = torch.from_numpy(matrix.col).to(torch.long)
    index = torch.stack([row, col], dim=0)
    value = torch.from_numpy(matrix.data).to(torch.float)
    print(f'{name}.mat: shape={matrix.shape} nnz={row.numel()}')
    return torch.sparse_coo_tensor(index, value, matrix.shape).coalesce()


def time_func(matrix, op, duration=5.0, warmup=1.0):
    t = time.time()
    while (time.time() - t) < warmup:
        op(matrix)
    count = 0
    t = time.time()
    while (time.time() - t) < duration:
        op(matrix)
        count += 1
    return (time.time() - t) / count


def op1(matrix):
    # https://github.com/pearu/gcs/blob/b54ba0cba9c853b797274ef26b6c42386f2cafa3/gcs/storage.py#L24-L45
    row_indices = matrix.indices()[0]
    nnz = row_indices.shape[0]
    compressed = [0] * (matrix.shape[0] + 1)

    k = 1
    last_index = 0
    for i in range(nnz):
        index = row_indices[i]
        for n in range(last_index, index):
            compressed[k] = i
            k += 1
        last_index = index

    for n in range(k, matrix.shape[0] + 1):
        compressed[n] = nnz

    torch.tensor(compressed, dtype=torch.long)


def op2(matrix):
    # https://github.com/pytorch/pytorch/blob/3a777b67926c5f02bc287b25e572c521d6d11fb0/torch/_tensor.py#L928-L940
    row_indices = matrix.indices()[0]
    ro = [0]
    i = 0
    for irow in range(matrix.shape[0]):
        while i < row_indices.shape[0] and row_indices[i] == irow:
            i += 1
        ro.append(i)
    torch.tensor(ro, dtype=torch.long)


def op3(matrix):
    row_indices = matrix.indices()[0]

    bincount = torch.bincount(row_indices)

    out = torch.empty((matrix.shape[0] + 1), dtype=torch.long)
    out[0] = 0
    torch.cumsum(bincount, dim=0, out=out[1:])
    out[bincount.numel() + 1:] = row_indices.shape[0]


def op4(matrix):
    row_indices = matrix.indices()[0]
    torch.ops.torch_sparse.ind2ptr(row_indices, matrix.shape[0])


for group, name in matrices:
    matrix = get_torch_sparse_coo_tensor(args.root, group, name)

    duration = time_func(matrix, op1, duration=5.0, warmup=1.0)
    print('current implementation', duration)
    duration = time_func(matrix, op2, duration=5.0, warmup=1.0)
    print('compressed indices implementation', duration)
    duration = time_func(matrix, op3, duration=5.0, warmup=1.0)
    print('vectorized implementation:', duration)
    duration = time_func(matrix, op4, duration=5.0, warmup=1.0)
    print('torch-sparse implementation:', duration)

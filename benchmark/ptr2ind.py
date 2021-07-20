import os
import wget
import time
import errno
import argparse
import os.path as osp

import torch
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
    torch.cuda.synchronize()
    count = 0
    t = time.time()
    while (time.time() - t) < duration:
        op(matrix)
        count += 1
        torch.cuda.synchronize()
    return (time.time() - t) / count


def bucketize(matrix):
    row_indices = matrix.indices()[0]
    arange = torch.arange(matrix.size(0) + 1, device=row_indices.device)
    return torch.bucketize(arange, row_indices)


def convert_coo_to_csr(matrix):
    row_indices = matrix.indices()[0]
    return torch._convert_coo_to_csr(row_indices, matrix.size(0))


for device in ['cpu', 'cuda']:
    print('DEVICE:', device)
    for group, name in matrices:
        matrix = get_torch_sparse_coo_tensor(args.root, group, name)
        matrix = matrix.to(device)

        out1 = bucketize(matrix)
        out2 = convert_coo_to_csr(matrix)
        assert out1.tolist() == out2.tolist()

        t = time_func(matrix, bucketize, duration=5.0, warmup=1.0)
        print('old impl:', t)
        t = time_func(matrix, convert_coo_to_csr, duration=5.0, warmup=1.0)
        print('new impl:', t)
        print()

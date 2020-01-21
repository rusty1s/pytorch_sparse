import time
import os.path as osp
import itertools

import argparse
import wget
import torch
from scipy.io import loadmat

from torch_sparse import spmm_cpu
from torch_sparse.tensor import SparseTensor

short_rows = [
    ('DIMACS10', 'citationCiteseer'),
    ('SNAP', 'web-Stanford'),
]
long_rows = [
    ('Janna', 'StocF-1465'),
    ('GHS_psdef', 'ldoor'),
]


def download(dataset):
    url = 'https://sparse.tamu.edu/mat/{}/{}.mat'
    for group, name in itertools.chain(long_rows, short_rows):
        if not osp.exists(f'{name}.mat'):
            print(f'Downloading {group}/{name}:')
            wget.download(url.format(group, name))
            print('')


def bold(text, flag=True):
    return f'\033[1m{text}\033[0m' if flag else text


@torch.no_grad()
def correctness(dataset):
    pass


def time_func(func, x):
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()

        if not args.with_backward:
            with torch.no_grad():
                for _ in range(iters):
                    func(x)
        else:
            x = x.requires_grad_()
            for _ in range(iters):
                out = func(x)
                out = out[0] if isinstance(out, tuple) else out
                torch.autograd.grad(out, x, out, only_inputs=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter() - t
    except RuntimeError as e:
        if 'out of memory' not in str(e):
            raise RuntimeError(e)
        torch.cuda.empty_cache()
        return float('inf')


def timing(dataset):
    group, name = dataset
    mat_scipy = loadmat(f'{name}.mat')['Problem'][0][0][2].tocsr()
    row = torch.from_numpy(mat_scipy.tocoo().row).to(args.device, torch.long)
    col = torch.from_numpy(mat_scipy.tocoo().col).to(args.device, torch.long)
    index = torch.stack([row, col], dim=0)
    mat_own = SparseTensor(index, sparse_size=mat_scipy.shape)
    rowptr, col, value = mat_own.csr()
    mat_pytorch = mat_own.to_torch_sparse_coo_tensor().coalesce()

    def spmm_scipy(x):
        return mat_scipy @ x

    def spmm_pytorch(x):
        return mat_pytorch @ x

    def spmm_own(x):
        return spmm_cpu.spmm(rowptr, col, value, x, 'sum')

    t1, t2, t3 = [], [], []

    for size in sizes:
        try:
            x = torch.randn((mat_own.size(1), size), device=args.device)

            t1 += [time_func(spmm_scipy, x)]
            t2 += [time_func(spmm_pytorch, x)]
            t3 += [time_func(spmm_own, x)]

            del x

        except RuntimeError as e:
            if 'out of memory' not in str(e):
                raise RuntimeError(e)
            torch.cuda.empty_cache()
            for t in (t1, t2, t3):
                t.append(float('inf'))

    ts = torch.tensor([t1, t2, t3])
    winner = torch.zeros_like(ts, dtype=torch.bool)
    winner[ts.argmin(dim=0), torch.arange(len(sizes))] = 1
    winner = winner.tolist()

    name = f'{group}/{name}'
    print(f'{bold(name)} (avg row length: {mat_own.avg_row_length():.2f}):')
    print('\t'.join(['            '] + [f'{size:>5}' for size in sizes]))
    print('\t'.join([bold('SPMM SciPy  ')] +
                    [bold(f'{t:.5f}', f) for t, f in zip(t1, winner[0])]))
    print('\t'.join([bold('SPMM PyTorch')] +
                    [bold(f'{t:.5f}', f) for t, f in zip(t2, winner[1])]))
    print('\t'.join([bold('SPMM Own    ')] +
                    [bold(f'{t:.5f}', f) for t, f in zip(t3, winner[2])]))
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_backward', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    iters = 1 if args.device == 'cpu' else 20
    sizes = [1, 16, 32, 64, 128, 256, 512]
    sizes = sizes[:4] if args.device == 'cpu' else sizes

    for _ in range(10):  # Warmup.
        torch.randn(100, 100, device=args.device).sum()
    for dataset in itertools.chain(short_rows, long_rows):
        download(dataset)
        correctness(dataset)
        timing(dataset)

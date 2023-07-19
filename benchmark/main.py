import argparse
import itertools
import os.path as osp
import time

import torch
import wget
from scipy.io import loadmat
from torch_scatter import scatter_add

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
    group, name = dataset
    mat_scipy = loadmat(f'{name}.mat')['Problem'][0][0][2].tocsr()
    row = torch.from_numpy(mat_scipy.tocoo().row).to(args.device, torch.long)
    col = torch.from_numpy(mat_scipy.tocoo().col).to(args.device, torch.long)
    mat = SparseTensor(row=row, col=col, sparse_sizes=mat_scipy.shape)
    mat.fill_cache_()
    mat_pytorch = mat.to_torch_sparse_coo_tensor().coalesce()

    for size in sizes:
        try:
            x = torch.randn((mat.size(1), size), device=args.device)

            out1 = mat @ x
            out2 = mat_pytorch @ x

            assert torch.allclose(out1, out2, atol=1e-4)

        except RuntimeError as e:
            if 'out of memory' not in str(e):
                raise RuntimeError(e)
            torch.cuda.empty_cache()


def time_func(func, x):
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            import torch.mps
            torch.mps.synchronize()
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
        elif torch.backends.mps.is_available():
            import torch.mps
            torch.mps.synchronize()
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
    mat = SparseTensor(row=row, col=col, sparse_sizes=mat_scipy.shape)
    mat.fill_cache_()
    mat_pytorch = mat.to_torch_sparse_coo_tensor().coalesce()
    mat_scipy = mat.to_scipy(layout='csr')

    def scatter(x):
        return scatter_add(x[col], row, dim=0, dim_size=mat_scipy.shape[0])

    def spmm_scipy(x):
        if x.is_cuda:
            raise RuntimeError('out of memory')
        return mat_scipy @ x

    def spmm_pytorch(x):
        return mat_pytorch @ x

    def spmm(x):
        return mat @ x

    t1, t2, t3, t4 = [], [], [], []

    for size in sizes:
        try:
            x = torch.randn((mat.size(1), size), device=args.device)

            t1 += [time_func(scatter, x)]
            t2 += [time_func(spmm_scipy, x)]
            t3 += [time_func(spmm_pytorch, x)]
            t4 += [time_func(spmm, x)]

            del x

        except RuntimeError as e:
            if 'out of memory' not in str(e):
                raise RuntimeError(e)
            torch.cuda.empty_cache()
            for t in (t1, t2, t3, t4):
                t.append(float('inf'))

    ts = torch.tensor([t1, t2, t3, t4])
    winner = torch.zeros_like(ts, dtype=torch.bool)
    winner[ts.argmin(dim=0), torch.arange(len(sizes))] = 1
    winner = winner.tolist()

    name = f'{group}/{name}'
    print(f'{bold(name)} (avg row length: {mat.avg_row_length():.2f}):')
    print('\t'.join(['            '] + [f'{size:>5}' for size in sizes]))
    print('\t'.join([bold('Scatter     ')] +
                    [bold(f'{t:.5f}', f) for t, f in zip(t1, winner[0])]))
    print('\t'.join([bold('SPMM SciPy  ')] +
                    [bold(f'{t:.5f}', f) for t, f in zip(t2, winner[1])]))
    print('\t'.join([bold('SPMM PyTorch')] +
                    [bold(f'{t:.5f}', f) for t, f in zip(t3, winner[2])]))
    print('\t'.join([bold('SPMM Own    ')] +
                    [bold(f'{t:.5f}', f) for t, f in zip(t4, winner[3])]))
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

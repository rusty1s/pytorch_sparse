import importlib
import os.path as osp

import torch

__version__ = '0.7.0'

for library in [
        '_version', '_convert', '_diag', '_spmm', '_spspmm', '_metis', '_rw',
        '_saint', '_sample', '_ego_sample', '_hgt_sample', '_neighbor_sample',
        '_relabel'
]:
    cuda_spec = importlib.machinery.PathFinder().find_spec(
        f'{library}_cuda', [osp.dirname(__file__)])
    cpu_spec = importlib.machinery.PathFinder().find_spec(
        f'{library}_cpu', [osp.dirname(__file__)])
    spec = cuda_spec or cpu_spec
    if spec is not None:
        torch.ops.load_library(spec.origin)
    else:  # pragma: no cover
        raise ImportError(f"Could not find module '{library}_cpu' in "
                          f"{osp.dirname(__file__)}")

cuda_version = torch.ops.torch_sparse.cuda_version()
if torch.version.cuda is not None and cuda_version != -1:  # pragma: no cover
    if cuda_version < 10000:
        major, minor = int(str(cuda_version)[0]), int(str(cuda_version)[2])
    else:
        major, minor = int(str(cuda_version)[0:2]), int(str(cuda_version)[3])
    t_major, t_minor = [int(x) for x in torch.version.cuda.split('.')]

    if t_major != major:
        raise RuntimeError(
            f'Detected that PyTorch and torch_sparse were compiled with '
            f'different CUDA versions. PyTorch has CUDA version '
            f'{t_major}.{t_minor} and torch_sparse has CUDA version '
            f'{major}.{minor}. Please reinstall the torch_sparse that '
            f'matches your PyTorch install.')

from .add import add, add_, add_nnz, add_nnz_  # noqa
from .bandwidth import reverse_cuthill_mckee  # noqa
from .cat import cat  # noqa
from .coalesce import coalesce  # noqa
from .convert import to_scipy  # noqa
from .convert import from_scipy, from_torch_sparse, to_torch_sparse
from .diag import fill_diag, get_diag, remove_diag, set_diag  # noqa
from .eye import eye  # noqa
from .index_select import index_select, index_select_nnz  # noqa
from .masked_select import masked_select, masked_select_nnz  # noqa
from .matmul import matmul  # noqa
from .metis import partition  # noqa
from .mul import mul, mul_, mul_nnz, mul_nnz_  # noqa
from .narrow import __narrow_diag__, narrow  # noqa
from .padding import padded_index, padded_index_select  # noqa
from .permute import permute  # noqa
from .reduce import max, mean, min, sum  # noqa
from .rw import random_walk  # noqa
from .saint import saint_subgraph  # noqa
from .sample import sample, sample_adj  # noqa
from .select import select  # noqa
from .spadd import spadd  # noqa
from .spmm import spmm  # noqa
from .spspmm import spspmm  # noqa
from .storage import SparseStorage  # noqa
from .tensor import SparseTensor  # noqa
from .transpose import t  # noqa
from .transpose import transpose  # noqa

__all__ = [
    'SparseStorage',
    'SparseTensor',
    't',
    'narrow',
    '__narrow_diag__',
    'select',
    'index_select',
    'index_select_nnz',
    'masked_select',
    'masked_select_nnz',
    'permute',
    'remove_diag',
    'set_diag',
    'fill_diag',
    'get_diag',
    'add',
    'add_',
    'add_nnz',
    'add_nnz_',
    'mul',
    'mul_',
    'mul_nnz',
    'mul_nnz_',
    'sum',
    'mean',
    'min',
    'max',
    'matmul',
    'cat',
    'random_walk',
    'partition',
    'reverse_cuthill_mckee',
    'saint_subgraph',
    'padded_index',
    'padded_index_select',
    'to_torch_sparse',
    'from_torch_sparse',
    'to_scipy',
    'from_scipy',
    'coalesce',
    'transpose',
    'eye',
    'spmm',
    'spspmm',
    'spadd',
    '__version__',
]

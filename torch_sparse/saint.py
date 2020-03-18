from typing import Tuple

import torch
import numpy as np
from torch_scatter import scatter_add
from torch_sparse.tensor import SparseTensor


def sample_node(src: SparseTensor,
                num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    row, col, _ = src.coo()

    inv_in_deg = src.storage.colcount().to(torch.float).pow_(-1)
    inv_in_deg[inv_in_deg == float('inf')] = 0

    prob = inv_in_deg[col]
    prob.mul_(prob)

    prob = scatter_add(prob, row, dim=0, dim_size=src.size(0))
    prob.div_(prob.sum())

    node_idx = prob.multinomial(num_nodes, replacement=True).unique()

    return src.permute(node_idx), node_idx


def sample_edge(src: SparseTensor,
                num_edges: int) -> Tuple[torch.Tensor, torch.Tensor]:

    row, col, _ = src.coo()

    inv_out_deg = src.storage.rowcount().to(torch.float).pow_(-1)
    inv_out_deg[inv_out_deg == float('inf')] = 0
    inv_in_deg = src.storage.colcount().to(torch.float).pow_(-1)
    inv_in_deg[inv_in_deg == float('inf')] = 0

    prob = inv_out_deg[row] + inv_in_deg[col]
    prob.div_(prob.sum())

    edge_idx = prob.multinomial(num_edges, replacement=True)
    node_idx = col[edge_idx].unique()

    return src.permute(node_idx), node_idx


def sample_rw(src: SparseTensor, num_root_nodes: int,
              walk_length: int) -> Tuple[torch.Tensor, torch.Tensor]:

    start = np.random.choice(src.size(0), size=num_root_nodes, replace=False)
    start = torch.from_numpy(start).to(src.device())

    # get random walks of length `walk_length`:
    # => `rw.size(1) == walk_length + 1

    return None, None


SparseTensor.sample_node = sample_node
SparseTensor.sample_edge = sample_edge
SparseTensor.sample_rw = sample_rw

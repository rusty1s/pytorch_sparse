import torch


def eye(m, dtype=None, device=None):
    """Returns a sparse matrix with ones on the diagonal and zeros elsewhere.

    Args:
        m (int): The first dimension of corresponding dense matrix.
        dtype (`torch.dtype`, optional): The desired data type of returned
            value vector. (default is set by `torch.set_default_tensor_type()`)
        device (`torch.device`, optional): The desired device of returned
            tensors. (default is set by `torch.set_default_tensor_type()`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """

    row = torch.arange(m, dtype=torch.long, device=device)
    index = torch.stack([row, row], dim=0)

    value = torch.ones(m, dtype=dtype, device=device)

    return index, value

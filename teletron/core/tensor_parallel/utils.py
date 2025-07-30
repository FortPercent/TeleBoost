import torch
from typing import List

def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""

    def ensure_divisibility(numerator, denominator):
        """Ensure that numerator is divisible by the denominator."""
        assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)

    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def split_tensor_along_last_dim(
    tensor: torch.Tensor, num_partitions: int, contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """ Split a tensor along its last dimension."""
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list
"""Diffusion-aware Ulysses sequence parallel helpers.

Pre-X3 lived in the in-tree verl/utils/ulysses.py fork. After X3 dropped that fork,
upstream verl 0.4.0's ulysses.py only ships the LM-shape helpers (`gather_seq_scatter_heads`,
`slice_input_tensor`, etc.). The Wan diffusion path needs:

  - module-level state for the current sequence pad/target sizes (set during input
    slicing inside the patched block.forward, read during head-gather inside
    Head.forward — same iteration of the same model);
  - a `DiffusionGather` autograd Function that's like upstream `Gather` but keeps
    the local batch-dim shape so split/cat round-trips work for image latents;
  - `split_forward_gather_backward` / `gather_forward_split_backward` round-trippers
    used by the Wan block.forward / Head.forward monkey-patches in
    `teleboost.models.transformers.wan`.

Importing from `verl.utils.ulysses` for the upstream-supplied helpers
(`get_ulysses_sequence_parallel_group`, `_pad_tensor`, `_unpad_tensor`,
`all_gather_tensor`).
"""
from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.distributed as dist
from torch import Tensor

from verl.utils.ulysses import (
    _pad_tensor,
    _unpad_tensor,
    all_gather_tensor,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_rank,
)


# ----- module-level state ----------------------------------------------------
# Target / pad sizes are set by the input-slicing wrapper at the start of each
# transformer block.forward and read by the head-gather wrapper at the end of
# Head.forward (same iteration, same model). Storing them as module-level
# globals matches the pre-X3 fork's contract; callers don't pass them through.
_TARGET_SIZE: Optional[int] = None
_PAD_SIZE: Optional[int] = None


def set_target_len(target_size: int) -> None:
    global _TARGET_SIZE
    _TARGET_SIZE = target_size


def get_target_len() -> Optional[int]:
    return _TARGET_SIZE


def set_pad_size(pad_size: int) -> None:
    global _PAD_SIZE
    _PAD_SIZE = pad_size


def get_pad_size() -> Optional[int]:
    return _PAD_SIZE


# ----- autograd Functions ----------------------------------------------------
class DiffusionGather(torch.autograd.Function):
    """All-gather along `gather_dim`, splitting the batch dim back out on the way in.

    Variant of upstream `verl.utils.ulysses.Gather` for image-latent shape conventions:
    the gathered output has the same leading batch size as the local input
    (concat happens along `gather_dim`, not along dim=0).
    """

    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_tensor: Tensor,
        gather_dim: int,
        grad_scaler: bool = True,
        async_op: bool = False,
    ) -> Tensor:
        ctx.group = group
        ctx.gather_dim = gather_dim
        ctx.grad_scaler = grad_scaler

        ctx.sp_world_size = dist.get_world_size(group=group)
        ctx.sp_rank = dist.get_rank(group=group)

        local_shape = list(local_tensor.size())
        split_size = local_shape[0]
        ctx.part_size = local_shape[gather_dim]

        output = all_gather_tensor(local_tensor, group, async_op)
        return torch.cat(output.split(split_size, dim=0), dim=gather_dim)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor):
        return (
            None,
            grad_output.split(ctx.part_size, dim=ctx.gather_dim)[ctx.sp_rank].contiguous(),
            None,
            None,
            None,
            None,
        )


class _GatherForwardSplitBackward(torch.autograd.Function):
    """Forward: all-gather across `process_group` along `dim`. Backward: split + grad scale."""

    @staticmethod
    def forward(ctx, input_, process_group, dim, gather_sizes, grad_scale="up"):
        ctx.mode = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        ctx.gather_sizes = gather_sizes
        return _gather(input_, process_group, dim, gather_sizes)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.mode)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.mode)
        return _split(grad_output, ctx.mode, ctx.dim, ctx.gather_sizes), None, None, None, None


class _SplitForwardGatherBackward(torch.autograd.Function):
    """Forward: split along `dim`, keep this rank's chunk. Backward: gather + grad scale."""

    @staticmethod
    def forward(ctx, input_, process_group, dim, split_sizes, grad_scale):
        ctx.mode = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        ctx.split_sizes = split_sizes
        return _split(input_, process_group, dim, split_sizes)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.mode)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.mode)
        return _gather(grad_output, ctx.mode, ctx.dim, ctx.split_sizes), None, None, None, None


# ----- low-level split/gather (support unaligned shapes) ---------------------
def _split(
    input_: torch.Tensor,
    pg: dist.ProcessGroup,
    dim: int = -1,
    split_sizes: Optional[List[int]] = None,
) -> torch.Tensor:
    assert split_sizes is None or isinstance(split_sizes, list)

    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return input_

    if split_sizes is None:
        dim_size = input_.size(dim)
        base_size = dim_size // world_size
        remainder = dim_size % world_size
        # Distribute remainder to first `remainder` ranks (matches upstream LM split).
        split_sizes = [base_size + 1 if i < remainder else base_size for i in range(world_size)]

    tensor_list = torch.split(input_, split_sizes, dim=dim)
    rank = dist.get_rank(pg)
    return tensor_list[rank].contiguous()


def _gather(
    input_: torch.Tensor,
    pg: dist.ProcessGroup,
    dim: int = -1,
    gather_sizes: Optional[List[int]] = None,
) -> torch.Tensor:
    assert gather_sizes is None or isinstance(gather_sizes, list)

    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return input_

    input_ = input_.contiguous()

    if gather_sizes:
        tensor_shape_base = input_.size()
        tensor_list = []
        for i in range(world_size):
            tensor_shape = list(tensor_shape_base)
            tensor_shape[dim] = gather_sizes[i]
            tensor_list.append(torch.empty(tensor_shape, dtype=input_.dtype, device=input_.device))
    else:
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]

    assert input_.device.type == "cuda"
    dist.all_gather(tensor_list, input_, group=pg)
    return torch.cat(tensor_list, dim=dim).contiguous()


# ----- public round-trip helpers --------------------------------------------
def split_forward_gather_backward(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    dim: int,
    split_sizes: Optional[List[int]] = None,
    grad_scale: str = "down",
) -> torch.Tensor:
    return _SplitForwardGatherBackward.apply(input_, process_group, dim, split_sizes, grad_scale)


def gather_forward_split_backward(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    dim: int,
    gather_sizes: Optional[List[int]] = None,
    grad_scale: str = "up",
) -> torch.Tensor:
    return _GatherForwardSplitBackward.apply(input_, process_group, dim, gather_sizes, grad_scale)


# ----- diffusion-shape gather/slice -----------------------------------------
def diffusion_gather_outpus_and_unpad(
    x: Tensor,
    gather_dim: int,
    unpad_dim: Optional[int] = None,
    padding_size: int = 0,
    grad_scaler: bool = True,
    group: Optional[dist.ProcessGroup] = None,
) -> Tensor:
    """All-gather along `gather_dim`, then strip `padding_size` rows from `unpad_dim`.

    Used by the Wan Head.forward monkey-patch to undo the pre-block input slicing
    (shape now back to full sequence), then drop the padding the slicer added.
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if group is None:
        return x
    x = DiffusionGather.apply(group, x, gather_dim, grad_scaler)
    if unpad_dim is not None:
        assert isinstance(padding_size, int), "padding_size must be int"
        if padding_size == 0:
            return x
        x = _unpad_tensor(x, unpad_dim, padding_size)
    return x


def diffusion_slice_input_tensor_pad(
    x: Tensor,
    dim: int,
    padding: bool = False,
    grad_scaler: bool = True,
) -> Tensor:
    """Pad `x` on `dim` to be SP-divisible, then keep this rank's slice.

    Records the pre-slicing length and pad size into module-level globals so the
    matching head-gather wrapper can reverse the operation.
    """
    group = get_ulysses_sequence_parallel_group()
    sp_world_size = dist.get_world_size(group)
    dim_size = x.size(dim)
    padding_size = (sp_world_size - dim_size % sp_world_size) % sp_world_size
    set_target_len(dim_size)
    set_pad_size(padding_size)
    if padding and padding_size > 0:
        x = _pad_tensor(x, dim, padding_size)
    return split_forward_gather_backward(x, group, dim=dim, grad_scale="none")

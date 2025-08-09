# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utilities for DeepSpeed Ulysses Sequence Parallelism.
DeepSpeed Ulysses Paper: https://arxiv.org/abs/2309.14509
Inspired from: https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/sequence/layer.py
"""

from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from typing import Optional, List

_ULYSSES_SEQUENCE_PARALLEL_GROUP = None


def set_ulysses_sequence_parallel_group(group: dist.ProcessGroup):
    """
    Set ulysses sequence parallel process group.
    """
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    _ULYSSES_SEQUENCE_PARALLEL_GROUP = group


def get_ulysses_sequence_parallel_group() -> Optional[dist.ProcessGroup]:
    """
    Get ulysses sequence parallel process group.
    """
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    return _ULYSSES_SEQUENCE_PARALLEL_GROUP


def get_ulysses_sequence_parallel_world_size(group: ProcessGroup = None) -> int:
    """
    Get ulysses sequence parallel world size.
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    return dist.get_world_size(group) if group else 1


def get_ulysses_sequence_parallel_rank(group: ProcessGroup = None) -> int:
    """
    Get ulysses sequence parallel rank.
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    return dist.get_rank(group) if group else 0


def gather_seq_scatter_heads(
    x: Tensor,
    seq_dim: int,
    head_dim: int,
    unpadded_dim_size: int = 0,
    group: ProcessGroup = None,
) -> Tensor:
    """
    A func to sync embedding input with alltoall in sequence parallel
    gather sequence dimension and scatter head dim:
    e.g. seq_dim: 1, head_dim: 2
    [bsz, seq/n, h, ...] -> [bsz, seq, h/n, ...]
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if not group:
        return x
    sp_world = get_ulysses_sequence_parallel_world_size(group)

    x = SeqAllToAll.apply(group, x, head_dim, seq_dim)

    if unpadded_dim_size and unpadded_dim_size % sp_world != 0:
        padding_size = x.size(seq_dim) - unpadded_dim_size
        x = _unpad_tensor(x, seq_dim, padding_size)
    return x


def gather_heads_scatter_seq(x: Tensor, head_dim: int, seq_dim: int, group: ProcessGroup = None) -> Tensor:
    """
    A func to sync attention result with alltoall in sequence parallel
    gather head dimension and scatter seq dim:
    e.g. seq_dim: 1, head_dim: 2
    [bsz, seq, h/n, ...] -> [bsz, seq/n, h, ...]
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if not group:
        return x
    dim_size = x.size(seq_dim)
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    if dim_size % sp_world != 0:
        padding_size = sp_world - (dim_size % sp_world)
        x = _pad_tensor(x, seq_dim, padding_size)
    return SeqAllToAll.apply(group, x, seq_dim, head_dim, False)

def _pad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    shape = list(x.shape)
    shape[dim] = padding_size
    pad = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)


def _unpad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(0, -padding_size)
    return x[slc]


def slice_input_tensor(x: Tensor, dim: int, padding: bool = True, group: ProcessGroup = None) -> Tensor:
    group = get_ulysses_sequence_parallel_group() if group is None else group
    sp_world_size = dist.get_world_size(group)
    sp_rank = get_ulysses_sequence_parallel_rank()
    dim_size = x.size(dim)
    # pad before slice
    if padding and dim_size % sp_world_size:
        padding_size = sp_world_size - (dim_size % sp_world_size)
        x = _pad_tensor(x, dim, padding_size)
    # slice the input tensor
    parts = x.size(dim) // sp_world_size
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(sp_rank * parts, (sp_rank + 1) * parts)
    return x[slc].contiguous()


def all_to_all_tensor(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    group = get_ulysses_sequence_parallel_group() if group is None else group
    seq_world_size = dist.get_world_size(group)
    input_list = [t.contiguous() for t in torch.tensor_split(local_input, seq_world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
    comm = dist.all_to_all(output_list, input_list, group=group, async_op=async_op)
    if async_op:

        def wait():
            comm.wait()
            return torch.cat(output_list, dim=gather_dim).contiguous()

        return wait
    return torch.cat(output_list, dim=gather_dim).contiguous()


def all_gather_tensor(local_tensor: Tensor, group: Optional[dist.ProcessGroup] = None, async_op: bool = False):
    group = get_ulysses_sequence_parallel_group() if group is None else group
    sp_world_size = dist.get_world_size(group=group)
    output_shape = list(local_tensor.shape)
    output_shape[0] = output_shape[0] * sp_world_size
    output = torch.empty(output_shape, dtype=local_tensor.dtype, device=local_tensor.device)
    dist.all_gather_into_tensor(output, local_tensor, group=group, async_op=async_op)
    return output


class SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_input: Tensor,
        scatter_dim: int,
        gather_dim: int,
        async_op: bool = False,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.async_op = async_op
        return all_to_all_tensor(local_input, scatter_dim, gather_dim, group, async_op)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        input_t = torch.cat(grad_output[1:], dim=ctx.gather_dim).contiguous() if ctx.async_op else grad_output[0]
        return (
            None,
            all_to_all_tensor(input_t, ctx.gather_dim, ctx.scatter_dim, ctx.group, False),
            None,
            None,
            None,
            None,
        )


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_tensor: Tensor,
        gather_dim: int,
        grad_scaler: bool = True,
        async_op=False,
    ) -> Tensor:
        ctx.group = group
        ctx.gather_dim = gather_dim
        ctx.grad_scaler = grad_scaler
        ctx.async_op = async_op

        sp_world_size = dist.get_world_size(group=group)
        ctx.sp_world_size = sp_world_size

        sp_rank = dist.get_rank(group=group)
        ctx.sp_rank = sp_rank

        local_shape = list(local_tensor.size())
        split_size = local_shape[0]
        part_size = local_shape[gather_dim]  # store original size
        ctx.part_size = part_size

        output = all_gather_tensor(local_tensor, group, async_op)
        return torch.cat(output.split(split_size, dim=0), dim=gather_dim)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Any:
        if ctx.grad_scaler:
            grad_output = grad_output * ctx.sp_world_size
        return (
            None,
            grad_output.split(ctx.part_size, dim=ctx.gather_dim)[ctx.sp_rank].contiguous(),
            None,
            None,
            None,
            None,
        )

class DiffusionGather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_tensor: Tensor,
        gather_dim: int,
        grad_scaler: bool = True,
        async_op=False,
    ) -> Tensor:
        ctx.group = group
        ctx.gather_dim = gather_dim
        ctx.grad_scaler = grad_scaler
        ctx.async_op = async_op

        sp_world_size = dist.get_world_size(group=group)
        ctx.sp_world_size = sp_world_size

        sp_rank = dist.get_rank(group=group)
        ctx.sp_rank = sp_rank

        local_shape = list(local_tensor.size())
        split_size = local_shape[0]
        part_size = local_shape[gather_dim]  # store original size
        ctx.part_size = part_size

        output = all_gather_tensor(local_tensor, group, async_op)
        return torch.cat(output.split(split_size, dim=0), dim=gather_dim)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Any:
        if ctx.grad_scaler:
            grad_output = grad_output
        return (
            None,
            grad_output.split(ctx.part_size, dim=ctx.gather_dim)[ctx.sp_rank].contiguous(),
            None,
            None,
            None,
            None,
        )

_TARGET_SIZE = None

def set_target_len(target_size):
    global _TARGET_SIZE
    _TARGET_SIZE=target_size

def get_target_len():
    return _TARGET_SIZE

_PAD_SIZE = None

def set_pad_size(pad_size):
    global _PAD_SIZE
    _PAD_SIZE=pad_size

def get_pad_size():
    return _PAD_SIZE

def register_cp_grad_reduce_hook(model):
    def cp_grad_reduce(grad):
        with torch.no_grad():
            # 使用 all_reduce 直接对梯度进行求和
            print(f"Rank {torch.distributed.get_rank()}: before all_reduce, grad = {grad.type(),grad.float().norm().item()}")
            # orig_dtype = grad.dtype
            # # 转成 fp32 做通信（可加 contiguous 以稳妥）
            # buf = grad.detach().to(torch.float32)
            torch.distributed.all_reduce(
                grad, op=torch.distributed.ReduceOp.SUM, group=get_ulysses_sequence_parallel_group()
            )
            # 写回到原 grad（不改计算图）
            # grad.copy_(buf.to(orig_dtype))
            print(f"Rank {torch.distributed.get_rank()}: after all_reduce, grad = {grad.float().norm().item()}")
            return grad

    for name, param in model.named_parameters():
        # if ("patch_embedding" in name or
        #     "time" in name or
        #     "head" in name or
        #     "modulation" in name):
        if "blocks" in name:
            print("in register cp",name)
            param.register_hook(cp_grad_reduce)

       
        # param.register_hook(cp_grad_reduce)

def gather_outpus_and_unpad(
    x: Tensor,
    gather_dim: int,
    unpad_dim: int = None,
    padding_size: int = 0,
    grad_scaler: bool = True,
    group: Optional[dist.ProcessGroup] = None,
):
    """
    Gather a tensor across a process group and optionally unpad its padded elements.

    Args:
        x (Tensor): Input tensor to gather.
        gather_dim (int): Dimension along which to gather across ranks.
        unpad_dim (int, optional): Dimension from which to remove padding. If None, no unpadding.
        padding_size (int): Number of padding elements to remove on `unpad_dim`. Defaults to 0.
        grad_scaler (bool): Whether to apply gradient scaling during gather. Defaults to True.
        group (ProcessGroup, optional): Process group for gathering. If None, uses
            `get_ulysses_sequence_parallel_group()`. If still None, returns `x` unchanged.

    Returns:
        Tensor: The gathered tensor, with padding removed if requested.
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if group is None:
        return x
    x = Gather.apply(group, x, gather_dim, grad_scaler)
    if unpad_dim is not None:
        assert isinstance(padding_size, int), "padding size is not given or is not an integer"
        if padding_size == 0:
            return x
        x = _unpad_tensor(x, unpad_dim, padding_size)
    return x

def diffusion_gather_outpus_and_unpad(
    x: Tensor,
    gather_dim: int,
    unpad_dim: int = None,
    padding_size: int = 0,
    grad_scaler: bool = True,
    group: Optional[dist.ProcessGroup] = None,
):
    """
    Gather a tensor across a process group and optionally unpad its padded elements.

    Args:
        x (Tensor): Input tensor to gather.
        gather_dim (int): Dimension along which to gather across ranks.
        unpad_dim (int, optional): Dimension from which to remove padding. If None, no unpadding.
        padding_size (int): Number of padding elements to remove on `unpad_dim`. Defaults to 0.
        grad_scaler (bool): Whether to apply gradient scaling during gather. Defaults to True.
        group (ProcessGroup, optional): Process group for gathering. If None, uses
            `get_ulysses_sequence_parallel_group()`. If still None, returns `x` unchanged.

    Returns:
        Tensor: The gathered tensor, with padding removed if requested.
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if group is None:
        return x
    x = DiffusionGather.apply(group, x, gather_dim, grad_scaler)
    if unpad_dim is not None:
        assert isinstance(padding_size, int), "padding size is not given or is not an integer"
        if padding_size == 0:
            return x
        x = _unpad_tensor(x, unpad_dim, padding_size)
    return x

def ulysses_pad(input_ids_rmpad: torch.Tensor, position_ids_rmpad: Optional[torch.Tensor] = None, sp_size: int = 1):
    if position_ids_rmpad is not None:
        assert position_ids_rmpad.size(-2) == 1
        assert input_ids_rmpad.size(-1) == position_ids_rmpad.size(-1)
    if sp_size <= 1:
        return input_ids_rmpad, position_ids_rmpad, 0
    _, total_seq_len = input_ids_rmpad.shape
    pad_size = (sp_size - total_seq_len % sp_size) % sp_size
    if pad_size > 0:
        input_ids_rmpad = torch.nn.functional.pad(input_ids_rmpad, (0, pad_size), value=0)
        if position_ids_rmpad is not None:
            pad_pos_ids = torch.arange(pad_size, device=position_ids_rmpad.device).unsqueeze(0)
            if position_ids_rmpad.dim() == 3:
                pad_pos_ids = pad_pos_ids.unsqueeze(0).repeat(3, 1, 1)
            position_ids_rmpad = torch.cat((position_ids_rmpad, pad_pos_ids), dim=-1)
    return input_ids_rmpad, position_ids_rmpad, pad_size


def ulysses_pad_and_slice_inputs(input_ids_rmpad: torch.Tensor, position_ids_rmpad: Optional[torch.Tensor] = None, sp_size: int = 1):
    """
    Pad and slice input_ids to be divisible by sp_size
    Pad position_ids to be divisible by sp_size.

    Note both input_ids_rmpad and position_ids_rmpad will be padded and sliced.

    The is the utility of pre-forward for ulysses sequence parallelism

    Args:
        input_ids_rmpad: shape of [bsz, seqlen]
        position_ids_rmpad: shape of [bsz, seqlen], where bsz must be 1
        sp_size (int): ulysses sequence parallelism size

    Returns:
        torch.Tensor: padded and sliced input_ids
        torch.Tensor: padded and sliced position_ids
        int: pad size
    """
    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(input_ids_rmpad, position_ids_rmpad, sp_size)
    input_ids_rmpad = slice_input_tensor(input_ids_rmpad, dim=1, padding=False)
    if position_ids_rmpad is not None:
        position_ids_rmpad = slice_input_tensor(position_ids_rmpad, dim=1, padding=False)
    return input_ids_rmpad, position_ids_rmpad, pad_size


def validate_ulysses_config(num_heads, ulysses_sequence_size):
    if ulysses_sequence_size > 1:
        assert num_heads % ulysses_sequence_size == 0, f"num_heads ({num_heads}) must be divisible by ulysses sequence size({ulysses_sequence_size})"

def diffusion_slice_input_tensor_pad(x: Tensor, dim: int, padding: bool = False, grad_scaler: bool = True):
    # Step 1: Padding to make divisible
    group = get_ulysses_sequence_parallel_group()
    sp_world_size = dist.get_world_size(group)
    sp_rank = get_ulysses_sequence_parallel_rank()
    dim_size = x.size(dim)
    # pad before slice
    #TODO neet custom utils
    padding_size = (sp_world_size - dim_size % sp_world_size) % sp_world_size
    set_target_len(dim_size)
    set_pad_size(padding_size)
    if padding and padding_size > 0:
        x = _pad_tensor(x, dim, padding_size)

    return split_forward_gather_backward(x,group,dim=dim, grad_scale="none")

def split_forward_gather_backward(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    dim: int,
    split_sizes: Optional[List[int]] = None,
    grad_scale: str = "down"

) -> torch.Tensor:
    """
    Splits the input tensor and keeps only the corresponding chunk for the current rank.
    During the backward pass, it gathers the gradients and scales them according to the gradient scaling mode.
    This function supports both aligned and unaligned data.
    Args:
        input_ (torch.Tensor): The input tensor to be processed.
        process_group (dist.ProcessGroup): The process group to perform the operation within.
        dim (int): The dimension along which to split the tensor.
        split_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be split.
            If not provided, the tensor will be split equally among the processes. Defaults to None.
        grad_scale (str, optional): Gradient scaling mode. Can be "up", "down", or None. Defaults to "down".

    Returns:
        torch.Tensor: The resulting tensor after splitting and keeping only the corresponding chunk.
    """
    
    return _SplitForwardGatherBackward.apply(input_, process_group, dim, split_sizes, grad_scale)


def gather_forward_split_backward(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    dim: int,
    gather_sizes: Optional[List[int]] = None,
    grad_scale: str = "up"
) -> torch.Tensor:
    """
    Gathers the input tensor from all processes in the model parallel region and concatenates them along the specified
    dimension. During the backward pass, it splits the gradients and scales them according to the gradient scaling mode.
    This function handles both aligned and unaligned data during the gather and scatter operations.
    Args:
        input_ (torch.Tensor): The input tensor to be processed.
        process_group (dist.ProcessGroup): The process group to perform the operation within.
        dim (int): The dimension along which to concatenate the gathered tensors.
        gather_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be gathered.
            If not provided, it is assumed that all tensors have the same shape as the input tensor. Defaults to None.
        grad_scale (str, optional): Gradient scaling mode. Can be "up", "down", or None. Defaults to "up".

    Returns:
        torch.Tensor: The resulting tensor after gathering and concatenating.
    """
    return _GatherForwardSplitBackward.apply(input_, process_group, dim, gather_sizes, grad_scale)

class _GatherForwardSplitBackward(torch.autograd.Function):
    """
    Custom autograd function that gathers the input tensor from all processes in the model parallel region and
    concatenates them.
    During the backward pass, it splits the gradients and scales them according to the gradient scaling mode.

    """

    @staticmethod
    def symbolic(graph, input_, process_group, dim, gather_sizes):
        """
        Define the symbolic representation of the custom operation.
        """
        return _gather(input_, process_group, dim, gather_sizes)

    @staticmethod
    def forward(ctx, input_, process_group, dim, gather_sizes, grad_scale="up"):
        """
        Forward pass: Gathers tensors from all processes in the specified process group and concatenates them along the specified dimension.

        Args:
            input_ (torch.Tensor): The input tensor to be processed.
            process_group (dist.ProcessGroup): The process group to perform the operation within.
            dim (int): The dimension along which to concatenate the gathered tensors.
            gather_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be gathered.
            grad_scale (str, optional): Gradient scaling mode. Can be "up", "down", or None. Defaults to "up".

        Returns:
            torch.Tensor: The resulting tensor after gathering and concatenating.
        """
        ctx.mode = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale

        ctx.gather_sizes = gather_sizes
        return _gather(input_, process_group, dim, ctx.gather_sizes)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Distribute the gradients to the input tensors and scales them according to the gradient scaling mode.

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            torch.Tensor: The gradient of the input with respect to the loss.
        """
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.mode)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.mode)

        return _split(grad_output, ctx.mode, ctx.dim, ctx.gather_sizes), None, None, None, None


class _SplitForwardGatherBackward(torch.autograd.Function):
    """
    Custom autograd function that splits the input tensor and keeps only the corresponding chunk for the current rank.
    During the backward pass, it gathers the gradients and scales them according to the gradient scaling mode.

    """
    @staticmethod
    def symbolic(graph, input_, process_group, dim, split_sizes):
        return _split(input_, process_group, dim, split_sizes)

    @staticmethod
    def forward(ctx, input_, process_group, dim, split_sizes, grad_scale):
        ctx.mode = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale

        ctx.split_sizes = split_sizes
        return _split(input_, process_group, dim, ctx.split_sizes)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.mode)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.mode)
        return _gather(grad_output, ctx.mode, ctx.dim, ctx.split_sizes), None, None, None, None



def _split(
        input_: torch.Tensor,
        pg: dist.ProcessGroup,
        dim: int = -1,
        split_sizes: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Splits a tensor across the specified dimension and returns the part corresponding to the current rank,
    supporting aligned and unaligned data.

    Args:
        input_ (torch.Tensor): The input tensor to be split.
        pg (dist.ProcessGroup): The process group to perform the operation within.
        dim (int, optional): The dimension along which to split the tensor. Defaults to -1 (last dimension).
        split_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be split.
            If not provided, the tensor will be split equally among the processes, with the remainder
            distributed to the first few processes. Defaults to None.

    Returns:
        torch.Tensor: The part of the tensor corresponding to the current rank in the process group.
    """
    # Ensure split_sizes is a list if provided
    assert split_sizes is None or isinstance(split_sizes, list)

    # skip if only one rank involved
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return input_

    # Calculate split sizes if not provided
    if split_sizes is None:
        dim_size = input_.size(dim)
        base_size = dim_size // world_size
        remainder = dim_size % world_size

        # Calculate the size for each process
        split_sizes = [base_size + 1 if i < remainder else base_size for i in range(world_size)]

    tensor_list = torch.split(input_, split_sizes, dim=dim)

    # Get the part corresponding to the current rank
    rank = dist.get_rank(pg)
    output = tensor_list[rank].contiguous()

    return output


def _gather(input_: torch.Tensor,
            pg: dist.ProcessGroup,
            dim: int = -1,
            gather_sizes: Optional[List[int]] = None):
    """
    Gathers tensors from all processes in the process group and concatenates them along the specified dimension,
    supporting aligned and unaligned data.

    Args:
        input_ (torch.Tensor): The input tensor to be gathered.
        pg (dist.ProcessGroup): The process group to perform the operation within.
        dim (int, optional): The dimension along which to concatenate the gathered tensors. Defaults to -1 (last dimension).
        gather_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be gathered.
            If not provided, it is assumed that all tensors have the same shape as the input tensor. Defaults to None.

    Returns:
        torch.Tensor: The concatenated tensor after gathering from all processes in the process group.
    """
    # Ensure gather_sizes is a list if provided
    assert gather_sizes is None or isinstance(gather_sizes, list)

    # Skip if only one rank is involved
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return input_

    input_ = input_.contiguous()

    # Prepare the output list with appropriate shapes
    if gather_sizes:
        tensor_list = []
        tensor_shape_base = input_.size()
        for i in range(world_size):
            tensor_shape = list(tensor_shape_base)
            tensor_shape[dim] = gather_sizes[i]
            tensor_list.append(torch.empty(tensor_shape, dtype=input_.dtype, device=input_.device))
    else:
        tensor_list = [torch.empty_like(input_, dtype=input_.dtype, device=input_.device) for _ in range(world_size)]

    assert input_.device.type == "cuda"
    torch.distributed.all_gather(tensor_list, input_, group=pg)

    # concat
    output = torch.cat(tensor_list, dim=dim).contiguous()
    return output

def gate_with_cp_grad_reduce(x, gate, residual):
    return GateWithGradReduce.apply(x, gate, residual)


class GateWithGradReduce(torch.autograd.Function ):
    @staticmethod
    def forward(ctx, x, gate, residual):
        ctx.save_for_backward(gate, residual)
        return x + gate * residual
    
    @staticmethod
    def backward(ctx, x_grad):
        gate, residual = ctx.saved_tensors
        r_grad = x_grad * gate 
        gate_grad = torch.sum((x_grad * residual), dim=1, keepdim=True)
        torch.distributed.all_reduce(gate_grad, group=get_ulysses_sequence_parallel_group())
        return x_grad, gate_grad, r_grad


class ModulateWithCPGradReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shift, scale):
        ctx.save_for_backward(x, scale)
        return (x * (1 + scale) + shift)
    
    @staticmethod 
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        x_grad = grad_output * (1 + scale) 
        scale_grad = torch.sum((grad_output * x), dim=1, keepdim=True)
        torch.distributed.all_reduce(scale_grad, group=get_ulysses_sequence_parallel_group())
        shift_grad = torch.sum(grad_output, dim=1, keepdim=True)
        torch.distributed.all_reduce(shift_grad, group=get_ulysses_sequence_parallel_group())
        return x_grad, shift_grad, scale_grad


def modulate_with_cp_grad_reduce(x, shift, scale):
    return ModulateWithCPGradReduce.apply(x, shift, scale)
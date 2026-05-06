import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from megatron.core import mpu
from yunchang.comm.all_to_all import SeqAllToAll4D
from .mappings import split_forward_gather_backward, gather_forward_split_backward, SeqAllToAll
from .layers import GateWithGradReduce, ModulateWithCPGradReduce
from teleboost.utils import get_args

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False


class ContextParallelMixin:
    """
    Stateless CP helpers.

    pad_for_context_parallel returns ``(padded, origin_length)``; the matching
    remove_pad takes ``origin_length`` explicitly. split_input/gather_output
    follow the same contract.

    forward_attn is monkey-patched onto an attention module via
    enable_context_parallel(); the calling block must set
    ``self._cp_origin_length`` (the OUTER seq's pre-pad length, returned by
    its own split_input call) before invoking the attention chain. This lets
    forward_attn keep its (q, k, v) signature so we don't have to plumb
    origin_length through SelfAttention layers we don't own.
    """

    @staticmethod
    def cp_grad_reduce(grad):
        # Prints are kept for hang-locate diagnostics in CP backward paths.
        with torch.no_grad():
            reduced_grad = grad.contiguous()
            rank = torch.distributed.get_rank()
            print(f"[cp_grad_reduce] rank={rank} BEFORE all_reduce, shape={list(reduced_grad.shape)}")
            torch.distributed.all_reduce(reduced_grad, group=mpu.get_context_parallel_group())
            print(f"[cp_grad_reduce] rank={rank} AFTER all_reduce")
        return reduced_grad

    def enable_context_parallel(self, attn_module: nn.Module):
        attn_module.forward = self.forward_attn

    @staticmethod
    def pad_for_context_parallel(tensor, dim):
        cp_size = mpu.get_context_parallel_world_size()
        origin_length = tensor.shape[dim]
        padded_length = math.ceil(origin_length / cp_size) * cp_size
        pad_size = padded_length - origin_length
        if pad_size <= 0:
            return tensor, origin_length
        pad = [0] * (2 * tensor.dim())
        pad[-(2 * dim + 1)] = pad_size
        return torch.nn.functional.pad(tensor, pad), origin_length

    @staticmethod
    def remove_pad_for_context_parallel(tensor, dim, origin_length):
        return tensor.narrow(dim, 0, origin_length)

    @staticmethod
    def remove_pad_with_encoder_for_context_parallel(tensor, encoder_length, dim, origin_length):
        total_length = tensor.size(dim)
        split_point = total_length - encoder_length
        first_raw = tensor.narrow(dim, 0, split_point)
        first = first_raw.narrow(dim, 0, origin_length)
        second = tensor.narrow(dim, split_point, encoder_length)
        return torch.cat([first, second], dim=dim)

    def split_input(self, x, dim):
        cp_group = mpu.get_context_parallel_group()
        x, origin_length = self.pad_for_context_parallel(x, dim)
        x = split_forward_gather_backward(x, cp_group, dim=dim, grad_scale="none")
        return x, origin_length

    def gather_output(self, output, dim, origin_length):
        cp_group = mpu.get_context_parallel_group()
        output = gather_forward_split_backward(output, cp_group, dim=dim, grad_scale="none")
        return self.remove_pad_for_context_parallel(output, dim, origin_length)

    def forward_attn(self, q, k, v):
        # The block that owns this attention monkey-patched our forward_attn here
        # and is responsible for setting _cp_origin_length on itself before
        # invoking attention. Reading missing => programmer error, fail loud.
        origin_length = self._cp_origin_length
        cp_group = mpu.get_context_parallel_group()
        args = get_args()
        num_heads = args.num_attention_heads // mpu.get_tensor_model_parallel_world_size()

        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)

        if mpu.get_context_parallel_world_size() > 1:
            q = SeqAllToAll.apply(cp_group, q, 2, 1)
            k = SeqAllToAll.apply(cp_group, k, 2, 1)
            v = SeqAllToAll.apply(cp_group, v, 2, 1)
            q, k, v = (
                self.remove_pad_for_context_parallel(t, 1, origin_length) for t in (q, k, v)
            )

        if FLASH_ATTN_3_AVAILABLE:
            x = flash_attn_interface.flash_attn_func(q, k, v)[0]
            x = x.transpose(1, 2).contiguous()
        else:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            x = F.scaled_dot_product_attention(q, k, v)

        if mpu.get_context_parallel_world_size() > 1:
            x, _ = self.pad_for_context_parallel(x, 2)
            x = SeqAllToAll.apply(cp_group, x, 2, 1)

        x = x.transpose(1, 2).flatten(2, 3).contiguous()
        return x

    def gate_with_cp_grad_reduce(self, x, gate, residual):
        return GateWithGradReduce.apply(x, gate, residual)

    def modulate_with_cp_grad_reduce(self, x, shift, scale):
        return ModulateWithCPGradReduce.apply(x, shift, scale)

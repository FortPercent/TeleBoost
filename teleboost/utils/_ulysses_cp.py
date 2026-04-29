"""TeleBoost autograd Functions for cp-aware modulation/gate inside wan blocks.

Upstream verl@v0.4.0 has the Ulysses sequence-parallel scaffolding but not
these wan-specific autograd Functions; the project added them. They live
here in teleboost so we can keep the upstream verl pin clean, and we patch
them onto verl.utils.ulysses at runtime via teleboost.patches.

NOTE: the backward of `ModulateWithCPGradReduce` and `GateWithGradReduce`
already SUM-allreduces the modulation/shift/scale grads inside the
sp_group; the cp-fix patch in teleboost.patches.ulysses_cp_fix accounts
for that by skipping modulation in `register_cp_grad_reduce_hook`.
"""
from __future__ import annotations

import torch


def _sp_group():
    from verl.utils.ulysses import get_ulysses_sequence_parallel_group
    return get_ulysses_sequence_parallel_group()


class GateWithGradReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, residual):
        ctx.save_for_backward(gate, residual)
        return x + gate * residual

    @staticmethod
    def backward(ctx, x_grad):
        gate, residual = ctx.saved_tensors
        r_grad = x_grad * gate
        gate_grad = torch.sum(x_grad * residual, dim=1, keepdim=True)
        torch.distributed.all_reduce(gate_grad, group=_sp_group())
        return x_grad, gate_grad, r_grad


class ModulateWithCPGradReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shift, scale):
        ctx.save_for_backward(x, scale)
        return x * (1 + scale) + shift

    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        x_grad = grad_output * (1 + scale)
        scale_grad = torch.sum(grad_output * x, dim=1, keepdim=True)
        torch.distributed.all_reduce(scale_grad, group=_sp_group())
        shift_grad = torch.sum(grad_output, dim=1, keepdim=True)
        torch.distributed.all_reduce(shift_grad, group=_sp_group())
        return x_grad, shift_grad, scale_grad


def gate_with_cp_grad_reduce(x, gate, residual):
    return GateWithGradReduce.apply(x, gate, residual)


def modulate_with_cp_grad_reduce(x, shift, scale):
    return ModulateWithCPGradReduce.apply(x, shift, scale)

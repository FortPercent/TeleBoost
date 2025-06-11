
import torch.nn as nn 
import torch
from vast.models.dit.wan_dit.wan_video_dit import DiTBlock
from megatron.core import mpu 


class ContextParallelGateModule(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, gate, residual):
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
        torch.distributed.all_reduce(gate_grad, group=mpu.get_context_parallel_group())
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
        torch.distributed.all_reduce(scale_grad, group=mpu.get_context_parallel_group())
        
        return x_grad, grad_output, scale_grad


def modulate_with_cp_grad_reduce(x, shift, scale):
    return ModulateWithCPGradReduce.apply(x, shift, scale)



class ContextParallelDitBlock(DiTBlock):
    def forward(self, x, context, t_mod, freqs):
        # msa: multi-head self-attention  mlp: multi-layer perceptron

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate_with_cp_grad_reduce(self.norm1(x), shift_msa, scale_msa)
        attn_output = self.self_attn(input_x, freqs)
        # print("before gate", attn_output.shape, gate_msa.shape, x.shape)
        x = self.gate(x, gate_msa, attn_output)
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate_with_cp_grad_reduce(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate2(x, gate_mlp, self.ffn(input_x))
        return x

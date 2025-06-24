from typing import Tuple, Optional
import torch
import torch.nn as nn
from teletron.core.context_parallel import ContextParallelMixin
from teletron.core.transformer import TransformerGeneralMixin
from .wan_model import WanModel, DiTBlock, sinusoidal_embedding_1d
from megatron.core import mpu 
import logging


class ContextParallelWanDitBlock(ContextParallelMixin, DiTBlock):
    def __init__(self, *args, **kwargs):
        DiTBlock.__init__(self, *args, **kwargs)
        # from ContextParallelMixin
        self.enable_context_parallel(self.self_attn.attn)

    def forward(self, x, context, t_mod, freqs):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = self.modulate_with_cp_grad_reduce(self.norm1(x), shift_msa, scale_msa)
        attn_output = self.self_attn(input_x, freqs)
        x = self.gate_with_cp_grad_reduce(x, gate_msa, attn_output)
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = self.modulate_with_cp_grad_reduce(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate_with_cp_grad_reduce(x, gate_mlp, self.ffn(input_x))
        return x


class ParallelWanModel(ContextParallelMixin, TransformerGeneralMixin, WanModel):
    def __init__(self, dim: int=40*128,
                in_dim: int=36,
                ffn_dim: int=13824,
                out_dim: int=16,
                text_dim: int=4096,
                freq_dim: int=256,
                eps: float=1e-6,
                patch_size: Tuple[int, int, int]=(1,2,2),
                num_heads: int=40,
                num_layers: int=5,
                has_image_input: bool=True,
                has_image_pos_emb: bool=False,
                context_parallel_dim: int = 1,
                config=None):
        self.config=config
        WanModel.__init__(self, dim,
                        in_dim,
                        ffn_dim,
                        out_dim,
                        text_dim,
                        freq_dim,
                        eps,
                        patch_size,
                        num_heads,
                        num_layers,
                        has_image_input,
                        has_image_pos_emb)
        
        self.blocks = nn.ModuleList([
            ContextParallelWanDitBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])

        # from TransformerGeneralMixin
        from teletron.utils import get_args
        args = get_args()
        if args.activation_offload:
            self.enable_activation_offload(self.blocks)
        else:
            self.enable_activation_checkpointing(self.blocks)

        # from ContextParallelMixin
        self.register_cp_grad_reduce_hook()

    
    def register_cp_grad_reduce_hook(self):
        
        # layers with parallel input sequence need to reduce its param gradient.
        # list the parameters that needs grad reduce and register tensor grad hook

        for name, param in self.named_parameters():
            if name.startswith("patch_embedding") or \
                    name.startswith("time") or\
                        name.startswith("head") or \
                             "modulation" in name:
                continue 

            param.register_hook(self.cp_grad_reduce)

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                cn_images=None, 
                **kwargs,):
        # Do whatever necessary before forward transformer blocks
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        if cn_images is not None:
            x = torch.cat([x, cn_images], dim=1)  # (b, c_x + c_y, f, h, w)
        
        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        # Split input sequence and rope (with methods from CPMixin), and forward CP transformer blocks
        x = self.split_input(x, dim=1)
        freqs = self.split_input(freqs, dim=0)
        x = self.blocks(x, context, t_mod, freqs)
        x = self.gather_output(x, dim=1)

        # Now x is in full shape, just do regular forward
        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x


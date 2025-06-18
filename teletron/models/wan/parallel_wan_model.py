from typing import Tuple, Optional
import torch
from teletron.core.context_parallel import ContextParallelMixin, \
    modulate_with_cp_grad_reduce, gate_with_cp_grad_reduce
from teletron.core.transformer import TransformerGeneralMixin
from .wan_model import WanModel, DiTBlock
from megatron.core import mpu
from teletron import get_args



class ContextParallelWanDitBlock(DiTBlock):
    def forward(self, x, context, t_mod, freqs):
        # msa: multi-head self-attention  mlp: multi-layer perceptron

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate_with_cp_grad_reduce(self.norm1(x), shift_msa, scale_msa)
        attn_output = self.self_attn(input_x, freqs)
        # print("before gate", attn_output.shape, gate_msa.shape, x.shape)
        x = gate_with_cp_grad_reduce(x, gate_msa, attn_output)
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate_with_cp_grad_reduce(self.norm2(x), shift_mlp, scale_mlp)
        x = gate_with_cp_grad_reduce(x, gate_mlp, self.ffn(input_x))
        return x


class ParallelWanModel(ContextParallelMixin, TransformerGeneralMixin, WanModel):
    def __init__(self, dim: int,
                in_dim: int,
                ffn_dim: int,
                out_dim: int,
                text_dim: int,
                freq_dim: int,
                eps: float,
                patch_size: Tuple[int, int, int],
                num_heads: int,
                num_layers: int,
                has_image_input: bool,
                has_image_pos_emb: bool,
                context_parallel_dim: int = 1):
        
        WanModel.__init__(dim,
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
        args = get_args()
        self.num_layers = num_layers
        self.activation_recompute_method = args.recompute_method
        self.recompute_granularity = args.recompute_granularity
        self.recompute_num_layers = args.recompute_num_layers
        self.enable_activation_checkpointing(self.blocks)

        # from ContextParallelMixin
        self.cp_size = mpu.get_context_parallel_world_size()
        self.cp_group = mpu.get_context_parallel_group()
        self.split_dim = context_parallel_dim
        self.gather_dim = context_parallel_dim
        for i in range(len(self.blocks)):
            self.enable_context_parallel_attention(self.blocks[i].self_attn.attn)
    
    # TODO: grad reduce hook, gate grad reduce and modulation grad reduce

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


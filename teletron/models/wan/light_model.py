import torch 
import torch.nn as nn
from typing import Tuple, Optional
from vast.models.dit.wan_dit import WanModel 
from megatron.core import mpu
from teletron.core.context_parallel.pad import pad_for_context_parallel, \
        remove_pad_for_context_parallel, \
        set_origin_length, set_target_length
from teletron.models.wan.layers import ContextParallelGateModule, ContextParallelDitBlock

from teletron.models.wan.attention import ContextParallelAttentionModule, ContextParallelCrossAttentionModule

def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


class TeletronWanModel(WanModel):
    def __init__(
        self,
        wanConfig
    ):
        super().__init__(
            dim=wanConfig.num_attention_heads * wanConfig.attention_head_dim,
            in_dim=wanConfig.in_channels,
            ffn_dim=wanConfig.ffn_dim,
            out_dim=wanConfig.out_channels,
            text_dim=wanConfig.text_dim,
            freq_dim=wanConfig.freq_dim,
            eps=wanConfig.eps,
            patch_size=wanConfig.patch_size,
            num_heads=wanConfig.num_attention_heads,
            num_layers=wanConfig.num_layers,
            has_image_input=wanConfig.has_image_input,
            has_image_pos_emb=wanConfig.has_image_pos_emb
        )
        self.wan_config = wanConfig

        self.substitute_attention_modules()
        self.register_cp_grad_reduce_hook()
    
    def register_cp_grad_reduce_hook(self):
        def cp_grad_reduce(grad):
            with torch.no_grad():
                grad_list = [torch.empty_like(grad) for _ in range(mpu.get_context_parallel_world_size())]
                torch.distributed.all_gather(grad_list, grad, group=mpu.get_context_parallel_group())
                reduced_grad = torch.sum(torch.stack(grad_list), dim=0)
            
            return reduced_grad
        
        # def get_layer_param_name(key, weight=True, bias=True, num_layers=None):

        # layers with input sequence that is not parallel do not need to reduce its gradient.
        self.wgrad_not_to_reduce = [
            "head.head.weight",
            "head.head.bias",
            "head.modulation"] + [
                f"blocks.{i}.cross_attn.v_img.bias" for i in range(self.wan_config.num_layers)
            ] + [
                f"blocks.{i}.cross_attn.v_img.weight" for i in range(self.wan_config.num_layers)
            ] + [
                f"blocks.{i}.cross_attn.norm_k_img.weight" for i in range(self.wan_config.num_layers)
            ] + [
                f"blocks.{i}.cross_attn.k_img.weight"  for i in range(self.wan_config.num_layers)
            ] + [
                f"blocks.{i}.cross_attn.k_img.bias" for i in range(self.wan_config.num_layers)
            ] + [
                f"blocks.{i}.cross_attn.v.bias" for i in range(self.wan_config.num_layers)
            ] + [
                f"blocks.{i}.cross_attn.v.weight" for i in range(self.wan_config.num_layers)
            ] + [
                f"blocks.{i}.cross_attn.norm_k.weight" for i in range(self.wan_config.num_layers)
            ] + [
                f"blocks.{i}.cross_attn.k.bias" for i in range(self.wan_config.num_layers)
            ] + [
                f"blocks.{i}.cross_attn.k.weight" for i in range(self.wan_config.num_layers)
            ]
        
        for name, param in self.named_parameters():
            if name.startswith("patch_embedding") or\
                  name.startswith("img_emb") or \
                    name.startswith("text_embedding") or \
                        name.startswith("time") or "modulation" in name:
                continue 
            if name not in self.wgrad_not_to_reduce:
                param.register_hook(cp_grad_reduce)
    

    def substitute_attention_modules(self):
        for i in range(len(self.blocks)):
            self.blocks[i] = ContextParallelDitBlock(
                has_image_input=True, 
                dim=self.wan_config.num_attention_heads * self.wan_config.attention_head_dim,
                num_heads=self.wan_config.num_attention_heads, 
                ffn_dim=self.wan_config.ffn_dim, 
                eps=self.wan_config.eps
            )
            self.blocks[i].self_attn.attn = \
                ContextParallelAttentionModule(
                    num_heads=self.wan_config.num_attention_heads,
                )
            self.blocks[i].cross_attn.attn = \
                ContextParallelCrossAttentionModule(
                    num_heads=self.wan_config.num_attention_heads,
                )
            self.blocks[i].cross_attn.attn2 = \
                ContextParallelCrossAttentionModule(
                    num_heads=self.wan_config.num_attention_heads,
                )
            self.blocks[i].gate2 = ContextParallelGateModule()
            self.blocks[i].gate = ContextParallelGateModule()
    

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                cn_images=None, 
                **kwargs,):
        x, context, t, t_mod, freqs, fhw = self.preforward_transformer_blocks(
            x, y, timestep, context, clip_feature, cn_images
        )

        # split input sequences
        if mpu.get_context_parallel_world_size() > 1:
            # pad if sequence len cannot be divided by CP size
            length = x.shape[1]
            set_origin_length(length)
            use_padding = False
            seq_parallel_world_size = mpu.get_context_parallel_world_size()
            if length % seq_parallel_world_size != 0:
                pad_size = seq_parallel_world_size - (length % seq_parallel_world_size)
                length = length + pad_size
                set_target_length(length)
                x = pad_for_context_parallel(x, 1)
                freqs = pad_for_context_parallel(freqs, 0)
            
            

            from teletron.core.tensor_parallel.mappings import (
                split_forward_gather_backward,
                gather_forward_split_backward,
            )

            x = split_forward_gather_backward(
                x, mpu.get_context_parallel_group(), dim=1, grad_scale="None"
            )  # b s n d

            freqs = split_forward_gather_backward(
                freqs, mpu.get_context_parallel_group(), dim=0, grad_scale="None"
            )

        x = self.forward_transformer_blocks(
                x, context, t_mod, freqs,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                **kwargs,
        )

        # gather output sequences
        if mpu.get_context_parallel_world_size() > 1:
            
            x = gather_forward_split_backward(
                x, mpu.get_context_parallel_group(), dim=1, grad_scale="None"
            )
            x = remove_pad_for_context_parallel(x, 1)
        
        x = self.postforward_transformer_blocks(x, t, fhw)
        return x

    def preforward_transformer_blocks(self, x, y, timestep, context, clip_feature, cn_images):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)
        
        if cn_images is not None:
            x = torch.cat([x, cn_images], dim=1)  # (b, c_x + c_y, f, h, w)
        
        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        return x, context, t, t_mod, freqs, (f, h, w)

    def forward_transformer_blocks(self,
                x: torch.Tensor,
                context: torch.Tensor,
                t_mod: torch.Tensor,
                freqs: torch.Tensor, 
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,):

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs)
        return x 

    
    def postforward_transformer_blocks(self, x, t, fhw):
        x = self.head(x, t)
        x = self.unpatchify(x, fhw)
        return x


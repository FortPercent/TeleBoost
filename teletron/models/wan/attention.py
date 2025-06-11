import torch 
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from vast.models.dit.wan_dit.wan_video_dit import AttentionModule, CrossAttention
from megatron.core import mpu
from teletron.core.context_parallel.pad import pad_for_context_parallel, \
    remove_pad_for_context_parallel, \
    remove_pad_with_encoder_for_context_parallel

from teletron.core.tensor_parallel.mappings import (
            split_forward_gather_backward,
            gather_forward_split_backward,
        )

# def rope_apply(x, freqs, num_heads):
#     x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
#     x_out = torch.view_as_complex(x.to(torch.float64).reshape(
#         x.shape[0], x.shape[1], x.shape[2], -1, 2))
#     x_out = torch.view_as_real(x_out * freqs).flatten(2)
#     return x_out.to(x.dtype)
T5_CONTEXT_TOKEN_NUMBER = 512

class ContextParallelAttentionModule(AttentionModule):
    
    def forward(self, q, k, v):
        # print("in attention qkv", q.shape, k.shape, v.shape)
        q = rearrange(q, "b s (n d) -> b s n d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=self.num_heads)
        if mpu.get_context_parallel_world_size() > 1:
            from yunchang.comm.all_to_all import SeqAllToAll4D
            # qkv: b s/CP n d
            q = SeqAllToAll4D.apply(mpu.get_context_parallel_group(), q, 2, 1)
            k = SeqAllToAll4D.apply(mpu.get_context_parallel_group(), k, 2, 1)
            v = SeqAllToAll4D.apply(mpu.get_context_parallel_group(), v, 2, 1)
            # qkv: b s n/CP d
            q,k,v = map(
                lambda x: remove_pad_for_context_parallel(x, dim=1),
                [q,k,v]
            )
            torch.cuda.empty_cache()
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        # qkv: b n/CP s d

        # print("before sdpa", q.shape, k.shape, v.shape)
        x = F.scaled_dot_product_attention(q, k, v)
        # print("after sdpa", x.shape)
        if mpu.get_context_parallel_world_size() > 1:
            if x.shape[1] % mpu.get_context_parallel_world_size() != 0:
                x = pad_for_context_parallel(x, 2)
            x = SeqAllToAll4D.apply(
                mpu.get_context_parallel_group(), x, 2, 1
            )  # b img_seq sub_n d
            torch.cuda.empty_cache()
            # x: b n s/CP d
        # print("after all2all", x.shape)
        x = x.transpose(1, 2).flatten(2, 3).contiguous()
        # x: b s h

        return x
    

class ContextParallelCrossAttentionModule(AttentionModule):
    def forward(self, q, k, v):
        # print("in attention qkv", q.shape, k.shape, v.shape)
        q = rearrange(q, "b s (n d) -> b s n d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=self.num_heads)
        if mpu.get_context_parallel_world_size() > 1:
            from yunchang.comm.all_to_all import SeqAllToAll4D
            # qkv: b s/CP n d
            q = SeqAllToAll4D.apply(mpu.get_context_parallel_group(), q, 2, 1)
            v = split_forward_gather_backward(
                v, mpu.get_context_parallel_group(), dim=2, grad_scale="None"
            ) 
            k = split_forward_gather_backward(
                k, mpu.get_context_parallel_group(), dim=2, grad_scale="None"
            )  # b s n d
            torch.cuda.empty_cache()
            # qkv: b s n/CP d
            q = remove_pad_for_context_parallel(q, dim=1)

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        # qkv: b n/CP s d

        # print("before sdpa", q.shape, k.shape, v.shape)
        x = F.scaled_dot_product_attention(q, k, v)
        # print("after sdpa", x.shape)
        if mpu.get_context_parallel_world_size() > 1:
            if x.shape[1] % mpu.get_context_parallel_world_size() != 0:
                x = pad_for_context_parallel(x, 2)
            x = SeqAllToAll4D.apply(
                mpu.get_context_parallel_group(), x, 2, 1
            )  # b img_seq sub_n d
            torch.cuda.empty_cache()
            # x: b n s/CP d
        # print("after all2all", x.shape)
        x = x.transpose(1, 2).flatten(2, 3).contiguous()
        # x: b s h

        return x
    

import torch
import torch.nn.functional as F
from einops import rearrange
from megatron.core import mpu
from teletron.core.context_parallel.mappings import split_forward_gather_backward,\
        gather_forward_split_backward

class ContextParallelMixin:
    
    def split_input(self, x):
        # assert x is not parallel
        if x.shape[self.split_dim] % self.cp_size != 0 :
            self.origin_length = x.shape[self.split_dim]
            self.padded_length = self.origin_length + self.cp_size - \
                (self.origin_length % self.cp_size)
            x = self.pad_for_context_parallel(x)
            self.use_pad = True
        else:
            self.use_pad = False
            
        x = split_forward_gather_backward(x, self.cp_group, dim=self.split_dim, grad_scale="none")
        return x
    
    def gather_output(self, output):
        output = gather_forward_split_backward(output, self.cp_group, dim=self.gather_dim, grad_scale="none")
        if self.use_pad:
            output = self.remove_pad_for_context_parallel(output)
        return output 

    def pad_for_context_parallel(self, tensor):
        pad_size = int(self.padded_length - self.origin_length)

        if pad_size <= 0:
            return tensor  # No padding needed

        # Create pad tuple: (dim_n_before, dim_n_after, ..., dim_0_before, dim_0_after)
        pad = [0] * (2 * tensor.dim())
        pad[-(2 * self.split_dim + 1)] = pad_size  # pad after the dimension
        return torch.nn.functional.pad(tensor, pad) 
    
    def remove_pad_for_context_parallel(self, tensor):
        return tensor.narrow(self.gather_dim, 0, self.origin_length)

    def forward_attn(self, q, k, v):
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
                lambda x: self.remove_pad_for_context_parallel(x, dim=1),
                [q,k,v]
            )
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        # qkv: b n/CP s d

        # print("before sdpa", q.shape, k.shape, v.shape)
        x = F.scaled_dot_product_attention(q, k, v)
        # print("after sdpa", x.shape)
        if mpu.get_context_parallel_world_size() > 1:
            if x.shape[2] % mpu.get_context_parallel_world_size() != 0:
                x = self.pad_for_context_parallel(x, 2)
            x = SeqAllToAll4D.apply(
                mpu.get_context_parallel_group(), x, 2, 1
            )  # b img_seq sub_n d
            torch.cuda.empty_cache()
            # x: b n s/CP d
        # print("after all2all", x.shape)
        x = x.transpose(1, 2).flatten(2, 3).contiguous()
        # x: b s h

        return x
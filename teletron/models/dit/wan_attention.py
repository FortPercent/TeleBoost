from dataclasses import dataclass
from typing import Union
from einops import rearrange
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
import torch
from megatron.core import mpu

# from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from diffusers.models.embeddings import apply_rotary_emb
from megatron.core.transformer.attention import Attention, SelfAttention,CrossAttention
from megatron.core.transformer.custom_layers.transformer_engine import SplitAlongDim
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from teletron.core.tensor_parallel.mappings import split_forward_gather_backward, gather_forward_split_backward
import torch.nn as nn
from teletron.core.process_groups_config import ModelCommProcessGroups

from megatron.core.transformer.attention import (
    CrossAttention,
    CrossAttentionSubmodules,
    SelfAttention,
    SelfAttentionSubmodules,
)
from diffusers.models.normalization import FP32LayerNorm, LpNorm, RMSNorm

# @dataclass
# class WanAttentionSubmodules:
#     linear_qkv: Union[ModuleSpec, type] = None
#     added_linear_qkv: Union[ModuleSpec, type] = None
#     core_attention: Union[ModuleSpec, type] = None
#     linear_proj: Union[ModuleSpec, type] = None
#     q_layernorm: Union[ModuleSpec, type] = None
#     k_layernorm: Union[ModuleSpec, type] = None
#     added_q_layernorm: Union[ModuleSpec, type] = None
#     added_k_layernorm: Union[ModuleSpec, type] = None


@dataclass
class WanCrossAttentionSubmodules:
    linear_q: Union[ModuleSpec, type] = None
    linear_k: Union[ModuleSpec, type] = None
    linear_v: Union[ModuleSpec, type] = None
    k_img: Union[ModuleSpec, type] = None
    v_img: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None
    norm_k_img: Union[ModuleSpec, type] = None


class WanSelfAttention(Attention):
    """Joint Self-attention layer class

    Used for MMDIT-like transformer block.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        eps: float = 1e-6,
        context_pre_only: bool = False,
        model_comm_pgs: ModelCommProcessGroups = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
        )
        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.q_layernorm = build_module(
            submodules.q_layernorm,
            hidden_size=config.hidden_size,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

        self.k_layernorm = build_module(
            submodules.k_layernorm,
            hidden_size=config.hidden_size,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

    def _split_qkv(self, mixed_qkv):
        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (  # [1, 360] + (12, ())
            self.num_query_groups_per_partition,
            (
                (
                    self.num_attention_heads_per_partition
                    // self.num_query_groups_per_partition
                    + 2
                )
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
        else:

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(
            query.size(0), query.size(1), -1, self.hidden_size_per_attention_head
        )
        return query, key, value

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        query, key, value = self._split_qkv(mixed_qkv)
        # [2, 9604, 12, 128] [b, s, num_heads, hiddensize_per_head]
        # batch_size, sequense_lenth, num_heads, hiddensize_per_head

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        # batch_size, num_head, sequense_lenth, hiddensize_per_head

        if self.config.test_mode:
            self.run_realtime_tests()

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        return query, key, value

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
        encoder_hidden_states=None,
    ):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # hidden_states: [sq, b, h]

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        # if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
        #     rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        # bs, img_seq_len, _, _ = img_q.shape
        query = self.linear_q(hidden_states)
        key = self.linear_k(hidden_states)
        value = self.linear_v(hidden_states)
        query = self.q_layernorm(query)
        key = self.k_layernorm(key)
        # batch_size, num_head, sequense_lenth, hiddensize_per_head

        # query = query.unflatten(2, (self.config.num_attention_heads, -1)).transpose(1, 2)
        # key = key.unflatten(2, (self.config.num_attention_heads, -1)).transpose(1, 2)
        # value = value.unflatten(2, (self.config.num_attention_heads, -1)).transpose(1, 2)
        # bs, _, img_seq_len, _ = query.shape
        # # ===================================================
        # # Adjust key, value, and rotary_pos_emb for inference
        # # ===================================================
        # key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
        #     inference_params, key, value, rotary_pos_emb
        # )

        # if packed_seq_params is not None:
        #     query = query.squeeze(1)
        #     key = key.squeeze(1)
        #     value = value.squeeze(1)

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rotary_pos_emb is not None:
            query = rope_apply(query, rotary_pos_emb, self.config.num_attention_heads)
            key = rope_apply(key, rotary_pos_emb, self.config.num_attention_heads)

        key = key.view(key.shape[0], key.shape[1], -1, self.config.num_attention_heads)
        query = query.view(
            key.shape[0], key.shape[1], -1, self.config.num_attention_heads
        )
        value = value.view(
            key.shape[0], key.shape[1], -1, self.config.num_attention_heads
        )

        if mpu.get_context_parallel_world_size() > 1:
            from yunchang.comm.all_to_all import SeqAllToAll4D

            query = SeqAllToAll4D.apply(mpu.get_context_parallel_group(), query, 2, 1)
            key = SeqAllToAll4D.apply(mpu.get_context_parallel_group(), key, 2, 1)
            value = SeqAllToAll4D.apply(mpu.get_context_parallel_group(), value, 2, 1)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        # TODO, can apply positional embedding to value_layer so it has
        # absolute positional embedding.
        # otherwise, only relative positional embedding takes effect
        # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)
        # ==================================
        # core attention computation
        # ==================================
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask, attn_mask_type=self.attn_mask_type
            )
        else:
            # core_attn_out = self.core_attention(
            #     query,
            #     key,
            #     value,
            #     attention_mask,
            #     attn_mask_type=self.attn_mask_type,
            # )
            import torch.nn.functional as F

            torch.backends.cuda.enable_cudnn_sdp(False)
            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )  # b h s d

        # if mpu.get_context_parallel_group() is not None:
        if mpu.get_context_parallel_world_size() > 1:
            hidden_states = SeqAllToAll4D.apply(
                mpu.get_context_parallel_group(), hidden_states, 2, 1
            )  # b img_seq sub_n d
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).contiguous()
        hidden_states = self.linear_proj(hidden_states)
        # hidden_states=hidden_states+bias
        return hidden_states


class WanCrossAttention(Attention):
    """Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: WanCrossAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        eps: float = 1e-6,
        context_pre_only: bool = False,
        model_comm_pgs: ModelCommProcessGroups = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="cross",
        )

        self.q_layernorm = build_module(
            submodules.q_layernorm,
            hidden_size=config.hidden_size,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

        self.k_layernorm = build_module(
            submodules.k_layernorm,
            hidden_size=config.hidden_size,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

        self.norm_k_img = build_module(
            submodules.norm_k_img,
            hidden_size=config.hidden_size,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.k_img = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_img = nn.Linear(config.hidden_size, config.hidden_size)

    def _split_qkv(self, mixed_qkv):
        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (  # [1, 360] + (12, ())
            self.num_query_groups_per_partition,
            (
                (
                    self.num_attention_heads_per_partition
                    // self.num_query_groups_per_partition
                    + 2
                )
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
        else:

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(
            query.size(0), query.size(1), -1, self.hidden_size_per_attention_head
        )
        return query, key, value

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        query, key, value = self._split_qkv(mixed_qkv)
        # [2, 9604, 12, 128] [b, s, num_heads, hiddensize_per_head]

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if self.config.test_mode:
            self.run_realtime_tests()

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        return query, key, value

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
    ):
        encoder_hidden_states_img = None
        if self.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]

        query = self.linear_q(hidden_states)
        key = self.linear_k(encoder_hidden_states)
        value = self.linear_v(encoder_hidden_states)

        query = self.q_layernorm(query)
        key = self.k_layernorm(key)

        if rotary_pos_emb is not None:
            query = rope_apply(query, rotary_pos_emb)
            key = rope_apply(key, rotary_pos_emb)

        hidden_states_img = None
        key_img = self.k_img(encoder_hidden_states_img)
        value_img = self.v_img(encoder_hidden_states_img)

        key_img = self.norm_k_img(key_img)

        # ==================================
        # core attention computation
        # ==================================
        key = key.view(key.shape[0], key.shape[1], self.config.num_attention_heads,-1)
        query = query.view(
            query.shape[0], query.shape[1],  self.config.num_attention_heads,-1
        )
        value = value.view(
            value.shape[0], value.shape[1], self.config.num_attention_heads,-1,
        )
        key_img = key_img.view(
            key_img.shape[0], key_img.shape[1],  self.config.num_attention_heads,-1
        )
        value_img = value_img.view(
            value_img.shape[0], value_img.shape[1],  self.config.num_attention_heads,-1
        )
        if mpu.get_context_parallel_world_size() > 1:
            from yunchang.comm.all_to_all import SeqAllToAll4D
            query = SeqAllToAll4D.apply(mpu.get_context_parallel_group(), query, 2, 1)
            value = split_forward_gather_backward(
                value, mpu.get_context_parallel_group(), dim=2, grad_scale="down"
            ) 
            key = split_forward_gather_backward(
                key, mpu.get_context_parallel_group(), dim=2, grad_scale="down"
            )  # b s n d
            key_img = split_forward_gather_backward(
                key_img, mpu.get_context_parallel_group(), dim=2, grad_scale="down"
            )  # b s n d
            value_img = split_forward_gather_backward(
                value_img, mpu.get_context_parallel_group(), dim=2, grad_scale="down"
            )  # b s n d
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        key_img = key_img.transpose(1, 2)
        value_img = value_img.transpose(1, 2)
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask, attn_mask_type=self.attn_mask_type
            )
        else:
            # core_attn_out = self.core_attention(
            #     query,
            #     key,
            #     value,
            #     attention_mask,
            #     attn_mask_type=self.attn_mask_type,
            # )
            import torch.nn.functional as F

            # query, key, value = [x.permute(1, 2, 0, 3).contiguous() for x in (query, key, value)] # sbhd -> bhsd
            torch.backends.cuda.enable_cudnn_sdp(False)
            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            hidden_states_img = F.scaled_dot_product_attention(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            hidden_states = hidden_states + hidden_states_img
        # # if mpu.get_context_parallel_group() is not None:
        if mpu.get_context_parallel_world_size() > 1:
            hidden_states = SeqAllToAll4D.apply(
                mpu.get_context_parallel_group(), hidden_states, 2, 1
        )
        # print("after all to all",hidden_states.shape)
        # print("&"*100)
        # b sub_n img_seq d
        #     encoder_hidden_states=gather_forward_split_backward(encoder_hidden_states, mpu.get_context_parallel_group(), 2) # b txt_seq n d
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).contiguous()
        hidden_states = self.linear_proj(hidden_states)

        return hidden_states


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)

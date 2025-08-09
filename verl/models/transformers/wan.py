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

import inspect
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLCausalLMOutputWithPast,
    Qwen2VLForConditionalGeneration,
)
from transformers.utils import is_flash_attn_greater_or_equal

from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_world_size,
    validate_ulysses_config,
)

def ulysses_self_flash_attn_forward(
    self,
    x: torch.Tensor,
    seq_lens, grid_sizes, freqs,  # will become mandatory in v4.46
    **kwargs,
):
    from wan.modules.model import rope_apply
    from wan.modules.attention import flash_attention
    from verl.utils.ulysses import get_target_len
    attention_mask=None
    # bsz, q_len, _ = x.size()  # q_len = seq_length / sp_size
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    f, h, w = grid_sizes[0,:]
    if ulysses_sp_size > 1:
        validate_ulysses_config(self.num_heads, ulysses_sp_size)
        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)
        target = get_target_len()
        q = gather_seq_scatter_heads(q, seq_dim=1, head_dim=2,unpadded_dim_size=target)
        k = gather_seq_scatter_heads(k, seq_dim=1, head_dim=2,unpadded_dim_size=target)
        v = gather_seq_scatter_heads(v, seq_dim=1, head_dim=2,unpadded_dim_size=target)
        
        full_q_len = q.size(1)  # full_q_len = seq_length
    else:
        full_q_len = s

    attn_output = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

    # attn_output = attn_output.transpose(1, 2).flatten(2, 3).contiguous()
    if ulysses_sp_size > 1:
        attn_output = gather_heads_scatter_seq(attn_output, head_dim=2, seq_dim=1)

    attn_output = attn_output.flatten(2).contiguous()

    attn_output = self.o(attn_output)
    return attn_output
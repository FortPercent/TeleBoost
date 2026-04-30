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


# ---------------------------------------------------------------------------
# Wan-specific Ulysses input-slicing / head-gather monkey patches.
# Pre-X3 lived inside the in-tree verl `apply_monkey_patch` for `model_type=="t2v"`.
# After X3 dropped that fork, we apply them directly to the Wan model from the
# recipe's `_build_model_optimizer` when `ulysses_sequence_parallel_size > 1`.
# ---------------------------------------------------------------------------
def patch_diffusion_for_ulysses_input_slicing(model) -> None:
    """Wrap `model.blocks[0].forward` to slice the input `x` along seq-dim across SP ranks.

    The pre-X3 contract: only the first block's forward is wrapped; subsequent
    blocks see the already-sliced tensor (the model passes `x` through the chain).
    The wrapper also stashes the original sequence length / pad size into module
    state so the matching head-gather wrapper can undo the operation.
    """
    from verl.utils.ulysses import (
        diffusion_slice_input_tensor_pad,
        get_ulysses_sequence_parallel_world_size,
    )

    def _wrap(original_forward):
        def wrapped(*args, **kwargs):
            x = kwargs.get("x")
            if x is not None and get_ulysses_sequence_parallel_world_size() > 1:
                kwargs["x"] = diffusion_slice_input_tensor_pad(x, dim=1, padding=True)
            return original_forward(*args, **kwargs)

        return wrapped

    try:
        model.blocks[0].forward = _wrap(model.blocks[0].forward)
        print(f"[teleboost] Patched {type(model).__name__}.blocks[0].forward for Ulysses SP input slicing.")
    except Exception as e:
        print(f"[teleboost] Failed to patch {type(model).__name__} for Ulysses SP input slicing: {e}")


def patch_diffusion_for_ulysses_head_gather(module_class) -> None:
    """Wrap `Head.forward` to all-gather the seq dim back, undoing the input slicing.

    Patched at the class level so it applies to every Head instance the model creates.
    """
    from verl.utils.ulysses import (
        diffusion_gather_outpus_and_unpad,
        get_pad_size,
        get_ulysses_sequence_parallel_world_size,
    )

    def _wrap(original_forward):
        def wrapped(self, *args, **kwargs):
            x = kwargs.get("x")
            if x is not None and get_ulysses_sequence_parallel_world_size() > 1:
                pad_size = get_pad_size() or 0
                kwargs["x"] = diffusion_gather_outpus_and_unpad(
                    x, gather_dim=1, unpad_dim=1, padding_size=pad_size
                )
            return original_forward(self, *args, **kwargs)

        return wrapped

    try:
        module_class.forward = _wrap(module_class.forward)
        print(f"[teleboost] Patched {module_class.__name__}.forward for Ulysses SP head gather.")
    except Exception as e:
        print(f"[teleboost] Failed to patch {module_class.__name__} for Ulysses SP head gather: {e}")


def apply_wan_ulysses_patches(model) -> None:
    """Install all three Wan-specific Ulysses patches on a (low or high) WanModel instance.

    Idempotent at the class level for `Head.forward` and `WanSelfAttention.forward`
    (they're class methods); re-running on a second model just re-wraps once more,
    which is benign since `get_ulysses_sequence_parallel_world_size()` short-circuits
    when SP == 1.
    """
    from wan.modules.model import Head, WanSelfAttention

    patch_diffusion_for_ulysses_input_slicing(model)
    patch_diffusion_for_ulysses_head_gather(Head)
    WanSelfAttention.forward = ulysses_self_flash_attn_forward
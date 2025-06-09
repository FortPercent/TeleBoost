# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import copy
from dataclasses import dataclass
from typing import Literal, Union
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from einops import rearrange
from megatron.core.jit import jit_fuser
from megatron.core.transformer.attention import (
    CrossAttention,
    CrossAttentionSubmodules,
    SelfAttention,
    SelfAttentionSubmodules,
)
from megatron.core import mpu
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TENorm,
    TERowParallelLinear,
)
from diffusers.models.normalization import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
)
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.utils import make_viewless_tensor
import torch.nn.functional as F
from teletron.models.dit.wan_attention import (
    WanCrossAttention,
    WanSelfAttention,
    WanCrossAttentionSubmodules,
)

from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.attention import FeedForward
from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
)

from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)

class WANFP32LayerNorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ) #.to(origin_dtype)



class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, config, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        gate * residual
        return x + gate * residual
    

class MLP_WAN(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.linear_fc1 = nn.Linear(dim, ffn_dim)
        self.gelu =  nn.GELU(approximate='tanh')
        self.linear_fc2 = nn.Linear(ffn_dim, dim)

    def forward(self, x):
        y = self.linear_fc1(x)
        y = self.gelu(y)
        y = self.linear_fc2(y)
        return y
    
class WanDiTLayer(TransformerLayer):
    """A double transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.

    WanDiT layer implementation from [https://arxiv.org/pdf/2403.03206].
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        eps: float = 1e-6,
        cross_attn_norm: bool = True,
    ):
        hidden_size = config.hidden_size
        super().__init__(
            config=config, submodules=submodules, layer_number=layer_number
        )

        self.norm1 = WANFP32LayerNorm(hidden_size, eps, elementwise_affine=False)
        self.norm2 = (
            WANFP32LayerNorm(hidden_size, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )

        # 3. Feed-forward
        # self.ffn = FeedForward(hidden_size, inner_dim=config.ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = WANFP32LayerNorm(hidden_size, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 6, hidden_size) / hidden_size**0.5
        )
        self.gate = GateModule()
        # self.gelu = nn.GELU(approximate='tanh')
        ffn_dim = self.mlp.linear_fc1.weight.shape[0]
        del self.mlp
        self.mlp = MLP_WAN(hidden_size, ffn_dim)

        
    def forward(self, x, context, t_mod, freqs):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        norm1_out = self.norm1(x.float())
        #norm1_out = torch.ones_like(norm1_out)
        input_x = modulate(norm1_out, shift_msa, scale_msa)
        print("x",x.shape)
        print("gate_msa",gate_msa.shape)
        print("input_x",input_x.shape)
        x = self.gate(x, gate_msa, self.self_attention(
            hidden_states=input_x.bfloat16(),
            rotary_pos_emb=freqs,
            attention_mask=None))
        norm2_out = self.norm2(x.float())
        #norm2_out = torch.ones_like(norm2_out)
        x = x + self.cross_attention(
            hidden_states=norm2_out.bfloat16(),
            encoder_hidden_states=context,
            attention_mask=None)
        norm3_out = self.norm3(x.float())
        #norm3_out = torch.ones_like(norm3_out)
        input_x = modulate(norm3_out, shift_mlp, scale_mlp)
        ff_output = self.mlp(input_x.bfloat16())
        print("x",x.shape)
        print("ff_output",ff_output.shape)
        print("gate_mlp",gate_mlp.shape)
        x = self.gate(x, gate_mlp, ff_output)
        return x
        

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the transformer block.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
                Defaults to an empty string.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (dict, optional): Additional metadata for sharding.
                Can specify if layers are non-homogeneous. Defaults to None.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the model.
        """
        assert not sharded_offsets, "Unexpected sharded offsets"
        non_homogeneous_layers = metadata is not None and metadata.get(
            "non_homogeneous_layers", False
        )
        if self.config.num_moe_experts is not None:
            non_homogeneous_layers = True

        sharded_state_dict = {}

        layer_prefix = f"{prefix}layers."
        num_layers = self.config.num_layers
        for layer in self.layers:
            offset = TransformerLayer._get_layer_offset(self.config)

            global_layer_offset = (
                layer.layer_number - 1
            )  # self.layer_number starts at 1
            state_dict_prefix = f"{layer_prefix}{global_layer_offset - offset}."  # module list index in TransformerBlock # pylint: disable=line-too-long
            if non_homogeneous_layers:
                sharded_prefix = f"{layer_prefix}{global_layer_offset}."
                sharded_pp_offset = []
            else:
                sharded_prefix = layer_prefix
                sharded_pp_offset = [
                    (0, global_layer_offset, num_layers)
                ]  # PP sharding offset for ShardedTensors
            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )
            replace_prefix_for_sharding(
                layer_sharded_state_dict, state_dict_prefix, sharded_prefix
            )

            sharded_state_dict.update(layer_sharded_state_dict)

        # Add modules other than self.layers
        for name, module in self.named_children():
            if not module is self.layers:
                sharded_state_dict.update(
                    sharded_state_dict_default(
                        module, f"{prefix}{name}.", sharded_offsets, metadata
                    )
                )

        return sharded_state_dict


def get_wan_spec() -> ModuleSpec:
    return ModuleSpec(
        module=WanDiTLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=WanSelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    q_layernorm=RMSNorm,
                    k_layernorm=RMSNorm,
                    linear_proj=RowParallelLinear,
                ),
            ),
            cross_attention=ModuleSpec(
                module=WanCrossAttention,
                submodules=WanCrossAttentionSubmodules(
                    linear_q=ColumnParallelLinear,
                    linear_k=ColumnParallelLinear,
                    linear_v=ColumnParallelLinear,
                    k_img=ColumnParallelLinear,
                    v_img=ColumnParallelLinear,
                    q_layernorm=RMSNorm,
                    k_layernorm=RMSNorm,
                    core_attention=TEDotProductAttention,
                    linear_proj=RowParallelLinear,
                    norm_k_img=RMSNorm,
                ),
            ),
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear,
                    # dropout？dropout=0
                    linear_fc2=RowParallelLinear,
                ),
            ),
        ),
    )

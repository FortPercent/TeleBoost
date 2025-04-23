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
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import make_viewless_tensor

from teletron.models.dit.dit_attention import (
    FluxSingleAttention,
    JointSelfAttention,
    JointSelfAttentionSubmodules,
    JointHunyuanAttention,
    HunyuanSingleAttention,
)

from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
)



@dataclass
class DiTWithAdaLNSubmodules(TransformerLayerSubmodules):
    temporal_self_attention: Union[ModuleSpec, type] = IdentityOp
    full_self_attention: Union[ModuleSpec, type] = IdentityOp


@dataclass
class STDiTWithAdaLNSubmodules(TransformerLayerSubmodules):
    spatial_self_attention: Union[ModuleSpec, type] = IdentityOp
    temporal_self_attention: Union[ModuleSpec, type] = IdentityOp
    full_self_attention: Union[ModuleSpec, type] = IdentityOp


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


class AdaLN(MegatronModule):
    """
    Adaptive Layer Normalization Module for DiT.
    """

    def __init__(
        self,
        config: TransformerConfig,
        n_adaln_chunks=9,
        norm=nn.LayerNorm,
        modulation_bias=False,
        use_second_norm=False,
    ):
        super().__init__(config)
        if norm == TENorm:
            self.ln = norm(config, config.hidden_size, config.layernorm_epsilon)
        else:
            self.ln = norm(config.hidden_size, elementwise_affine=False, eps=self.config.layernorm_epsilon)
        self.n_adaln_chunks = n_adaln_chunks
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(config.hidden_size, self.n_adaln_chunks * config.hidden_size, bias=modulation_bias)
        )
        self.use_second_norm = use_second_norm
        if self.use_second_norm:
            self.ln2 = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)

        setattr(self.adaLN_modulation[-1].weight, "sequence_parallel", config.sequence_parallel)

    def forward(self, timestep_emb):
        return self.adaLN_modulation(timestep_emb).chunk(self.n_adaln_chunks, dim=-1)

    # @jit_fuser
    def modulate(self, x, shift, scale):
        return x * (1 + scale) + shift

    # @jit_fuser
    def scale_add(self, residual, x, gate):
        return residual + gate * x

    # @jit_fuser
    def modulated_layernorm(self, x, shift, scale, layernorm_idx=0):
        if self.use_second_norm and layernorm_idx == 1:
            layernorm = self.ln2
        else:
            layernorm = self.ln
        # Optional Input Layer norm
        input_layernorm_output = layernorm(x).type_as(x)

        # DiT block specific
        return self.modulate(input_layernorm_output, shift, scale)

    # @jit_fuser
    def scaled_modulated_layernorm(self, residual, x, gate, shift, scale, layernorm_idx=0):
        hidden_states = self.scale_add(residual, x, gate)
        shifted_pre_mlp_layernorm_output = self.modulated_layernorm(hidden_states, shift, scale, layernorm_idx)
        return hidden_states, shifted_pre_mlp_layernorm_output


class AdaLNContinuous(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        conditioning_embedding_dim: int,
        modulation_bias: bool = True,
        norm_type: str = "layer_norm",
    ):
        super().__init__(config)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(conditioning_embedding_dim, config.hidden_size * 2, bias=modulation_bias)
        )
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6, bias=modulation_bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(config.hidden_size, eps=1e-6)
        else:
            raise ValueError("Unknown normalization type {}".format(norm_type))

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.adaLN_modulation(conditioning_embedding)
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class STDiTLayerWithAdaLN(TransformerLayer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.

    Spatial-Temporal DiT with Adapative Layer Normalization.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
        position_embedding_type: Literal["learned_absolute", "rope"] = "learned_absolute",
    ):
        def _replace_no_cp_submodules(submodules):
            modified_submods = copy.deepcopy(submodules)
            modified_submods.cross_attention = IdentityOp
            modified_submods.spatial_self_attention = IdentityOp
            return modified_submods

        # Replace any submodules that will have CP disabled and build them manually later after TransformerLayer init.
        modified_submods = _replace_no_cp_submodules(submodules)
        super().__init__(
            config=config, submodules=modified_submods, layer_number=layer_number, hidden_dropout=hidden_dropout
        )

        # Override Spatial Self Attention and Cross Attention to disable CP.
        # Disable TP Comm overlap as well. Not disabling will attempt re-use of buffer size same as Q and lead to incorrect tensor shapes.
        sa_cp_override_config = copy.deepcopy(config)
        sa_cp_override_config.context_parallel_size = 1
        sa_cp_override_config.tp_comm_overlap = False
        self.spatial_self_attention = build_module(
            submodules.spatial_self_attention, config=sa_cp_override_config, layer_number=layer_number
        )
        self.cross_attention = build_module(
            submodules.cross_attention,
            config=sa_cp_override_config,
            layer_number=layer_number,
        )

        self.temporal_self_attention = build_module(
            submodules.temporal_self_attention,
            config=self.config,
            layer_number=layer_number,
        )

        self.full_self_attention = build_module(
            submodules.full_self_attention,
            config=self.config,
            layer_number=layer_number,
        )

        self.adaLN = AdaLN(config=self.config, n_adaln_chunks=3)

    def forward(
        self,
        hidden_states,
        attention_mask,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        inference_params=None,
        packed_seq_params=None,
    ):
        # timestep embedding
        timestep_emb = attention_mask

        # ******************************************** spatial self attention ******************************************************

        shift_sa, scale_sa, gate_sa = self.adaLN(timestep_emb)

        # adaLN with scale + shift
        pre_spatial_attn_layernorm_output_ada = self.adaLN.modulated_layernorm(
            hidden_states, shift=shift_sa, scale=scale_sa
        )

        attention_output, _ = self.spatial_self_attention(
            pre_spatial_attn_layernorm_output_ada,
            attention_mask=None,
            # packed_seq_params=packed_seq_params['self_attention'],
        )

        # ******************************************** full self attention *************************************************

        shift_full, scale_full, gate_full = self.adaLN(timestep_emb)

        # adaLN with scale + shift
        hidden_states, pre_full_attn_layernorm_output_ada = self.adaLN.scaled_modulated_layernorm(
            residual=hidden_states,
            x=attention_output,
            gate=gate_sa,
            shift=shift_full,
            scale=scale_full,
        )

        attention_output, _ = self.full_self_attention(
            pre_full_attn_layernorm_output_ada,
            attention_mask=None,
            # packed_seq_params=packed_seq_params['self_attention'],
        )

        # ******************************************** cross attention *****************************************************

        shift_ca, scale_ca, gate_ca = self.adaLN(timestep_emb)

        # adaLN with scale + shift
        hidden_states, pre_cross_attn_layernorm_output_ada = self.adaLN.scaled_modulated_layernorm(
            residual=hidden_states,
            x=attention_output,
            gate=gate_full,
            shift=shift_ca,
            scale=scale_ca,
        )

        attention_output, _ = self.cross_attention(
            pre_cross_attn_layernorm_output_ada,
            attention_mask=context_mask,
            key_value_states=context,
            # packed_seq_params=packed_seq_params['cross_attention'],
        )

        # ******************************************** temporal self attention *********************************************

        shift_ta, scale_ta, gate_ta = self.adaLN(timestep_emb)

        hidden_states, pre_temporal_attn_layernorm_output_ada = self.adaLN.scaled_modulated_layernorm(
            residual=hidden_states,
            x=attention_output,
            gate=gate_ca,
            shift=shift_ta,
            scale=scale_ta,
        )

        attention_output, _ = self.temporal_self_attention(
            pre_temporal_attn_layernorm_output_ada,
            attention_mask=None,
            # packed_seq_params=packed_seq_params['self_attention'],
        )

        # ******************************************** mlp *****************************************************************

        shift_mlp, scale_mlp, gate_mlp = self.adaLN(timestep_emb)

        hidden_states, pre_mlp_layernorm_output_ada = self.adaLN.scaled_modulated_layernorm(
            residual=hidden_states,
            x=attention_output,
            gate=gate_ta,
            shift=shift_mlp,
            scale=scale_mlp,
        )

        mlp_output, _ = self.mlp(pre_mlp_layernorm_output_ada)
        hidden_states = self.adaLN.scale_add(residual=hidden_states, x=mlp_output, gate=gate_mlp)

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)

        return output, context


class DiTLayerWithAdaLN(TransformerLayer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.

    DiT with Adapative Layer Normalization.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
        position_embedding_type: Literal["learned_absolute", "rope"] = "learned_absolute",
    ):
        def _replace_no_cp_submodules(submodules):
            modified_submods = copy.deepcopy(submodules)
            modified_submods.cross_attention = IdentityOp
            # modified_submods.temporal_self_attention = IdentityOp
            return modified_submods

        # Replace any submodules that will have CP disabled and build them manually later after TransformerLayer init.
        modified_submods = _replace_no_cp_submodules(submodules)
        super().__init__(
            config=config, submodules=modified_submods, layer_number=layer_number, hidden_dropout=hidden_dropout
        )

        # Override Cross Attention to disable CP.
        # Disable TP Comm overlap as well. Not disabling will attempt re-use of buffer size same as Q and lead to incorrect tensor shapes.
        if submodules.cross_attention != IdentityOp:
            cp_override_config = copy.deepcopy(config)
            cp_override_config.context_parallel_size = 1
            cp_override_config.tp_comm_overlap = False
            self.cross_attention = build_module(
                submodules.cross_attention,
                config=cp_override_config,
                layer_number=layer_number,
            )
        else:
            self.cross_attention = None

        self.full_self_attention = build_module(
            submodules.full_self_attention,
            config=self.config,
            layer_number=layer_number,
        )

        self.adaLN = AdaLN(config=self.config, n_adaln_chunks=9 if self.cross_attention else 6)

    def forward(
        self,
        hidden_states,
        attention_mask,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        inference_params=None,
        packed_seq_params=None,
    ):
        # timestep embedding
        timestep_emb = attention_mask

        # ******************************************** full self attention ******************************************************
        if self.cross_attention:
            shift_full, scale_full, gate_full, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN(timestep_emb)
            )
        else:
            shift_full, scale_full, gate_full, shift_mlp, scale_mlp, gate_mlp = self.adaLN(timestep_emb)

        # adaLN with scale + shift
        pre_full_attn_layernorm_output_ada = self.adaLN.modulated_layernorm(
            hidden_states, shift=shift_full, scale=scale_full
        )

        attention_output, _ = self.full_self_attention(
            pre_full_attn_layernorm_output_ada,
            attention_mask=None,
            packed_seq_params=None if packed_seq_params is None else packed_seq_params['self_attention'],
        )

        if self.cross_attention:
            # ******************************************** cross attention ******************************************************
            # adaLN with scale + shift
            hidden_states, pre_cross_attn_layernorm_output_ada = self.adaLN.scaled_modulated_layernorm(
                residual=hidden_states,
                x=attention_output,
                gate=gate_full,
                shift=shift_ca,
                scale=scale_ca,
            )

            attention_output, _ = self.cross_attention(
                pre_cross_attn_layernorm_output_ada,
                attention_mask=context_mask,
                key_value_states=context,
                packed_seq_params=None if packed_seq_params is None else packed_seq_params['cross_attention'],
            )

        # ******************************************** mlp ******************************************************
        hidden_states, pre_mlp_layernorm_output_ada = self.adaLN.scaled_modulated_layernorm(
            residual=hidden_states,
            x=attention_output,
            gate=gate_ca if self.cross_attention else gate_full,
            shift=shift_mlp,
            scale=scale_mlp,
        )

        mlp_output, _ = self.mlp(pre_mlp_layernorm_output_ada)
        hidden_states = self.adaLN.scale_add(residual=hidden_states, x=mlp_output, gate=gate_mlp)

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)

        return output, context


class DiTLayer(TransformerLayer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.

    Original DiT layer implementation from [https://arxiv.org/pdf/2212.09748].
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        mlp_ratio: int = 4,
        n_adaln_chunks: int = 6,
        modulation_bias: bool = True,
    ):
        # Modify the mlp layer hidden_size of a dit layer according to mlp_ratio
        config.ffn_hidden_size = int(mlp_ratio * config.hidden_size)
        super().__init__(config=config, submodules=submodules, layer_number=layer_number)

        self.adaLN = AdaLN(
            config=config, n_adaln_chunks=n_adaln_chunks, modulation_bias=modulation_bias, use_second_norm=True
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        inference_params=None,
        packed_seq_params=None,
    ):
        # passing in conditioning information via attention mask here
        c = attention_mask

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c)

        shifted_input_layernorm_output = self.adaLN.modulated_layernorm(
            hidden_states, shift=shift_msa, scale=scale_msa, layernorm_idx=0
        )

        x, bias = self.self_attention(shifted_input_layernorm_output, attention_mask=None)

        hidden_states = self.adaLN.scale_add(hidden_states, x=(x + bias), gate=gate_msa)

        residual = hidden_states

        shited_pre_mlp_layernorm_output = self.adaLN.modulated_layernorm(
            hidden_states, shift=shift_mlp, scale=scale_mlp, layernorm_idx=1
        )

        x, bias = self.mlp(shited_pre_mlp_layernorm_output)

        hidden_states = self.adaLN.scale_add(residual, x=(x + bias), gate=gate_mlp)

        return hidden_states, context

class HunyuanDiTLayer(TransformerLayer):
    """A double transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.

    HunyuanDiTLayer layer implementation from [https://arxiv.org/pdf/2403.03206].
    """
    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        context_pre_only: bool = False,
    ):
        hidden_size = config.hidden_size
        super().__init__(config=config, submodules=submodules, layer_number=layer_number)

        self.norm1 = AdaLayerNormZero(hidden_size, norm_type="layer_norm")
        self.norm1_context= AdaLayerNormZero(hidden_size, norm_type="layer_norm")
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2_context = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        cp_override_config = copy.deepcopy(config)
        cp_override_config.context_parallel_size = 1
        cp_override_config.tp_comm_overlap = False

        self.ff_context=build_module(
            submodules.mlp,
            config=cp_override_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin:  Optional[torch.Tensor] = None,
    ):
        # 1. Input normalization
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            self.norm1_context(encoder_hidden_states, emb=temb)
        )

        # 2. Joint attention
        attn_output, context_attn_output = self.self_attention(
            # hidden_states=norm_hidden_states,
            # additional_hidden_states=norm_encoder_hidden_states,
            # attention_mask=attention_mask,
            # rotary_pos_emb=freqs_cis,
            norm_hidden_states, # [2,9604, 3072]
            attention_mask=attention_mask,  
            key_value_states=None,
            additional_hidden_states=norm_encoder_hidden_states,    # [2, 226, 3072]
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
        )

        # #sbd -> bsd
        # attn_output = attn_output.transpose(0, 1)
        # context_attn_output = context_attn_output.transpose(0, 1)

        # 3. Modulation and residual connection
        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)

        encoder_hidden_states = (
            encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)
        )

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None])
            + c_shift_mlp[:, None]
        )

        norm_hidden_states = norm_hidden_states.transpose(0, 1)
        norm_encoder_hidden_states = norm_encoder_hidden_states.transpose(0, 1)

        # 4. Feed-forward
        ff_output = self.mlp(norm_hidden_states)

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        if len(ff_output)==2: 
            ff=(ff_output[0]+ff_output[1]).transpose(0, 1)
        if len(context_ff_output)==2: 
            context_ff=(context_ff_output[0]+context_ff_output[1]).transpose(0, 1)

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff
        encoder_hidden_states = (
            encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff
        )

        return hidden_states, encoder_hidden_states
    

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: dict = None
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
            'non_homogeneous_layers', False
        )
        if self.config.num_moe_experts is not None:
            non_homogeneous_layers = True

        sharded_state_dict = {}

        layer_prefix = f'{prefix}layers.'
        num_layers = self.config.num_layers
        for layer in self.layers:
            offset = TransformerLayer._get_layer_offset(self.config)

            global_layer_offset = layer.layer_number - 1  # self.layer_number starts at 1
            state_dict_prefix = f'{layer_prefix}{global_layer_offset - offset}.'  # module list index in TransformerBlock # pylint: disable=line-too-long
            if non_homogeneous_layers:
                sharded_prefix = f'{layer_prefix}{global_layer_offset}.'
                sharded_pp_offset = []
            else:
                sharded_prefix = layer_prefix
                sharded_pp_offset = [
                    (0, global_layer_offset, num_layers)
                ]  # PP sharding offset for ShardedTensors
            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)

            sharded_state_dict.update(layer_sharded_state_dict)

        # Add modules other than self.layers
        for name, module in self.named_children():
            if not module is self.layers:
                sharded_state_dict.update(
                    sharded_state_dict_default(
                        module, f'{prefix}{name}.', sharded_offsets, metadata
                    )
                )

        return sharded_state_dict


class MMDiTLayer(TransformerLayer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.

    MMDiT layer implementation from [https://arxiv.org/pdf/2403.03206].
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        context_pre_only: bool = False,
    ):

        hidden_size = config.hidden_size
        super().__init__(config=config, submodules=submodules, layer_number=layer_number)

        self.adaln = AdaLN(config, modulation_bias=True, n_adaln_chunks=6, use_second_norm=True)

        self.context_pre_only = context_pre_only
        context_norm_type = "ada_norm_continous" if context_pre_only else "ada_norm_zero"

        if context_norm_type == "ada_norm_continous":
            self.adaln_context = AdaLNContinous(config, hidden_size, modulation_bias=True, norm_type="layer_norm")
        elif context_norm_type == "ada_norm_zero":
            self.adaln_context = AdaLN(config, modulation_bias=True, n_adaln_chunks=6, use_second_norm=True)
        else:
            raise ValueError(
                f"Unknown context_norm_type: {context_norm_type}, currently only support `ada_norm_continous`, `ada_norm_zero`"
            )
        # Override Cross Attention to disable CP.
        # Disable TP Comm overlap as well. Not disabling will attempt re-use of buffer size same as Q and lead to incorrect tensor shapes.
        cp_override_config = copy.deepcopy(config)
        cp_override_config.context_parallel_size = 1
        cp_override_config.tp_comm_overlap = False

        if not context_pre_only:
            self.context_mlp = build_module(
                submodules.mlp,
                config=cp_override_config,
            )
        else:
            self.context_mlp = None

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        inference_params=None,
        packed_seq_params=None,
        emb=None,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(emb)

        norm_hidden_states = self.adaln.modulated_layernorm(
            hidden_states, shift=shift_msa, scale=scale_msa, layernorm_idx=0
        )
        if self.context_pre_only:
            norm_encoder_hidden_states = self.adaln_context(encoder_hidden_states, emb)
        else:
            c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.adaln_context(emb)
            norm_encoder_hidden_states = self.adaln_context.modulated_layernorm(
                encoder_hidden_states, shift=c_shift_msa, scale=c_scale_msa, layernorm_idx=0
            )

        attention_output, encoder_attention_output = self.self_attention(
            norm_hidden_states,
            attention_mask=attention_mask,
            key_value_states=None,
            additional_hidden_states=norm_encoder_hidden_states,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = self.adaln.scale_add(hidden_states, x=attention_output, gate=gate_msa)
        norm_hidden_states = self.adaln.modulated_layernorm(
            hidden_states, shift=shift_mlp, scale=scale_mlp, layernorm_idx=1
        )

        mlp_output, mlp_output_bias = self.mlp(norm_hidden_states)
        hidden_states = self.adaln.scale_add(hidden_states, x=(mlp_output + mlp_output_bias), gate=gate_mlp)

        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            encoder_hidden_states = self.adaln_context.scale_add(
                encoder_hidden_states, x=encoder_attention_output, gate=c_gate_msa
            )
            norm_encoder_hidden_states = self.adaln_context.modulated_layernorm(
                encoder_hidden_states, shift=c_shift_mlp, scale=c_scale_mlp, layernorm_idx=1
            )

            context_mlp_output, context_mlp_output_bias = self.context_mlp(norm_encoder_hidden_states)
            encoder_hidden_states = self.adaln.scale_add(
                encoder_hidden_states, x=(context_mlp_output + context_mlp_output_bias), gate=c_gate_mlp
            )

        return hidden_states, encoder_hidden_states

class HunyuanSingleDiTLayer(TransformerLayer):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        mlp_ratio: int = 4,
        n_adaln_chunks: int = 3,
        modulation_bias: bool = True,
    ):
        super().__init__(config=config, submodules=submodules, layer_number=layer_number)
        hidden_size = config.hidden_size

        self.norm = AdaLayerNormZeroSingle(hidden_size, norm_type="layer_norm")
        self.mlp_hidden_dim=hidden_size*mlp_ratio
        # self.proj_mlp=nn.Linear(hidden_size, self.mlp_hidden_dim)
        self.proj_mlp=TEColumnParallelLinear(
                hidden_size,
                self.mlp_hidden_dim,
                config=config,
                gather_output=False,
                init_method=config.init_method,
                bias=True,
                skip_bias_add=False,
                is_expert=False)
        
        self.act_mlp=nn.GELU(approximate="tanh")
        # self.proj_out =nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        self.proj_out=TEColumnParallelLinear(
                hidden_size + self.mlp_hidden_dim,
                hidden_size,
                config=config,
                gather_output=False,
                init_method=config.init_method,
                bias=True,
                skip_bias_add=False,
                is_expert=False)
    
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        residual = hidden_states

        # 1. Input normalization
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        x = self.proj_mlp(norm_hidden_states)
        # print(x[0])
        # mlp_hidden_states = self.act_mlp(x)
        mlp_hidden_states = self.act_mlp(x[0])
        mlp_hidden_states = gather_from_tensor_model_parallel_region(mlp_hidden_states)

        norm_hidden_states, norm_encoder_hidden_states = (
            norm_hidden_states[:, :-text_seq_length, :],
            norm_hidden_states[:, -text_seq_length:, :],
        )

        # 2. Attention
        attn_output, context_attn_output = self.self_attention(
            hidden_states=norm_hidden_states,
            additional_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
        )
        #sbd -> bsd
        # attn_output = attn_output.transpose(0, 1)
        # context_attn_output = context_attn_output.transpose(0, 1)
        attn_output = torch.cat([attn_output, context_attn_output], dim=1)
        output = gather_from_tensor_model_parallel_region(attn_output)
        # 3. Modulation and residual connection
        
        hidden_states = torch.cat([output, mlp_hidden_states], dim=2)

        x = self.proj_out(hidden_states)
        # print(f"x:{x[0].shape}")
        x = gather_from_tensor_model_parallel_region(x[0])

        hidden_states = gate.unsqueeze(1) * x
        hidden_states = hidden_states + residual

        hidden_states, encoder_hidden_states = (
            hidden_states[:, :-text_seq_length, :],
            hidden_states[:, -text_seq_length:, :],
        )
        return hidden_states, encoder_hidden_states

    
    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: dict = None
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
            'non_homogeneous_layers', False
        )
        if self.config.num_moe_experts is not None:
            non_homogeneous_layers = True

        sharded_state_dict = {}

        layer_prefix = f'{prefix}layers.'
        num_layers = self.config.num_layers
        for layer in self.layers:
            offset = TransformerLayer._get_layer_offset(self.config)

            global_layer_offset = layer.layer_number - 1  # self.layer_number starts at 1
            state_dict_prefix = f'{layer_prefix}{global_layer_offset - offset}.'  # module list index in TransformerBlock # pylint: disable=line-too-long
            if non_homogeneous_layers:
                sharded_prefix = f'{layer_prefix}{global_layer_offset}.'
                sharded_pp_offset = []
            else:
                sharded_prefix = layer_prefix
                sharded_pp_offset = [
                    (0, global_layer_offset, num_layers)
                ]  # PP sharding offset for ShardedTensors
            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)

            sharded_state_dict.update(layer_sharded_state_dict)

        # Add modules other than self.layers
        for name, module in self.named_children():
            if not module is self.layers:
                sharded_state_dict.update(
                    sharded_state_dict_default(
                        module, f'{prefix}{name}.', sharded_offsets, metadata
                    )
                )

        return sharded_state_dict
       
class FluxSingleTransformerBlock(TransformerLayer):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        mlp_ratio: int = 4,
        n_adaln_chunks: int = 3,
        modulation_bias: bool = True,
    ):
        super().__init__(config=config, submodules=submodules, layer_number=layer_number)
        hidden_size = config.hidden_size
        self.adaln = AdaLN(
            config=config, n_adaln_chunks=n_adaln_chunks, modulation_bias=modulation_bias, use_second_norm=False
        )
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.proj_in = nn.Linear(hidden_size, self.mlp_hidden_dim)
        self.activation = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        inference_params=None,
        packed_seq_params=None,
        emb=None,
    ):
        residual = hidden_states

        shift, scale, gate = self.adaln(emb)

        norm_hidden_states = self.adaln.modulated_layernorm(hidden_states, shift=shift, scale=scale)

        mlp_hidden_states = self.activation(self.proj_in(norm_hidden_states))

        attention_output = self.self_attention(
            norm_hidden_states, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb
        )

        hidden_states = torch.cat((attention_output, mlp_hidden_states), dim=2)

        hidden_states = self.proj_out(hidden_states)

        hidden_states = self.adaln.scale_add(residual, x=hidden_states, gate=gate)

        return hidden_states


def get_stdit_adaln_block_with_transformer_engine_spec() -> ModuleSpec:
    params = {"attn_mask_type": AttnMaskType.padding}
    return ModuleSpec(
        module=STDiTLayerWithAdaLN,
        submodules=STDiTWithAdaLNSubmodules(
            spatial_self_attention=ModuleSpec(
                module=SelfAttention,
                params=params,
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                ),
            ),
            temporal_self_attention=ModuleSpec(
                module=SelfAttention,
                params=params,
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                ),
            ),
            full_self_attention=ModuleSpec(
                module=SelfAttention,
                params=params,
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                ),
            ),
            cross_attention=ModuleSpec(
                module=CrossAttention,
                params=params,
                submodules=CrossAttentionSubmodules(
                    linear_q=TEColumnParallelLinear,
                    linear_kv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                ),
            ),
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
        ),
    )



def get_dit_adaln_block_with_transformer_engine_spec(attn_mask_type=AttnMaskType.padding) -> ModuleSpec:
    params = {"attn_mask_type": attn_mask_type}
    return ModuleSpec(
        module=DiTLayerWithAdaLN,
        submodules=DiTWithAdaLNSubmodules(
            full_self_attention=ModuleSpec(
                module=SelfAttention,
                params=params,
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=RMSNorm,
                    k_layernorm=RMSNorm,
                ),
            ),
            cross_attention=ModuleSpec(
                module=CrossAttention,
                params=params,
                submodules=CrossAttentionSubmodules(
                    linear_q=TEColumnParallelLinear,
                    linear_kv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=RMSNorm,
                    k_layernorm=RMSNorm,
                ),
            ),
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
        ),
    )


def get_official_dit_adaln_block_with_transformer_engine_spec() -> ModuleSpec:
    params = {"attn_mask_type": AttnMaskType.no_mask}
    return ModuleSpec(
        module=DiTLayerWithAdaLN,
        submodules=DiTWithAdaLNSubmodules(
            full_self_attention=ModuleSpec(
                module=SelfAttention,
                params=params,
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                ),
            ),
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
        ),
    )


def get_mm_dit_block_with_transformer_engine_spec() -> ModuleSpec:

    return ModuleSpec(
        module=MMDiTLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=JointSelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=JointSelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    added_linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                ),
            ),
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
        ),
    )


def get_flux_single_transformer_engine_spec() -> ModuleSpec:
    return ModuleSpec(
        module=FluxSingleTransformerBlock,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=FluxSingleAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    q_layernorm=RMSNorm,
                    k_layernorm=RMSNorm,
                    linear_proj=IdentityOp,
                ),
            ),
        ),
    )


def get_flux_double_transformer_engine_spec() -> ModuleSpec:
    return ModuleSpec(
        module=MMDiTLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=JointSelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=JointSelfAttentionSubmodules(
                    q_layernorm=RMSNorm,
                    k_layernorm=RMSNorm,
                    added_q_layernorm=RMSNorm,
                    added_k_layernorm=RMSNorm,
                    linear_qkv=TEColumnParallelLinear,
                    added_linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                ),
            ),
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
        ),
    )


def get_hunyuan_double_transformer_engine_spec() -> ModuleSpec:
    return ModuleSpec(
        module=HunyuanDiTLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=JointHunyuanAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=JointSelfAttentionSubmodules(
                    q_layernorm=RMSNorm,
                    k_layernorm=RMSNorm,
                    added_q_layernorm=RMSNorm,
                    added_k_layernorm=RMSNorm,
                    linear_qkv=TEColumnParallelLinear,
                    added_linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                ),
            ),
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    #dropout？dropout=0
                    linear_fc2=TERowParallelLinear,
                ),
            ),
        ),
    )

def get_hunyuan_single_transformer_engine_spec() -> ModuleSpec:
   return ModuleSpec(
        module=HunyuanSingleDiTLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=HunyuanSingleAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    q_layernorm=RMSNorm,
                    k_layernorm=RMSNorm,
                    linear_proj=IdentityOp,
                ),
            ),
        ),
    )
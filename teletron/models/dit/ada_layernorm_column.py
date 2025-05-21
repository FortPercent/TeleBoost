
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from megatron.core.transformer.custom_layers.transformer_engine import (TEColumnParallelLinear)
from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
)
from diffusers.models.embeddings import CombinedTimestepLabelEmbeddings
from diffusers.models.normalization import FP32LayerNorm




class AdaColumnLayerNormZeroSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, norm_type="layer_norm", bias=True, config=None):
        super().__init__()

        self.silu = nn.SiLU()
        # self.linear = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)

        self.column_linear = TEColumnParallelLinear(
                input_size=embedding_dim,  
                output_size=3 * embedding_dim,
                config=config,
                gather_output=False,
                init_method=config.init_method,
                bias=bias,
                skip_bias_add=False,
                is_expert=False)
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        silu_res = self.silu(emb)
        
        emb = self.column_linear(silu_res)
        emb = gather_from_tensor_model_parallel_region(emb[0])

        # emb = self.linear(silu_res)

        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa


class AdaColumnLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None, norm_type="layer_norm", bias=True, config=None):
        super().__init__()
        if num_embeddings is not None:
            self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        
        # self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)

        self.column_linear = TEColumnParallelLinear(
                input_size=embedding_dim,  
                output_size=6 * embedding_dim,
                config=config,
                gather_output=False,
                init_method=config.init_method,
                bias=bias,
                skip_bias_add=False,
                is_expert=False)
        
        # print("self.linear.weight.data.shape: ", self.linear.weight.data.shape)
        # print("self.column_linear.weight.data.shape: ", self.column_linear.weight.data.shape)

        # from megatron.core import mpu
        # weight_per_rank = int(self.linear.weight.data.shape[0] /  mpu.get_tensor_model_parallel_world_size())
        # tp_rank = mpu.get_tensor_model_parallel_rank()
        # print(f"weight_per_rank: {weight_per_rank}")
        # print(f"tp_rank: {tp_rank}")
        # self.column_linear.weight.data = self.linear.weight.data[tp_rank * weight_per_rank : (tp_rank + 1) * weight_per_rank][:]
        # self.column_linear.bias.data = self.linear.bias.data[tp_rank * weight_per_rank : (tp_rank + 1) * weight_per_rank][:]

        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        elif norm_type == "fp32_layer_norm":
            self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=False, bias=False)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        silu_res = self.silu(emb)

        # emb = self.linear(silu_res)
        
        emb= self.column_linear(silu_res)
        emb = gather_from_tensor_model_parallel_region(emb[0])

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


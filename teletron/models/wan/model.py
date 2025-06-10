import os
from dataclasses import dataclass
from megatron.core.transformer.utils import openai_gelu
from typing import Callable, Any, Dict, List, Optional, Tuple, Union
import torch.nn.functional as F
import torch
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import get_args
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from teletron.core.context_parallel.pad import pad_for_context_parallel, remove_pad_for_context_parallel, set_origin_length, set_target_length
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from teletron.models.dit.wan_layerspec import WanDiTLayer, get_wan_spec

from teletron.models.wan.module import (
    WanTimeTextImageEmbedding,
    WanRotaryPosEmbed,
    MLP,
)

from teletron.core.context_parallel.pad import pad_for_context_parallel, remove_pad_for_context_parallel, set_origin_length, set_target_length

from torch import nn
import math
from diffusers.utils import logging

from einops import rearrange

from megatron.core import mpu, tensor_parallel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class WanParams:
    patch_size: Tuple[int] = (1, 2, 2)
    num_attention_heads: int = 40
    attention_head_dim: int = 128
    activation_func: Callable = F.gelu
    in_channels: int = 36
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 13824
    num_layers: int = 30
    cross_attn_norm: bool = True
    qk_norm: Optional[str] = "rms_norm_across_heads"
    eps: float = 1e-6
    image_dim: int = 1280
    added_kv_proj_dim: int = 5120
    rope_max_seq_len: int = 1024
    has_image_pos_emb: bool = False


class WanVideoTransformer3DModel(VisionModule):
    def __init__(self, wan_config: WanParams, config: TransformerConfig):
        self.out_channels = wan_config.out_channels
        self.in_channels = wan_config.in_channels
        self.num_attention_heads = wan_config.num_attention_heads
        self.attention_head_dim = wan_config.attention_head_dim
        self.patch_size = wan_config.patch_size
        self.text_dim = wan_config.text_dim
        self.freq_dim = wan_config.freq_dim
        self.ffn_dim = wan_config.ffn_dim
        self.cross_attn_norm = wan_config.cross_attn_norm
        self.qk_norm = wan_config.qk_norm
        self.eps = wan_config.eps
        self.rope_max_seq_len = wan_config.rope_max_seq_len
        self.patch_size = wan_config.patch_size
        self.image_dim = wan_config.image_dim
        self.added_kv_proj_dim = wan_config.added_kv_proj_dim
        self.num_layers = wan_config.num_layers
        self.has_image_pos_emb = wan_config.has_image_pos_emb

        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        config.hidden_size = self.hidden_size
        config.ffn_hidden_size = self.ffn_dim
        config.num_attention_heads = self.num_attention_heads
        config.num_layers = self.num_layers
        # TODO: Assume not use GQA
        config.num_query_groups = config.num_attention_heads
        # config.use_cpu_initialization = True
        config.hidden_dropout = 0
        config.attention_dropout = 0
        config.layernorm_epsilon = 1e-6
        config.rotary_interleaved = True
        config.attention_dropout = (
            config.attention_dropout[0]
            if isinstance(config.attention_dropout, tuple)
            else config.attention_dropout
        )
        config.has_image_pos_emb = self.has_image_pos_emb
        transformer_config = config

        super().__init__(transformer_config)
        self.inner_dim = self.num_attention_heads * self.attention_head_dim

        out_channels = self.out_channels or self.in_channels

        # 1. Patch & position embedding
        # self.rope = WanRotaryPosEmbed(self.attention_head_dim, self.patch_size, self.rope_max_seq_len)

        self.patch_embedding = nn.Conv3d(
            self.in_channels,
            self.inner_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # 2. Condition embeddings
        # # image_embedding_dim=1280 for I2V model
        # self.condition_embedder = WanTimeTextImageEmbedding(
        #     dim=self.inner_dim,
        #     time_freq_dim=self.freq_dim,
        #     time_proj_dim=self.inner_dim * 6,
        #     text_embed_dim=self.text_dim,
        #     image_embed_dim=self.image_dim,
        # )

        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.inner_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.inner_dim, self.inner_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.inner_dim),
            nn.SiLU(),
            nn.Linear(self.inner_dim, self.inner_dim),
        )


        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(self.inner_dim, self.inner_dim * 6)
        )

        # if self.has_image_input: TODO, align
        self.img_emb = MLP(self.image_dim, self.inner_dim, has_image_pos_emb=self.has_image_pos_emb)
        self.freqs = precompute_freqs_cis_3d(self.attention_head_dim)

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanDiTLayer(
                    config=transformer_config,
                    submodules=get_wan_spec().submodules,
                    layer_number=i,
                )
                for i in range(self.num_layers)
            ]
        )

        # 4. Output norm & projection

        self.norm_out = FP32LayerNorm(self.inner_dim, eps=self.eps, elementwise_affine=False)
        self.proj_out = nn.Linear(self.inner_dim, out_channels * math.prod(self.patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, self.inner_dim) / self.inner_dim**0.5)


        print("Wan3DModel Init Finish!")

    def _get_block(self, layer_number: int):
        return self.blocks[layer_number]

    def _checkpointed_forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, *args
    ):
        "Forward method with activation checkpointing."
        recompute_layers = self.num_layers

        def custom(start, end):
            def custom_forward(*args):
                for index in range(start, end):
                    layer = self._get_block(index)
                    x_ = layer(*args)
                return x_

            return custom_forward

        if self.config.recompute_method == "uniform":
            # Uniformly divide the total number of Transformer layers and
            # checkpoint the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            _layer_num = 0
            while _layer_num < self.num_layers:
                hidden_states = tensor_parallel.checkpoint(
                    custom(_layer_num, _layer_num + recompute_layers),
                    self.config.distribute_saved_activations,
                    hidden_states,
                    encoder_hidden_states,
                    *args
                )
                _layer_num += recompute_layers
        elif self.config.recompute_method == "block":
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            if os.environ.get("PROFILE_MEMORY"):
                from my_utils import ProfilerWrapper

                profiler = ProfilerWrapper(is_st=False, enable_record_cuda_mm=True)
            for _layer_num in range(recompute_layers):
                if _layer_num < recompute_layers:
                    hidden_states = tensor_parallel.checkpoint(
                        custom(_layer_num, _layer_num + 1),
                        self.config.distribute_saved_activations,
                        hidden_states,
                        encoder_hidden_states,
                        *args
                    )
                else:
                    block = self._get_block(_layer_num)
                    hidden_states = block(*hidden_states, *encoder_hidden_states, *args)
                if os.environ.get("PROFILE_MEMORY"):
                    profiler.record()

        return hidden_states

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.LongTensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        cn_images=None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))

        t_mod = self.time_projection(t).unflatten(1, (6, -1))

        context = self.text_embedding(context)

        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        clip_embdding = self.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

        if cn_images is not None:
            x = torch.cat([x, cn_images], dim=1)  # (b, c_x + c_y, f, h, w)

        x, (f, h, w) = self.patchify(x)

        freqs = (
            torch.cat(
                [
                    self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )

        # if encoder_hidden_states_image is not None:
        #     encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)
        if mpu.get_context_parallel_world_size() > 1:
            length = x.shape[1]
            set_origin_length(length)
            seq_parallel_world_size = mpu.get_context_parallel_world_size()
            if length % seq_parallel_world_size != 0:
                pad_size = seq_parallel_world_size - (length % seq_parallel_world_size)
                length = length + pad_size
            set_target_length(length)
            x = pad_for_context_parallel(x, 1)
            # import pdb; pdb.set_trace()
            # freqs_cos,freqs_sin=freqs
            # freqs_cos = pad_for_context_parallel(freqs_cos, 0)
            # freqs_sin = pad_for_context_parallel(freqs_sin, 0)
            freqs = pad_for_context_parallel(freqs, 0)
            # freqs=(freqs_cos,freqs_sin)

            from teletron.core.tensor_parallel.mappings import (
                split_forward_gather_backward,
                gather_forward_split_backward,
            )

            x = split_forward_gather_backward(
                x, mpu.get_context_parallel_group(), dim=1, grad_scale="down"
            )  # b s n d

            freqs = split_forward_gather_backward(
                freqs, mpu.get_context_parallel_group(), dim=0, grad_scale="down"
            )
        # 4. Transformer blocks
        if self.config.recompute_granularity == "full":
            hidden_states = self._checkpointed_forward(x, context, t_mod, freqs)
        else:
            if os.environ.get("PROFILE_MEMORY"):
                from my_utils import ProfilerWrapper

                profiler = ProfilerWrapper(is_st=False, enable_record_cuda_mm=True)

            for block in self.blocks:
                hidden_states = block(x, context, t_mod, freqs)
                if os.environ.get("PROFILE_MEMORY"):
                    profiler.record()

        if mpu.get_context_parallel_world_size() > 1:
            hidden_states = gather_forward_split_backward(
                hidden_states, mpu.get_context_parallel_group(), dim=1, grad_scale="up"
            )
            hidden_states = remove_pad_for_context_parallel(hidden_states, 1)

        shift, scale = (self.scale_shift_table.to(dtype=t.dtype, device=t.device) + t).chunk(2, dim=1)
        norm_out_temp = self.norm_out(hidden_states.float())
        #norm_out_temp = torch.ones_like(norm_out_temp)
        hidden_states = (self.proj_out((norm_out_temp * (1 + scale) + shift).bfloat16()))


        hidden_states = self.unpatchify(hidden_states, (f, h, w))
        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(
                dim // 2
            ),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)

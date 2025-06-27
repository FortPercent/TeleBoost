import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False

T5_CONTEXT_TOKEN_NUMBER = 512


class VastParams:
    hidden_size: int = 5120
    in_channels: int = 36
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 13824
    eps: float = 1e-6
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    num_attention_heads: int = 40
    num_layers: int = 3
    has_image_input: bool = True
    has_image_pos_emb: bool = False


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, hidden_dimension, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.scale_parameter = nn.Parameter(torch.ones(hidden_dimension))

    def normalize(self, input_tensor):
        return input_tensor * torch.rsqrt(input_tensor.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)

    def forward(self, input_tensor):
        original_dtype = input_tensor.dtype
        return self.normalize(input_tensor.float()).to(original_dtype) * self.scale_parameter


class AttentionModule(nn.Module):
    def __init__(self, attention_head_count):
        super().__init__()
        self.attention_head_count = attention_head_count

    def forward(self, query_tensor, key_tensor, value_tensor):
        output_tensor = flash_attention(
            q=query_tensor, k=key_tensor, v=value_tensor, num_heads=self.attention_head_count)
        return output_tensor


class SelfAttention(nn.Module):
    def __init__(self, hidden_dimension: int, attention_head_count: int, epsilon: float = 1e-6):
        super().__init__()
        self.hidden_dimension = hidden_dimension
        self.attention_head_count = attention_head_count
        self.attention_head_dimension = hidden_dimension // attention_head_count

        self.query_projection = nn.Linear(hidden_dimension, hidden_dimension)
        self.key_projection = nn.Linear(hidden_dimension, hidden_dimension)
        self.value_projection = nn.Linear(hidden_dimension, hidden_dimension)
        self.output_projection = nn.Linear(hidden_dimension, hidden_dimension)
        self.query_normalization = RMSNorm(hidden_dimension, eps=epsilon)
        self.key_normalization = RMSNorm(hidden_dimension, eps=epsilon)

        self.attention_mechanism = AttentionModule(self.attention_head_count)

    def forward(self, input_tensor, rotary_frequencies):
        query_tensor = self.query_normalization(
            self.query_projection(input_tensor))
        key_tensor = self.key_normalization(self.key_projection(input_tensor))
        value_tensor = self.value_projection(input_tensor)
        query_tensor = rope_apply(
            query_tensor, rotary_frequencies, self.attention_head_count)
        key_tensor = rope_apply(
            key_tensor, rotary_frequencies, self.attention_head_count)
        attention_output = self.attention_mechanism(
            query_tensor, key_tensor, value_tensor)
        return self.output_projection(attention_output)


class CrossAttention(nn.Module):
    def __init__(self, hidden_dimension: int, attention_head_count: int, epsilon: float = 1e-6, supports_image_input: bool = False):
        super().__init__()
        self.hidden_dimension = hidden_dimension
        self.attention_head_count = attention_head_count
        self.attention_head_dimension = hidden_dimension // attention_head_count

        self.query_projection = nn.Linear(hidden_dimension, hidden_dimension)
        self.key_projection = nn.Linear(hidden_dimension, hidden_dimension)
        self.value_projection = nn.Linear(hidden_dimension, hidden_dimension)
        self.output_projection = nn.Linear(hidden_dimension, hidden_dimension)
        self.query_normalization = RMSNorm(hidden_dimension, eps=epsilon)
        self.key_normalization = RMSNorm(hidden_dimension, eps=epsilon)
        self.supports_image_input = supports_image_input
        if supports_image_input:
            self.image_key_projection = nn.Linear(
                hidden_dimension, hidden_dimension)
            self.image_value_projection = nn.Linear(
                hidden_dimension, hidden_dimension)
            self.image_key_normalization = RMSNorm(
                hidden_dimension, eps=epsilon)

        self.primary_attention = AttentionModule(self.attention_head_count)
        self.secondary_attention = AttentionModule(self.attention_head_count)

    def forward(self, input_tensor: torch.Tensor, context_tensor: torch.Tensor):
        if self.supports_image_input:
            image_context_length = context_tensor.shape[1] - \
                T5_CONTEXT_TOKEN_NUMBER
            image_features = context_tensor[:, :image_context_length]
            text_context = context_tensor[:, image_context_length:]
        else:
            text_context = context_tensor

        query_tensor = self.query_normalization(
            self.query_projection(input_tensor))
        key_tensor = self.key_normalization(self.key_projection(text_context))
        value_tensor = self.value_projection(text_context)
        attention_output = self.primary_attention(
            query_tensor, key_tensor, value_tensor)
        if self.supports_image_input:
            image_key_tensor = self.image_key_normalization(
                self.image_key_projection(image_features))
            image_value_tensor = self.image_value_projection(image_features)
            image_attention_output = self.secondary_attention(
                query_tensor, image_key_tensor, image_value_tensor)
            attention_output = attention_output + image_attention_output
        return self.output_projection(attention_output)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, input_tensor, gate_tensor, residual_tensor):
        return input_tensor + gate_tensor * residual_tensor


class DiTBlock(nn.Module):
    def __init__(self, supports_image_input: bool, hidden_dimension: int, attention_head_count: int, feedforward_dimension: int, epsilon: float = 1e-6):
        super().__init__()
        self.hidden_dimension = hidden_dimension
        self.attention_head_count = attention_head_count
        self.feedforward_dimension = feedforward_dimension

        self.self_attention_layer = SelfAttention(
            hidden_dimension, attention_head_count, epsilon)
        self.cross_attention_layer = CrossAttention(
            hidden_dimension, attention_head_count, epsilon, has_image_input=supports_image_input)
        self.pre_attention_normalization = nn.LayerNorm(
            hidden_dimension, eps=epsilon, elementwise_affine=False)
        self.pre_feedforward_normalization = nn.LayerNorm(
            hidden_dimension, eps=epsilon, elementwise_affine=False)
        self.pre_cross_attention_normalization = nn.LayerNorm(
            hidden_dimension, eps=epsilon)
        self.feedforward_network = nn.Sequential(nn.Linear(hidden_dimension, feedforward_dimension), nn.GELU(
            approximate='tanh'), nn.Linear(feedforward_dimension, hidden_dimension))
        self.modulation_parameters = nn.Parameter(
            torch.randn(1, 6, hidden_dimension) / hidden_dimension**0.5)
        self.attention_gate = GateModule()
        self.feedforward_gate = GateModule()

    def forward(self, input_tensor, context_tensor, timestep_modulation, rotary_frequencies):
        # msa: multi-head self-attention  mlp: multi-layer perceptron

        shift_self_attn, scale_self_attn, gate_self_attn, shift_feedforward, scale_feedforward, gate_feedforward = (
            self.modulation_parameters.to(dtype=timestep_modulation.dtype, device=timestep_modulation.device) + timestep_modulation).chunk(6, dim=1)
        modulated_input = modulate(self.pre_attention_normalization(
            input_tensor), shift_self_attn, scale_self_attn)
        self_attention_output = self.self_attention_layer(
            modulated_input, rotary_frequencies)
        # print("before gate", self_attention_output.shape, gate_self_attn.shape, input_tensor.shape)
        input_tensor = self.attention_gate(
            input_tensor, gate_self_attn, self_attention_output)
        input_tensor = input_tensor + self.cross_attention_layer(
            self.pre_cross_attention_normalization(input_tensor), context_tensor)
        modulated_input = modulate(self.pre_feedforward_normalization(
            input_tensor), shift_feedforward, scale_feedforward)
        input_tensor = self.feedforward_gate(
            input_tensor, gate_feedforward, self.feedforward_network(modulated_input))
        return input_tensor


class MLP(torch.nn.Module):
    def __init__(self, input_dimension, output_dimension, supports_position_embedding=False):
        super().__init__()
        self.projection_network = torch.nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, input_dimension),
            nn.GELU(),
            nn.Linear(input_dimension, output_dimension),
            nn.LayerNorm(output_dimension)
        )
        self.supports_position_embedding = supports_position_embedding
        if supports_position_embedding:
            self.position_embedding = torch.nn.Parameter(
                torch.zeros((1, 514, 1280)))

    def forward(self, input_tensor):
        if self.supports_position_embedding:
            input_tensor = input_tensor + \
                self.position_embedding.to(
                    dtype=input_tensor.dtype, device=input_tensor.device)
        return self.projection_network(input_tensor)


class Head(nn.Module):
    def __init__(self, hidden_dimension: int, output_dimension: int, patch_dimensions: Tuple[int, int, int], epsilon: float):
        super().__init__()
        self.hidden_dimension = hidden_dimension
        self.patch_dimensions = patch_dimensions
        self.normalization_layer = nn.LayerNorm(
            hidden_dimension, eps=epsilon, elementwise_affine=False)
        self.output_projection = nn.Linear(
            hidden_dimension, output_dimension * math.prod(patch_dimensions))
        self.modulation_parameters = nn.Parameter(
            torch.randn(1, 2, hidden_dimension) / hidden_dimension**0.5)

    def forward(self, input_tensor, timestep_modulation):
        shift_parameter, scale_parameter = (self.modulation_parameters.to(dtype=timestep_modulation.dtype,
                                                                          device=timestep_modulation.device) + timestep_modulation).chunk(2, dim=1)
        output_tensor = (self.output_projection(self.normalization_layer(
            input_tensor) * (1 + scale_parameter) + shift_parameter))
        return output_tensor


class VastModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # vast_config
        vast_config = VastParams()
        self.in_dim = vast_config.in_channels
        self.ffn_dim = vast_config.ffn_dim
        self.out_dim = vast_config.out_channels
        self.text_dim = vast_config.text_dim
        self.freq_dim = vast_config.freq_dim
        self.eps = vast_config.eps
        self.patch_size = vast_config.patch_size
        self.has_image_input = vast_config.has_image_input
        self.has_image_pos_emb = vast_config.has_image_pos_emb

        # config
        self.dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_layers = config.num_layers

        self.patch_embedding_layer = nn.Conv3d(
            self.in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.text_embedding_layer = nn.Sequential(
            nn.Linear(self.text_dim, self.dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(self.dim, self.dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(self.dim, self.dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(self.has_image_input, self.dim,
                     self.num_heads, self.ffn_dim, self.eps)
            for _ in range(self.num_layers)
        ])
        self.head = Head(self.dim, self.out_dim, self.patch_size, self.eps)
        head_dim = self.dim // self.num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if self.has_image_input:
            # clip_feature_dim = 1280
            self.image_embedding_layer = MLP(
                1280, self.dim, has_pos_emb=self.has_image_pos_emb)

    def patchify(self, input_tensor: torch.Tensor):
        input_tensor = self.patch_embedding_layer(input_tensor)
        spatial_dimensions = input_tensor.shape[2:]
        input_tensor = rearrange(
            input_tensor, 'b c f h w -> b (f h w) c').contiguous()
        # input_tensor, spatial_dimensions: (f, h, w)
        return input_tensor, spatial_dimensions

    def unpatchify(self, input_tensor: torch.Tensor, spatial_dimensions: torch.Tensor):
        return rearrange(
            input_tensor, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=spatial_dimensions[0], h=spatial_dimensions[1], w=spatial_dimensions[2],
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                input_tensor: torch.Tensor,
                timestep_tensor: torch.Tensor,
                context_tensor: torch.Tensor,
                clip_feature_tensor: Optional[torch.Tensor] = None,
                additional_input_tensor: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                controlnet_images=None,
                **kwargs,
                ):
        timestep_embedding = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep_tensor))
        timestep_modulation = self.time_projection(
            timestep_embedding).unflatten(1, (6, self.dim))
        context_embedding = self.text_embedding_layer(context_tensor)

        if self.has_image_input:
            # (b, c_x + c_y, f, h, w)
            input_tensor = torch.cat(
                [input_tensor, additional_input_tensor], dim=1)
            clip_embedding = self.image_embedding_layer(clip_feature_tensor)
            context_embedding = torch.cat(
                [clip_embedding, context_embedding], dim=1)

        if controlnet_images is not None:
            # (b, c_x + c_y, f, h, w)
            input_tensor = torch.cat([input_tensor, controlnet_images], dim=1)

        input_tensor, (frame_count, height_count,
                       width_count) = self.patchify(input_tensor)

        rotary_embeddings = torch.cat([
            self.freqs[0][:frame_count].view(
                frame_count, 1, 1, -1).expand(frame_count, height_count, width_count, -1),
            self.freqs[1][:height_count].view(
                1, height_count, 1, -1).expand(frame_count, height_count, width_count, -1),
            self.freqs[2][:width_count].view(
                1, 1, width_count, -1).expand(frame_count, height_count, width_count, -1)
        ], dim=-1).reshape(frame_count * height_count * width_count, 1, -1).to(input_tensor.device)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for transformer_block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        input_tensor = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(transformer_block),
                            input_tensor, context_embedding, timestep_modulation, rotary_embeddings,
                            use_reentrant=False,
                        )
                else:
                    input_tensor = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(transformer_block),
                        input_tensor, context_embedding, timestep_modulation, rotary_embeddings,
                        use_reentrant=False,
                    )
            else:
                input_tensor = transformer_block(
                    input_tensor, context_embedding, timestep_modulation, rotary_embeddings)

        input_tensor = self.head(input_tensor, timestep_embedding)
        input_tensor = self.unpatchify(
            input_tensor, (frame_count, height_count, width_count))
        return input_tensor

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        state_dict = self.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars)
        return state_dict

    def sharded_state_dict(self):
        return self.state_dict()

from safetensors.torch import load_file
import torch.nn as nn
import torch

# from vast.train.configs.config import load_config
from collections.abc import Mapping, Sequence
import numpy as np
import os
import argparse
from collections import OrderedDict
from transformers import AutoModelForCausalLM, GPT2Config
import re
from diffusers.utils import WEIGHTS_NAME, WEIGHTS_INDEX_NAME
import json
from huggingface_hub import DDUFEntry, create_repo, split_torch_state_dict_into_shards

# ### logging
# import logging
# from rich.logging import RichHandler
# log = logging.basicConfig(
#     level="NOTSET",
#     format="%(message)s",
#     datefmt="[%X]",
#     handlers=[RichHandler(rich_tracebacks=True),logging.FileHandler("convert.log")]
# )or logging.getLogger(__name__)
# ###


# log.debug("This is a debug message")
# log.info("This is an info message")
# log.warning("This is a warning message")
# log.error("This is an error message")
# log.critical("This is a critical message")


# Load all safetensor files
def load_all_safetensors(model_dir):
    import os

    state_dict = {}

    for file in os.listdir(model_dir):
        if file.endswith('.safetensors'):
            file_path = os.path.join(model_dir, file)
            file_state_dict = load_file(file_path)
            state_dict.update(file_state_dict)

    return state_dict


@torch.inference_mode()
def clone_state_dict(elem):
    """clone all tensors in the elem to cpu device."""
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        elem = elem.clone()
    elif isinstance(elem, (np.ndarray, str)):
        pass
    elif isinstance(elem, Mapping):
        elem = dict(elem)
        for k, v in elem.items():
            elem[k] = clone_state_dict(v)
        elem = elem_type(elem)
    elif isinstance(elem, Sequence):
        elem = list(elem)
        for i in range(len(elem)):
            elem[i] = clone_state_dict(elem[i])
        elem = elem_type(elem)
    return elem


def get_element_from_dict_by_path(d, path):
    if path not in d:
        d[path] = {}
    d = d[path]
    return d


# def get_megatron_wan_state_dict(args):
# from megatron.core.models.wan.model import WanParams, WanVideoTransformer3DModel
# from megatron.training.arguments import core_transformer_config_from_args
# from megatron.training import get_args

# # initialize_megatron(parsed_args=args)
# args = get_args()

# wanConfig = WanParams()
# config = core_transformer_config_from_args(args)
# transformer = WanVideoTransformer3DModel(wanConfig, config)

# return transformer.state_dict().keys()
# return


def get_vast_wan_state_dict():
    from vast.models.dit.wan_dit import ModelManager
    from vast.pipelines.wan.wan_video import WanVideoPipeline

    text_encoder_path = (
        "/nvfile-heatstorage/model_zoo//Wan2___1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth"
    )
    vae_path = "/workspace/Wan2___1-I2V-14B-480P/Wan2.1_VAE.pth"
    image_encoder_path = (
        "/workspace/Wan2___1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    )
    dit_path = "/workspace/Wan2___1-FLF2V-14B-480P-init"

    model_path = [text_encoder_path, vae_path, image_encoder_path]
    dit_path_all = [
        os.path.join(dit_path, f) for f in os.listdir(dit_path) if f.endswith(".safetensors")
    ]
    dit_path_all = sorted(dit_path_all)

    model_path.append(dit_path_all)
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models(model_path)

    pipe = WanVideoPipeline.from_model_manager(model_manager)

    state_dict_dit = pipe.dit.state_dict()
    state_dict_vae = pipe.vae.state_dict()
    state_dict_text_encoder = pipe.text_encoder.state_dict()
    state_dict_image_encoder = pipe.image_encoder.state_dict()
    # print(state_dict_dit.keys())
    # for key in state_dict_dit.keys():
    #     print(key)
    # log.info(f"DIT: {state_dict_dit.keys()}")
    # log.info(f"VAE: {state_dict_vae.keys()}")
    # log.info(f"Text_Encoder: {state_dict_text_encoder.keys()}")
    # log.info(f"Image_Encoder: {state_dict_image_encoder.keys()}")
    return state_dict_dit, state_dict_vae, state_dict_text_encoder, state_dict_image_encoder


def update_params_with_identical_weights(params_dict, state_dict, weight_keys):
    for ori_key in weight_keys:
        if ori_key in state_dict:  # 确保 key 在 state_dict 中存在
            params_dict[ori_key] = state_dict[ori_key]


def update_dit_params(state_dict_dit, args=None):
    import re

    meg_state_dict_dit = {}
    replacement_rules = [
        (r'^', 'transformer.'),
        (r'text_embedding\.0', r'condition_embedder.text_embedder.linear_1'),
        (r'text_embedding\.2', r'condition_embedder.text_embedder.linear_2'),
        (r'time_embedding\.0', r'condition_embedder.time_embedder.linear_1'),
        (r'time_embedding\.2', r'condition_embedder.time_embedder.linear_2'),
        (r'time_projection\.1', r'condition_embedder.time_proj'),
        # No longer need.
        # (r'img_emb.proj.0', r'condition_embedder.image_embedder.norm1'),
        # (r'img_emb.proj.1', r'condition_embedder.image_embedder.ff.net.0.proj'),
        # (r'img_emb.proj.3', r'condition_embedder.image_embedder.ff.net.2'),
        # (r'img_emb.proj.4', r'condition_embedder.image_embedder.norm2'),
        (r'blocks\.(\d+)\.modulation', r'blocks.\1.scale_shift_table'),
        (r'blocks\.(\d+)\.self_attn\.o\.', r'blocks.\1.self_attention.linear_proj.'),
        (r'blocks\.(\d+)\.self_attn\.norm_q\.', r'blocks.\1.self_attention.q_layernorm.'),
        (r'blocks\.(\d+)\.self_attn\.norm_k\.', r'blocks.\1.self_attention.k_layernorm.'),
        (r'blocks\.(\d+)\.cross_attn\.q\.', r'blocks.\1.cross_attention.linear_q.'),
        (r'blocks\.(\d+)\.cross_attn\.k\.', r'blocks.\1.cross_attention.linear_k.'),
        (r'blocks\.(\d+)\.cross_attn\.v\.', r'blocks.\1.cross_attention.linear_v.'),
        (r'blocks\.(\d+)\.cross_attn\.o\.', r'blocks.\1.cross_attention.linear_proj.'),
        (r'blocks\.(\d+)\.cross_attn\.norm_q\.', r'blocks.\1.cross_attention.q_layernorm.'),
        (r'blocks\.(\d+)\.cross_attn\.norm_k\.', r'blocks.\1.cross_attention.k_layernorm.'),
        (r'blocks\.(\d+)\.cross_attn\.k_img\.', r'blocks.\1.cross_attention.add_k_proj.'),
        (r'blocks\.(\d+)\.cross_attn\.v_img\.', r'blocks.\1.cross_attention.add_v_proj.'),
        (
            r'blocks\.(\d+)\.cross_attn\.norm_k_img\.',
            r'blocks.\1.cross_attention.added_k_layernorm.',
        ),
        (r'blocks\.(\d+)\.norm3\.', r'blocks.\1.norm2.'),
        (r'blocks\.(\d+)\.ffn.0\.', r'blocks.\1.mlp.linear_fc1.'),
        (r'blocks\.(\d+)\.ffn.2\.', r'blocks.\1.mlp.linear_fc2.'),
        (r'head.modulation', r'scale_shift_table'),
        (r'head.head.weight', r'proj_out.weight'),
        (r'head.head.bias', r'proj_out.bias'),
    ]
    for src_key in state_dict_dit.keys():
        dst_key = src_key
        for pattern, repl in replacement_rules:
            dst_key = re.sub(pattern, repl, dst_key)
        meg_state_dict_dit[dst_key] = state_dict_dit[src_key]

    meg_state_dict_dit = convert_module_self_attention(meg_state_dict_dit)
    meg_state_dict_dit = convert_module_cross_attention(meg_state_dict_dit)
    meg_state_dict_dit = convert_module_mlp_attention(meg_state_dict_dit)

    return meg_state_dict_dit


def convert_module_self_attention(meg_state_dict_dit, args):
    config = GPT2Config.from_pretrained(args.hf_ckpt_path)
    hidden_size = config.attention_head_dim * config.num_attention_heads
    num_heads = config.num_attention_heads
    hidden_size_per_head = config.attention_head_dim
    group_per_split = config.num_attention_heads // args.target_tensor_model_parallel_size
    seg = (
        meg_state_dict_dit['transformer.blocks.0.self_attention.linear_proj.weight'].shape[1]
        // args.target_tensor_model_parallel_size
    )
    for pp_rank in range(args.target_pipeline_model_parallel_size):
        for i in range(args.target_tensor_model_parallel_size):
            for layer_id in range(config.num_layers):
                meg_state_dict_dit[
                    f'transformer.blocks.{layer_id}.self_attention.linear_proj.weight'
                ] = meg_state_dict_dit[f'blocks.{layer_id}.self_attention.linear_proj.weight'][
                    :, seg * i : seg * (i + 1)
                ]

                temp = torch.cat(
                    [
                        meg_state_dict_dit[
                            'transformer.blocks.'
                            + str(layer_id)
                            + '.self_attention.linear_q.weight'
                        ].view([num_heads, -1, hidden_size_per_head, hidden_size]),
                        meg_state_dict_dit[
                            'transformer.blocks.'
                            + str(layer_id)
                            + '.self_attention.linear_k.weight'
                        ].view([num_heads, -1, hidden_size_per_head, hidden_size]),
                        meg_state_dict_dit[
                            'transformer.blocks.'
                            + str(layer_id)
                            + '.self_attention.linear_v.weight'
                        ].view([num_heads, -1, hidden_size_per_head, hidden_size]),
                    ],
                    dim=1,
                )
                meg_state_dict_dit[
                    f'transformer.blocks.{layer_id}.self_attention.linear_qkv.weight'
                ] = (
                    temp[group_per_split * i : group_per_split * (i + 1)]
                    .view(-1, hidden_size)
                    .contiguous()
                )

                temp = torch.cat(
                    [
                        meg_state_dict_dit[
                            'transformer.blocks.' + str(layer_id) + '.self_attention.linear_q.bias'
                        ].view([num_heads, -1, hidden_size_per_head]),
                        meg_state_dict_dit[
                            'transformer.blocks.' + str(layer_id) + '.self_attention.linear_k.bias'
                        ].view([num_heads, -1, hidden_size_per_head]),
                        meg_state_dict_dit[
                            'transformer.blocks.' + str(layer_id) + '.self_attention.linear_v.bias'
                        ].view([num_heads, -1, hidden_size_per_head]),
                    ],
                    dim=1,
                )
                meg_state_dict_dit[
                    f'transformer.blocks.{layer_id}.self_attention.linear_qkv.bias'
                ] = (temp[group_per_split * i : group_per_split * (i + 1)].view(-1).contiguous())

    return meg_state_dict_dit


def convert_module_cross_attention(meg_state_dict_dit, args):
    config = GPT2Config.from_pretrained(args.hf_ckpt_path)
    hidden_size = config.attention_head_dim * config.num_attention_heads
    num_heads = config.num_attention_heads
    hidden_size_per_head = config.attention_head_dim
    group_per_split = config.num_attention_heads // args.target_tensor_model_parallel_size
    seg = (
        meg_state_dict_dit['transformer.blocks.0.self_attention.linear_proj.weight'].shape[1]
        // args.target_tensor_model_parallel_size
    )
    for pp_rank in range(args.target_pipeline_model_parallel_size):
        for i in range(args.target_tensor_model_parallel_size):
            for layer_id in range(config.num_layers):
                meg_state_dict_dit[f'blocks.{layer_id}.cross_attention.linear_proj.weight'] = (
                    meg_state_dict_dit[f'blocks.{layer_id}.cross_attn.o.weight'][
                        :, seg * i : seg * (i + 1)
                    ]
                )

                meg_state_dict_dit[f'blocks.{layer_id}.cross_attention.linear_q.weight'] = (
                    meg_state_dict_dit[f'blocks.{layer_id}.cross_attn.q.weight']
                    .view([num_heads, hidden_size_per_head, hidden_size])[
                        group_per_split * i : group_per_split * (i + 1)
                    ]
                    .view(-1, hidden_size)
                    .contiguous()
                )
                meg_state_dict_dit[f'blocks.{layer_id}.cross_attention.linear_q.bias'] = (
                    meg_state_dict_dit[f'blocks.{layer_id}.cross_attn.q.bias']
                    .view([num_heads, hidden_size_per_head])[
                        group_per_split * i : group_per_split * (i + 1)
                    ]
                    .view(-1)
                    .contiguous()
                )

                meg_state_dict_dit[f'blocks.{layer_id}.cross_attention.linear_k.weight'] = (
                    meg_state_dict_dit[f'blocks.{layer_id}.cross_attn.k.weight']
                    .view([num_heads, hidden_size_per_head, hidden_size])[
                        group_per_split * i : group_per_split * (i + 1)
                    ]
                    .view(-1, hidden_size)
                    .contiguous()
                )
                meg_state_dict_dit[f'blocks.{layer_id}.cross_attention.linear_k.bias'] = (
                    meg_state_dict_dit[f'blocks.{layer_id}.cross_attn.k.bias']
                    .view([num_heads, hidden_size_per_head])[
                        group_per_split * i : group_per_split * (i + 1)
                    ]
                    .view(-1)
                    .contiguous()
                )

                meg_state_dict_dit[f'blocks.{layer_id}.cross_attention.linear_v.weight'] = (
                    meg_state_dict_dit[f'blocks.{layer_id}.cross_attn.v.weight']
                    .view([num_heads, hidden_size_per_head, hidden_size])[
                        group_per_split * i : group_per_split * (i + 1)
                    ]
                    .view(-1, hidden_size)
                    .contiguous()
                )
                meg_state_dict_dit[f'blocks.{layer_id}.cross_attention.linear_v.bias'] = (
                    meg_state_dict_dit[f'blocks.{layer_id}.cross_attn.v.bias']
                    .view([num_heads, hidden_size_per_head])[
                        group_per_split * i : group_per_split * (i + 1)
                    ]
                    .view(-1)
                    .contiguous()
                )

                meg_state_dict_dit[f'blocks.{layer_id}.cross_attention.add_k_proj.weight'] = (
                    meg_state_dict_dit[f'blocks.{layer_id}.cross_attn.k_img.weight']
                    .view([num_heads, hidden_size_per_head, hidden_size])[
                        group_per_split * i : group_per_split * (i + 1)
                    ]
                    .view(-1, hidden_size)
                    .contiguous()
                )

                meg_state_dict_dit[f'blocks.{layer_id}.cross_attention.add_k_proj.bias'] = (
                    meg_state_dict_dit[f'blocks.{layer_id}.cross_attn.k_img.bias']
                    .view([num_heads, hidden_size_per_head])[
                        group_per_split * i : group_per_split * (i + 1)
                    ]
                    .view(-1)
                    .contiguous()
                )

                meg_state_dict_dit[f'blocks.{layer_id}.cross_attention.add_v_proj.weight'] = (
                    meg_state_dict_dit[f'blocks.{layer_id}.cross_attn.v_img.weight']
                    .view([num_heads, hidden_size_per_head, hidden_size])[
                        group_per_split * i : group_per_split * (i + 1)
                    ]
                    .view(-1, hidden_size)
                    .contiguous()
                )
                meg_state_dict_dit[f'blocks.{layer_id}.cross_attention.add_v_proj.bias'] = (
                    meg_state_dict_dit[f'blocks.{layer_id}.cross_attn.v_img.bias']
                    .view([num_heads, hidden_size_per_head])[
                        group_per_split * i : group_per_split * (i + 1)
                    ]
                    .view(-1)
                    .contiguous()
                )
    return meg_state_dict_dit


def convert_module_mlp_attention(meg_state_dict_dit, args):
    config = GPT2Config.from_pretrained(args.hf_ckpt_path)
    hidden_size = config.attention_head_dim * config.num_attention_heads
    num_heads = config.num_attention_heads
    hidden_size_per_head = config.attention_head_dim
    group_per_split = config.num_attention_heads // args.target_tensor_model_parallel_size
    seg = (
        meg_state_dict_dit['transformer.blocks.0.self_attention.linear_proj.weight'].shape[1]
        // args.target_tensor_model_parallel_size
    )
    for pp_rank in range(args.target_pipeline_model_parallel_size):
        for i in range(args.target_tensor_model_parallel_size):
            for layer_id in range(config.num_layers):
                seg = config.ffn_dim // args.target_tensor_model_parallel_size
                meg_state_dict_dit[f'blocks.{layer_id}.mlp.linear_fc1.weight'] = meg_state_dict_dit[
                    f'blocks.{layer_id}.ffn.0.weight'
                ][seg * i : seg * (i + 1), :]
                meg_state_dict_dit[f'blocks.{layer_id}.mlp.linear_fc1.bias'] = meg_state_dict_dit[
                    f'blocks.{layer_id}.ffn.0.bias'
                ][seg * i : seg * (i + 1)]
                meg_state_dict_dit[f'blocks.{layer_id}.mlp.linear_fc2.weight'] = meg_state_dict_dit[
                    f'blocks.{layer_id}.ffn.2.weight'
                ][:, seg * i : seg * (i + 1)]
                meg_state_dict_dit[f'blocks.{layer_id}.mlp.linear_fc2.bias'] = meg_state_dict_dit[
                    f'blocks.{layer_id}.ffn.2.bias'
                ]
    return meg_state_dict_dit


def update_vae_params(state_dict_vae):
    import re

    meg_state_dict_vae = {}
    for src_key in state_dict_vae.keys():
        dst_key = re.sub(r'^', 'vae.', src_key)
        meg_state_dict_vae[dst_key] = state_dict_vae[src_key]
    return meg_state_dict_vae


def update_text_encoder(state_dict_text_encoder):
    import re

    meg_state_dict_text_encoder = {}
    for src_key in state_dict_text_encoder.keys():
        dst_key = re.sub(r'^', 'text_encoder.', src_key)
        meg_state_dict_text_encoder[dst_key] = state_dict_text_encoder[src_key]
    return meg_state_dict_text_encoder


def update_image_encoder_params(state_dict_image_encoder):
    import re

    meg_state_dict_image_encoder = {}
    for src_key in state_dict_image_encoder.keys():
        dst_key = re.sub(r'^', 'image_encoder.', src_key)
        meg_state_dict_image_encoder[dst_key] = state_dict_image_encoder[src_key]
    return meg_state_dict_image_encoder


def convert_checkpoint_from_transformers_to_megatron(args):
    os.makedirs(args.save_path, exist_ok=True)
    os.system("cp -rf " + args.hf_ckpt_path + "/*.json " + args.save_path)

    # Saving the tracker file
    # log.critical('i am here')
    tracker_filepath = os.path.join(args.save_path, "latest_checkpointed_iteration.txt")
    with open(tracker_filepath, "w") as f:
        f.write("release")

    # create `release` dir in args.load_path
    release_dir = os.path.join(args.save_path, "release")
    os.makedirs(release_dir, exist_ok=True)

    config = GPT2Config.from_pretrained(args.hf_ckpt_path)
    config.num_layers = 1 # TODO

    megatron_args = {
        "attention_head_dim": config.attention_head_dim,
        "in_channels": config.in_channels,
        "num_layers": config.num_layers,
        "num_attention_heads": config.num_attention_heads,
        "hidden_size": config.attention_head_dim * config.num_attention_heads,
        "tensor_model_parallel_size": args.target_tensor_model_parallel_size,
        "pipeline_model_parallel_size": args.target_pipeline_model_parallel_size,
    }
    #from wan.configs.wan_flf import config as globalConfig
    import types

    # from vast.models.dit.wan_dit import WanVideoVAE, WanModel, ModelManager
    # from vast.pipelines.wan.wan_video import WanVideoPipeline
    # global_config = load_config(globalConfig)

    margs = types.SimpleNamespace()

    for k, v in megatron_args.items():
        setattr(margs, k, v)

    # model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")

    # model_config=global_config.models
    # model_path = [model_config.get("text_encoder_path"), model_config.get("vae_path"), model_config.get("image_encoder_path")]
    # dit_path = model_config.get("dit_path")
    # dit_path  = [os.path.join(dit_path, f) for f in os.listdir(dit_path) if f.endswith(".safetensors")]
    # dit_path = sorted(dit_path)

    # model_path.append(dit_path)
    # model_manager.load_models(model_path)

    # log.info(model_path)

    # layer_re = re.compile("transformer_blocks\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
    # hidden_size =  config.attention_head_dim*config.num_attention_heads
    # num_heads = config.num_attention_heads
    # hidden_size_per_head = config.attention_head_dim
    # group_per_split = config.num_attention_heads // args.target_tensor_model_parallel_size

    output_state_dict = []
    for i in range(args.target_tensor_model_parallel_size):
        output_state_dict.append(OrderedDict())

    # if args.target_params_dtype == "fp16":
    #     dtype = torch.float16
    # elif args.target_params_dtype == "bf16":
    #     dtype = torch.bfloat16
    # else:
    #     dtype = torch.float32

    # pipe = WanVideoPipeline.from_model_manager(model_manager)

    state_dict_dit, state_dict_vae, state_dict_text_encoder, state_dict_image_encoder = (
        get_vast_wan_state_dict()
    )
    # state_dict_dit ,state_dict_vae, state_dict_text_encoder, state_dict_image_encoder = get_megatron_wan_state_dict(args)

    # group_per_split = config.num_attention_heads // args.target_tensor_model_parallel_size
    for pp_rank in range(args.target_pipeline_model_parallel_size):
        # for layer in range(num_layers):
        for i in range(args.target_tensor_model_parallel_size):
            params_dict = get_element_from_dict_by_path(output_state_dict[i], "model")
            # log.info(params_dict)
            IDENTICAL_WEIGHT = {
                f'patch_embedding.weight',
                f'patch_embedding.bias',
                f'text_embedding.0.weight',
                f'text_embedding.0.bias',
                f'text_embedding.2.weight',
                f'text_embedding.2.bias',
                f'time_embedding.0.weight',
                f'time_embedding.0.bias',
                f'time_embedding.2.weight',
                f'time_embedding.2.bias',
                f'time_projection.1.weight',
                f'time_projection.1.bias',
                f'img_emb.emb_pos',
                f'img_emb.proj.0.weight',
                f'img_emb.proj.0.bias',
                f'img_emb.proj.1.weight',
                f'img_emb.proj.1.bias',
                f'img_emb.proj.3.weight',
                f'img_emb.proj.3.bias',
                f'img_emb.proj.4.weight',
                f'img_emb.proj.4.bias',
            }
            update_params_with_identical_weights(params_dict, state_dict_dit, IDENTICAL_WEIGHT)
            for layer_id in range(config.num_layers):
                params_dict[f'blocks.{layer_id}.norm2.weight'] = (
                    state_dict_dit[f'blocks.{layer_id}.norm3.weight']
                )
                params_dict[f'blocks.{layer_id}.norm2.bias'] = (
                    state_dict_dit[f'blocks.{layer_id}.norm3.bias']
                )
                
                params_dict[f'blocks.{layer_id}.self_attention.linear_proj.weight'] = (
                    state_dict_dit[f'blocks.{layer_id}.self_attn.o.weight']
                )
                params_dict[f'blocks.{layer_id}.self_attention.linear_proj.bias'] = state_dict_dit[
                    f'blocks.{layer_id}.self_attn.o.bias'
                ]
                params_dict[f'blocks.{layer_id}.self_attention.linear_q.weight'] = state_dict_dit[
                    f'blocks.{layer_id}.self_attn.q.weight'
                ]
                params_dict[f'blocks.{layer_id}.self_attention.linear_k.weight'] = state_dict_dit[
                    f'blocks.{layer_id}.self_attn.k.weight'
                ]
                params_dict[f'blocks.{layer_id}.self_attention.linear_v.weight'] = state_dict_dit[
                    f'blocks.{layer_id}.self_attn.v.weight'
                ]
                params_dict[f'blocks.{layer_id}.self_attention.linear_q.bias'] = state_dict_dit[
                    f'blocks.{layer_id}.self_attn.q.bias'
                ]
                params_dict[f'blocks.{layer_id}.self_attention.linear_k.bias'] = state_dict_dit[
                    f'blocks.{layer_id}.self_attn.k.bias'
                ]
                params_dict[f'blocks.{layer_id}.self_attention.linear_v.bias'] = state_dict_dit[
                    f'blocks.{layer_id}.self_attn.v.bias'
                ]
                params_dict[f'blocks.{layer_id}.scale_shift_table'] = state_dict_dit[
                    f'blocks.{layer_id}.modulation'
                ]
                params_dict[f'blocks.{layer_id}.self_attention.q_layernorm.weight'] = (
                    state_dict_dit[f'blocks.{layer_id}.self_attn.norm_q.weight']
                )
                params_dict[f'blocks.{layer_id}.self_attention.k_layernorm.weight'] = (
                    state_dict_dit[f'blocks.{layer_id}.self_attn.norm_k.weight']
                )
                params_dict[f'blocks.{layer_id}.cross_attention.linear_proj.weight'] = (
                    state_dict_dit[f'blocks.{layer_id}.cross_attn.o.weight']
                )
                params_dict[f'blocks.{layer_id}.cross_attention.linear_proj.bias'] = state_dict_dit[
                    f'blocks.{layer_id}.cross_attn.o.bias'
                ]
                params_dict[f'blocks.{layer_id}.cross_attention.q_layernorm.weight'] = (
                    state_dict_dit[f'blocks.{layer_id}.cross_attn.norm_q.weight']
                )
                params_dict[f'blocks.{layer_id}.cross_attention.k_layernorm.weight'] = (
                    state_dict_dit[f'blocks.{layer_id}.cross_attn.norm_k.weight']
                )
                params_dict[f'blocks.{layer_id}.cross_attention.norm_k_img.weight'] = (
                    state_dict_dit[f'blocks.{layer_id}.cross_attn.norm_k_img.weight']
                )
                params_dict[f'blocks.{layer_id}.cross_attention.linear_q.weight'] = state_dict_dit[
                    f'blocks.{layer_id}.cross_attn.q.weight'
                ]
                params_dict[f'blocks.{layer_id}.cross_attention.linear_q.bias'] = state_dict_dit[
                    f'blocks.{layer_id}.cross_attn.q.bias'
                ]
                params_dict[f'blocks.{layer_id}.cross_attention.linear_k.weight'] = state_dict_dit[
                    f'blocks.{layer_id}.cross_attn.k.weight'
                ]
                params_dict[f'blocks.{layer_id}.cross_attention.linear_k.bias'] = state_dict_dit[
                    f'blocks.{layer_id}.cross_attn.k.bias'
                ]
                params_dict[f'blocks.{layer_id}.cross_attention.linear_v.weight'] = state_dict_dit[
                    f'blocks.{layer_id}.cross_attn.v.weight'
                ]
                params_dict[f'blocks.{layer_id}.cross_attention.linear_v.bias'] = state_dict_dit[
                    f'blocks.{layer_id}.cross_attn.v.bias'
                ]
                params_dict[f'blocks.{layer_id}.cross_attention.k_img.weight'] = (
                    state_dict_dit[f'blocks.{layer_id}.cross_attn.k_img.weight']
                )
                params_dict[f'blocks.{layer_id}.cross_attention.k_img.bias'] = state_dict_dit[
                    f'blocks.{layer_id}.cross_attn.k_img.bias'
                ]
                params_dict[f'blocks.{layer_id}.cross_attention.v_img.weight'] = (
                    state_dict_dit[f'blocks.{layer_id}.cross_attn.v_img.weight']
                )
                params_dict[f'blocks.{layer_id}.cross_attention.v_img.bias'] = state_dict_dit[
                    f'blocks.{layer_id}.cross_attn.v_img.bias'
                ]
                seg = config.ffn_dim // args.target_tensor_model_parallel_size
                params_dict[f'blocks.{layer_id}.mlp.linear_fc1.weight'] = state_dict_dit[
                    f'blocks.{layer_id}.ffn.0.weight'
                ]
                params_dict[f'blocks.{layer_id}.mlp.linear_fc1.bias'] = state_dict_dit[
                    f'blocks.{layer_id}.ffn.0.bias'
                ]
                params_dict[f'blocks.{layer_id}.mlp.linear_fc2.weight'] = state_dict_dit[
                    f'blocks.{layer_id}.ffn.2.weight'
                ]
                params_dict[f'blocks.{layer_id}.mlp.linear_fc2.bias'] = state_dict_dit[
                    f'blocks.{layer_id}.ffn.2.bias'
                ]
                # params_dict[f'norm_out']=state_dict #有问题？
                params_dict[f'proj_out.weight'] = state_dict_dit['head.head.weight']
                params_dict[f'proj_out.bias'] = state_dict_dit['head.head.bias']
                params_dict[f'scale_shift_table'] = state_dict_dit['head.modulation']

            keys = list(params_dict.keys())
            for k in keys:
                params_dict[f"transformer.{k}"] = params_dict.pop(k)

            # log.info("begin vae")
            # for param_name, param_tensor in state_dict_vae.items():
            #     params_dict[f"vae.{param_name}"] = param_tensor
            # # log.info("text encoder")
            # for param_name, param_tensor in state_dict_text_encoder.items():
            #     params_dict[f"text_encoder.{param_name}"] = param_tensor
            # # log.info("image_encoder")
            # for param_name, param_tensor in state_dict_image_encoder.items():
            #     params_dict[f"image_encoder.{param_name}"] = param_tensor
    for tp_rank in range(args.target_tensor_model_parallel_size):
        output_state_dict[tp_rank]["checkpoint_version"] = 3.0
        output_state_dict[tp_rank]["args"] = margs
        checkpoint_dir = (
            f"mp_rank_{tp_rank:02d}"
            if args.target_pipeline_model_parallel_size == 1
            else f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"
        )
        checkpoint_name = "model_optim_rng.pt"
        checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        torch.save(clone_state_dict(output_state_dict[tp_rank]), checkpoint_path)
    # log.info("finish!!!!")


def update_params_with_identical_weights_for_megatron_to_transformers(
    params_dict, state_dict, weight_keys
):
    for ori_key in weight_keys:
        if ori_key in state_dict:
            new_key_for_transformers = ori_key.replace("transformer.", "", 1)
            params_dict[new_key_for_transformers] = state_dict.pop(ori_key)


def megatron_to_transformers_fix_query_key_value_ordering(
    param, checkpoint_version, num_splits, num_heads, hidden_size
):
    """
    Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :] for compatibility with later versions
    of NVIDIA Megatron-LM. The inverse operation is performed inside Megatron-LM to read checkpoints:
    https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209 If param is the weight tensor of the
    self-attention block, the returned tensor will have to be transposed one more time to be read by HuggingFace GPT2.
    This function is taken from `convert_megatron_gpt2_checkpoint.py`
    Args:
        param (torch.Tensor): the tensor to permute
        checkpoint_version (int): the version of the checkpoint.
        num_splits (int): the number of projections, usually 3 for (Query, Key, Value)
        num_heads (int): the number of attention heads
        hidden_size (int): the hidden size per head
    """

    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def get_megatron_sharded_states(args, tp_size, pp_size, pp_rank):
    """
    Get sharded checkpoints from NVIDIA Megatron-LM checkpoint based on the provided tensor parallel size, pipeline
    parallel size and pipeline parallel rank.
    Args:
        args (argparse.Namespace): the arguments to the script
        tp_size (int): the tensor parallel size
        pp_size (int): the pipeline parallel size
        pp_rank (int): the pipeline parallel rank
    """
    tp_state_dicts = []
    for i in range(tp_size):
        sub_dir_name = f"mp_rank_{i:02d}" if pp_size == 1 else f"mp_rank_{i:02d}_{pp_rank:03d}"
        checkpoint_name = os.listdir(os.path.join(args.load_path, sub_dir_name))[0]
        checkpoint_path = os.path.join(args.load_path, sub_dir_name, checkpoint_name)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        tp_state_dicts.append(state_dict)

    return tp_state_dicts


def convert_checkpoint_from_megatron_to_transformers(args):
    """
    Convert NVIDIA Megatron-LM checkpoint to HuggingFace Transformers checkpoint. This handles Megatron checkpoints
    with different tensor parallelism and pipeline parallelism sizes. It saves the converted checkpoint into shards
    using HuggingFace Transformers checkpoint sharding functionality. This greatly extends the functionality of
    `convert_megatron_gpt2_checkpoint.py`

    Args:
        args (argparse.Namespace): the arguments to the script
    """
    # Load Megatron-LM checkpoint arguments from the state dict
    os.makedirs(args.save_path, exist_ok=True)

    sub_dirs = os.listdir(args.load_path)
    possible_sub_dirs = ["mp_rank_00", "mp_rank_00_000"]
    for sub_dir in possible_sub_dirs:
        if sub_dir in sub_dirs:
            rank0_checkpoint_name = [
                i for i in os.listdir(os.path.join(args.load_path, sub_dir)) if 'rng' in i
            ][0]
            rank0_checkpoint_path = os.path.join(args.load_path, sub_dir, rank0_checkpoint_name)
            break

    print(f"Loading Megatron-LM checkpoint arguments from: {rank0_checkpoint_path}")
    state_dict = torch.load(rank0_checkpoint_path, map_location="cpu")
    megatron_args = state_dict.get("args", None)
    if megatron_args is None:
        raise ValueError(
            "Megatron-LM checkpoint does not contain arguments. This utility only supports Megatron-LM checkpoints"
            " containing all the megatron arguments. This is because it loads all config related to model"
            " architecture, the tensor and pipeline model parallel size from the checkpoint insead of user having to"
            " manually specify all the details. Please save Megatron-LM checkpoint along with all the megatron"
            " arguments to use this utility."
        )

    # params dtype
    if args.target_params_dtype == "fp16":
        dtype = torch.float16
    elif args.target_params_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    dir_path = os.path.dirname(args.load_path)
    config = GPT2Config.from_pretrained(dir_path)
    config.num_layers = 20
    config.num_single_layers = 40

    output_state_dict = {}

    checkpoint_version = state_dict.get("checkpoint_version", 3.0)
    tp_size = megatron_args.tensor_model_parallel_size
    pp_size = megatron_args.pipeline_model_parallel_size

    # The regex to extract layer names.
    layer_re = re.compile("layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
    path = 'model'
    # Convert.
    print("Converting")

    tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, 0)
    DIT_IDENTICAL_WEIGHT = {
        f'patch_embedding.weight',
        f'patch_embedding.bias',
        f'text_embedding.0.weight',
        f'text_embedding.0.bias',
        f'text_embedding.2.weight',
        f'text_embedding.2.bias',
        f'time_embedding.0.weight',
        f'time_embedding.0.bias',
        f'time_embedding.2.weight',
        f'time_embedding.2.bias',
        f'time_projection.1.weight',
        f'time_projection.1.bias',
        f'img_emb.emb_pos',
        f'img_emb.proj.0.weight',
        f'img_emb.proj.0.bias',
        f'img_emb.proj.1.weight',
        f'img_emb.proj.1.bias',
        f'img_emb.proj.3.weight',
        f'img_emb.proj.3.bias',
        f'img_emb.proj.4.weight',
        f'img_emb.proj.4.bias',
    }
    new_DIT_IDENTICAL_WEIGHT = ["transformer." + item for item in DIT_IDENTICAL_WEIGHT]
    update_params_with_identical_weights_for_megatron_to_transformers(
        output_state_dict,
        get_element_from_dict_by_path(tp_state_dicts[0], path),
        new_DIT_IDENTICAL_WEIGHT,
    )

    print("Converting transformer layers")
    hidden_size = config.attention_head_dim * config.num_attention_heads
    num_heads = config.num_attention_heads // tp_size
    hidden_size_per_head = config.attention_head_dim
    # Extract the layers.
    for key, val in get_element_from_dict_by_path(tp_state_dicts[0], path).items():
        if "extra_state" in key:
            print(f"key: {key}, val: {val}")
            continue
        else:
            print(key)
        # if isinstance(val, torch.Tensor):
        #     print(f"key: {key}, val: {val.shape}")
        # if key.startswith("transformer.transformer_blocks"):
        if key.startswith("blocks"):
            key_list = key.split('.')
            # layer_id = int(key_list[2])
            layer_id = int(key_list[1])
            if 'q_layernorm' in key:
                if 'self' in key:
                    output_state_dict[f'blocks.{layer_id}.self_attn.norm_q.weight'] = val
                else:
                    output_state_dict[f'blocks.{layer_id}.cross_attn.norm_q.weight'] = val
            if 'k_layernorm' in key:
                if 'self' in key:
                    output_state_dict[f'blocks.{layer_id}.self_attn.norm_k.weight'] = val
                else:
                    if 'added' in key:
                        output_state_dict[f'blocks.{layer_id}.cross_attn.norm_k_img.weight'] = val
                    else:
                        output_state_dict[f'blocks.{layer_id}.cross_attn.norm_k.weight'] = val
            if 'linear_fc' in key:
                dim = 1 if 'linear_fc2' in key else 0
                if "weight" in key:
                    params = torch.cat(
                        [val]
                        + [
                            get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
                            for tp_rank in range(1, tp_size)
                        ],
                        dim=dim,
                    ).to(dtype)
                    if 'linear_fc2' in key:
                        output_state_dict[f'blocks.{layer_id}.ffn.2.weight'] = params
                    else:
                        output_state_dict[f'blocks.{layer_id}.ffn.0.weight'] = params
                if "bias" in key:
                    print(f"bias: {key}")
                    if "linear_fc2" in key:
                        output_state_dict[f'blocks.{layer_id}.ffn.2.bias'] = val.to(dtype)
                    else:
                        params = torch.cat(
                            [val]
                            + [
                                get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[
                                    key
                                ]
                                for tp_rank in range(1, tp_size)
                            ],
                            dim=dim,
                        ).to(dtype)
                        output_state_dict[f'transformer_blocks.{layer_id}.ffn.0.bias'] = params
            if 'linear' in key:
                if 'self_attention' in key:
                    if 'weight' in key:
                        if 'proj' in key:
                            output_state_dict[f'blocks.{layer_id}.self_attn.o.weight'] = val
                        elif '_k' in key:
                            output_state_dict[f'blocks.{layer_id}.self_attn.k.weight'] = val
                        elif '_q' in key:
                            output_state_dict[f'blocks.{layer_id}.self_attn.q.weight'] = val
                        elif '_v' in key:
                            output_state_dict[f'blocks.{layer_id}.self_attn.v.weight'] = val
                    if 'bias' in key:
                        if 'proj' in key:
                            output_state_dict[f'blocks.{layer_id}.self_attn.o.bias'] = val
                        elif '_k' in key:
                            output_state_dict[f'blocks.{layer_id}.self_attn.k.bias'] = val
                        elif '_q' in key:
                            output_state_dict[f'blocks.{layer_id}.self_attn.q.bias'] = val
                        elif '_v' in key:
                            output_state_dict[f'blocks.{layer_id}.self_attn.v.bias'] = val
                if 'cross_attention' in key:
                    if 'weight' in key:
                        if 'k_img' in key:
                            output_state_dict[f'blocks.{layer_id}.cross_attn.k_img.weight'] = val
                        elif 'v_img' in key:
                            output_state_dict[f'blocks.{layer_id}.cross_attn.v_img.weight'] = val
                        elif 'proj' in key:
                            output_state_dict[f'blocks.{layer_id}.cross_attn.o.weight'] = val
                        elif '_k' in key:
                            output_state_dict[f'blocks.{layer_id}.cross_attn.k.weight'] = val
                        elif '_q' in key:
                            output_state_dict[f'blocks.{layer_id}.cross_attn.q.weight'] = val
                        elif '_v' in key:
                            output_state_dict[f'blocks.{layer_id}.cross_attn.v.weight'] = val
                    if 'bias' in key:
                        if 'add_k_proj' in key:
                            output_state_dict[f'blocks.{layer_id}.cross_attn.k_img.bias'] = val
                        elif 'add_v_proj' in key:
                            output_state_dict[f'blocks.{layer_id}.cross_attn.v_img.bias'] = val
                        elif 'proj' in key:
                            output_state_dict[f'blocks.{layer_id}.cross_attn.o.bias'] = val
                        elif '_k' in key:
                            output_state_dict[f'blocks.{layer_id}.cross_attn.k.bias'] = val
                        elif '_q' in key:
                            output_state_dict[f'blocks.{layer_id}.cross_attn.q.bias'] = val
                        elif '_v' in key:
                            output_state_dict[f'blocks.{layer_id}.cross_attn.v.bias'] = val
            if 'scale_shift_table' in key:
                output_state_dict[f'blocks.{layer_id}.modulation'] = val
            if 'norm2' in key:
                if 'weight' in key:
                    output_state_dict[f'blocks.{layer_id}.norm3.weight']=val
                if 'bias' in key:               
                    output_state_dict[f'blocks.{layer_id}.norm3.bias']=val
        if 'proj_out' in key:
            if 'weight' in key:
                output_state_dict['head.head.weight'] = val
            if 'bias' in key:
                output_state_dict['head.head.bias'] = val
        if 'scale_shift_table' in key:
            output_state_dict['head.modulation'] = val
    log_file = open("output_state_dict_kv.txt", "w")
    for key, val in output_state_dict.items():
        log_file.write(f"{key}: shape={tuple(val.shape)}, dtype={val.dtype}\n")
    log_file.close()

    filename_pattern = 'diffusion_pytorch_model{suffix}.bin'
    state_dict_split = split_torch_state_dict_into_shards(
        output_state_dict, max_shard_size="50GB", filename_pattern=filename_pattern
    )
    # Save the model
    if not os.path.exists(args.save_path):
        os.system(f'mkdir -p {args.save_path}')
    # 保存每个shard
    for filename, tensors in state_dict_split.filename_to_tensors.items():
        shard = {tensor: output_state_dict[tensor].contiguous() for tensor in tensors}
        filepath = os.path.join(args.save_path, filename)
        torch.save(shard, filepath)

    # 保存index文件
    if state_dict_split.is_sharded:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }
        with open(
            os.path.join(args.save_path, "diffusion_pytorch_model.safetensors.index.json"), "w"
        ) as f:
            f.write(json.dumps(index, indent=2))

        print(f"Sharded model saved successfully with index file at {args.save_path}")
    else:
        print(f"Model small enough, saved without sharding in {args.save_path}/{WEIGHTS_NAME}")

    config_path = '/'.join(args.load_path.split('/')[:-1])
    os.system("cp -rf " + config_path + "/config.json " + args.save_path)
    print("Conversion from Megatron-LM to Transformers is done!")


def add_extra_args(parser):
    parser.add_argument(
        '--convert-checkpoint-from-megatron-to-transformers',
        action='store_true',
        help=(
            'If True, convert a Megatron checkpoint to a Transformers checkpoint. '
            'If False, convert a Transformers checkpoint to a Megatron checkpoint.'
        ),
    )
    parser.add_argument("--target-tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--target-pipeline-model-parallel-size", type=int, default=1)
    parser.add_argument("--hf-ckpt-path", type=str)
    parser.add_argument("--load-path", type=str)
    parser.add_argument("--save-path", type=str)
    parser.add_argument("--target-params-dtype", type=str)
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="10GB",
        help=(
            "The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size "
            "lower than this size. If expressed as a string, needs to be digits followed by a unit (like `5MB`). "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )

    return parser


def main():
    parser = argparse.ArgumentParser()
    parser = add_extra_args(parser)

    # state_dict_dit, state_dict_vae, state_dict_text_encoder, state_dict_image_encoder = get_vast_wan_state_dict()
    # for key in updated_state_dict_dit.keys():
    #     print(key)
    args = parser.parse_args()
    # get_megatron_wan_state_dict(args)
    if args.convert_checkpoint_from_megatron_to_transformers:
        convert_checkpoint_from_megatron_to_transformers(args)
    else:
        convert_checkpoint_from_transformers_to_megatron(args)


if __name__ == "__main__":
    main()

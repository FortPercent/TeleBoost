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
from IPython import embed
### logging
import logging
from rich.logging import RichHandler
log = logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True),logging.FileHandler("convert.log")]
)or logging.getLogger(__name__)
###


log.debug("This is a debug message")
log.info("This is an info message")
log.warning("This is a warning message")
log.error("This is an error message")
log.critical("This is a critical message")


@torch.inference_mode()
def get_megatron_wan_state_dict():
    import torch
    from megatron.core.models.wan.model import WanParams, WanVideoTransformer3DModel
    from megatron.training.arguments import core_transformer_config_from_args
    from megatron.training import get_args
    from vast.models.dit.wan_dit import WanTextEncoder
    from vast.models.dit.wan_dit import WanVideoVAE
    from vast.models.dit.wan_dit import WanImageEncoder
    import torch.nn.init as init

    args = get_args()  
    args.init_method = init.xavier_uniform_  # Note the underscore! This is the callable.  
    args.output_layer_init_method = init.xavier_uniform
    args.window_size = (1,1) 
    args.gated_linear_unit = False  # Or True if your model needs GLU
    args.activation_func = "gelu" 
    wanConfig = WanParams()


    transformer = WanVideoTransformer3DModel(wanConfig, args)
    text_encoder = WanTextEncoder()
    image_encoder = WanImageEncoder()
    vae = WanVideoVAE()

    state_dict_dit = transformer.state_dict()
    state_dict_image_encoder = image_encoder.state_dict()
    state_dict_text_encoder =text_encoder.state_dict()
    state_dict_vae = vae.state_dict()

    return [state_dict_dit, state_dict_image_encoder, state_dict_text_encoder, state_dict_vae]
        

@torch.inference_mode()
def get_vast_wan_state_dict(args):
    from vast.models.dit.wan_dit import ModelManager
    from vast.pipelines.wan.wan_video import WanVideoPipeline

    #dit_path = "/workspace/Wan2___1-FLF2V-14B-480P-init"
    dit_path = args.load_path
    dit_path = [os.path.join(dit_path, f) for f in os.listdir(dit_path) if f.endswith(".pth")]
    vae_path = "/workspace/Wan2___1-I2V-14B-480P/Wan2.1_VAE.pth"
    image_encoder_path = "/workspace/Wan2___1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    text_encoder_path = "/workspace/Wan2___1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth"

    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([
        text_encoder_path, vae_path, image_encoder_path, 
        *dit_path
    ])

    pipe = WanVideoPipeline.from_model_manager(model_manager)

    state_dict_dit = pipe.dit.state_dict()
    state_dict_vae = pipe.vae.state_dict()
    state_dict_text_encoder = pipe.text_encoder.state_dict()
    state_dict_image_encoder = pipe.image_encoder.state_dict()
    return [state_dict_dit, state_dict_vae, state_dict_text_encoder, state_dict_image_encoder]

def convert_checkpoint_from_transformers_to_megatron_bak(args):
    src_dict = get_vast_wan_state_dict(args)
    # dst_dict = get_megatron_wan_state_dict()
    dst_dict = [{}, {}, {}, {}]

    update_dit_params(src_dict, dst_dict, args)
    # update_vae_params(src_dict, dst_dict)
    # update_text_encoder_params(src_dict, dst_dict)
    # update_image_encoder_params(src_dict, dst_dict)

    return save_to_tensor(dst_dict, args)

def save_to_tensor(dst_dict, args):
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    os.system("cp -rf " + "/nvfile-heatstorage/model_zoo/huggingface/Wan2.1-I2V-14B-720P-Diffusers/transformer" + "/*.json " + save_path)

    tracker_filepath = os.path.join(save_path, "latest_checkpointed_iteration.txt")
    with open(tracker_filepath, "w") as f:
        f.write("release")

    release_dir = os.path.join(save_path, "release")
    os.makedirs(release_dir, exist_ok=True)

    config = GPT2Config.from_pretrained("/nvfile-heatstorage/model_zoo/huggingface/Wan2.1-I2V-14B-720P-Diffusers/transformer")
    config.num_layers = args.num_layers

    megatron_args = {
        "attention_head_dim": config.attention_head_dim,
        "in_channels": config.in_channels,
        "num_layers": config.num_layers,
        "num_attention_heads": config.num_attention_heads,
        "hidden_size": config.attention_head_dim * config.num_attention_heads,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
    }
    import types

    margs = types.SimpleNamespace()
    for k, v in megatron_args.items():
        setattr(margs, k, v)

    output_state_dict = []
    pp_rank = 0
    for tp_rank in range(megatron_args["tensor_model_parallel_size"]):
        output_state_dict.append(OrderedDict({
            "args": margs,  
            "checkpoint_version": 3.0,
            "model":  {k: v for sub_dict in dst_dict for k, v in sub_dict.items()},
        }))
    
        checkpoint_dir = (
            f"mp_rank_{tp_rank:02d}"
            if megatron_args["tensor_model_parallel_size"] == 1
            else f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"
        )
        checkpoint_name = "model_optim_rng.pt"
        checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        torch.save(output_state_dict[tp_rank], checkpoint_path)

    log.info("Save all tensor")



def update_dit_params(src_dict, dst_dict, args):
    index = 0    
    src_dit = src_dict[index]
    dst_dit = dst_dict[index].copy()
    
    meg_state_dict = {}
    import re
    replacement_rules = [
        (r'^', 'transformer.'),
    ]
    for src_key in src_dit.keys():
        if "blocks" in src_key:
            if int(src_key.split(".")[1]) >= args.num_layers:
                continue
        dst_key = src_key
        for pattern, repl in replacement_rules:
            dst_key = re.sub(pattern, repl, dst_key)
        meg_state_dict[dst_key] = src_dit[src_key]
    dst_dict[index] = meg_state_dict
    

def update_vae_params(src_dict, dst_dict, args=None):
    index = 1
    src_dit = src_dict[index]
    dst_dit = dst_dict[index].copy()
    
    meg_state_dict = {}
    import re
    replacement_rules = [
        # add vae to all block
        (r'^', 'vae.'),
    ]
    for src_key in src_dit.keys():
        dst_key = src_key
        for pattern, repl in replacement_rules:
            dst_key = re.sub(pattern, repl, dst_key)
        meg_state_dict[dst_key] = src_dit[src_key]

    dst_dict[index] = meg_state_dict

def update_text_encoder_params(src_dict, dst_dict, args=None):
    index = 2
    src_dit = src_dict[index]
    dst_dit = dst_dict[index].copy()
    
    meg_state_dict = {}
    import re
    replacement_rules = [
        # add text_encoder. to all block
        (r'^', 'text_encoder.'),
    ]
    for src_key in src_dit.keys():
        dst_key = src_key
        for pattern, repl in replacement_rules:
            dst_key = re.sub(pattern, repl, dst_key)
        meg_state_dict[dst_key] = src_dit[src_key]

    dst_dict[index] = meg_state_dict

def update_image_encoder_params(src_dict, dst_dict, args=None):
    index = 3
    src_dit = src_dict[index]
    dst_dit = dst_dict[index].copy()
    
    meg_state_dict = {}
    import re
    replacement_rules = [
        # add image_encoder. to all block
        (r'^', 'image_encoder.'),
    ]
    for src_key in src_dit.keys():
        dst_key = src_key
        for pattern, repl in replacement_rules:
            dst_key = re.sub(pattern, repl, dst_key)
        meg_state_dict[dst_key] = src_dit[src_key]

    dst_dict[index] = meg_state_dict



def add_extra_args(parser):
    group = parser.add_argument_group("Custom Conversion Args")
    group.add_argument(
        '--convert-checkpoint-from-megatron-to-transformers',
        action='store_true',
        help=(
            'If True, convert a Megatron checkpoint to a Transformers checkpoint. '
            'If False, convert a Transformers checkpoint to a Megatron checkpoint.'
        ),
    )
    group.add_argument("--target-tensor-model-parallel-size", type=int, default=1)
    group.add_argument("--target-pipeline-model-parallel-size", type=int, default=1)
    group.add_argument("--hf-ckpt-path", type=str)
    group.add_argument("--load-path", type=str)
    group.add_argument("--save-path", type=str)
    group.add_argument("--num-layers", type=int)
    group.add_argument("--target-params-dtype", type=str)
    group.add_argument(
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
    args_defaults = {
        "num_layers": 1,
        "hidden_size": 5120, 
        "micro_batch_size": 1,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "no_load_optim": True,
        "no_load_rng": True,
        "num-attention-heads": 40,
        "seq-length": 512,
        "max_position_embeddings": 4096,
        "vocab-size": 0,
        "num_attention_heads": 40,
        "encoder_seq_length" : 1,
        "save_interval": False,
        'tokenizer_type': 'GPT2BPETokenizer',
        "vocab_file": "/nvfile-heatstorage/teleai-infra/wxe/Megatron-LM/data/gpt_2_vocab.json",
        "merge_file": "/nvfile-heatstorage/teleai-infra/wxe/Megatron-LM/data/gpt_2_merge.txt",
        "softmax_scale":1,
    }
    parser = argparse.ArgumentParser()
    parser = add_extra_args(parser)
    args = parser.parse_args()

    import os
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"

    # initialize_megatron(extra_args_provider=add_extra_args, args_defaults=args_defaults)

    convert_checkpoint_from_transformers_to_megatron_bak(args)


if __name__ == "__main__":
    main()
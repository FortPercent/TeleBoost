# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import os
import json
import torch
import teletron
# from megatron.core import parallel_state, tensor_parallel
from megatron.core.enums import ModelType
from megatron.training.arguments import core_transformer_config_from_args
# from megatron.core import mpu

from megatron.training import get_args, get_timers, get_tokenizer, print_rank_0,get_model
# from megatron.legacy.data.data_samplers import build_pretraining_data_loader
# from megatron.training import pretrain
# from teletron.core.training import pretrain
from teletron.models.wan.wan_producer import producer_process
# from vast.train.configs.config import load_config
from megatron.training.utils import (
    average_losses_across_data_parallel_group
)
import torch.distributed as dist
# from megatron.training.initialize import initialize_megatron

from teletron.train import Trainer
# from teletron.models.wan.light_pipeline import TeletronWanPipeline
from teletron.training.utils import get_batch_on_this_tp_cp_rank_vast
from teletron.datasets.utils import train_valid_test_datasets_provider, load_config_vast

from vast.schedulers.flow_match import FlowMatchScheduler
from megatron.core import mpu



class Config(dict):
    def __init__(self, d=None):
        if d is None:
            d = {}
        super().__init__(d)
        for k, v in d.items():
            if isinstance(v, dict):
                v = Config(v)
            setattr(self, k, v)


def get_batch(data_iterator):
    batch = get_batch_on_this_tp_cp_rank_vast(data_iterator)
    return batch


def extra_args_provider(parser):
    group = parser.add_argument_group(title='dataset')
    group.add_argument('--dataset-type', default="KoalaDataset")
    group.add_argument("--num-frames", type=int, default=9,
                       help='number of frames to train, must be of 4n+1, \
                        overloads yaml if using koala dataset. example: 45')
    group.add_argument("--video-resolution", nargs=2, type=int, default=[1280, 720], 
                       help='video resolution to train, overloads yaml if using koala dataset. \
                       width and height should satisfy: (width or height) // 8 % 2 == 0')
    group.add_argument("--koala-opt", type=str, default="./teletron/datasets/koala.yml", 
                        help="the koala dataset option file")


    group = parser.add_argument_group(title="diffusion")
    group.add_argument("--vae-slicing", action="store_false")
    group.add_argument("--vae-tiling", action="store_false")
    group.add_argument("--flow-resolution-shifting", action="store_true")
    group.add_argument("--flow-base-image-seq-len", type=int, default=256)
    group.add_argument("--flow-max-image-seg-len", type=int, default=4096)
    group.add_argument("--flow-base-shift", type=float, default=0.5)
    group.add_argument("--flow-max-shift", type=float, default=1.15)
    group.add_argument("--flow-shift", type=float, default=1.0)
    group.add_argument("--flow-weighting-scheme", type=str, default="none")
    group.add_argument("--flow-logit-mean", type=float, default=0.0)
    group.add_argument("--flow-logit-std", type=float, default=1.0)
    group.add_argument("--flow-mode-scale", type=float, default=1.29)
    
    group = parser.add_argument_group(title='debug')
    group.add_argument("--debug", action="store_true")
    group.add_argument("--debug_dir", type=str, default="./logs")
    group.add_argument("--sanity-check", action="store_true")

    group.add_argument("--distributed-vae", action="store_true")
    group.add_argument("--distributed-vae-world-size", type=int, default=0,required=False)

    group = parser.add_argument_group(title='training')
    group.add_argument("--task-type", type=str, choices=['wan_flf', 'wan_i2v_prone'], default="wan_flf")

    group = parser.add_argument_group(title='lora_cfg')
    group.add_argument("--lora", type=str, default="Fasle")
    group.add_argument("--lora-rank", type=int,)
    group.add_argument("--lora-alpha", type=int,)
    group.add_argument("--lora-dropout", type=float,)
    group.add_argument("--lora-target-modules", type=str,)
    group.add_argument("--lora-bias", type=str,  default="none")
    group.add_argument("--lora-task-type", type=str,  default="FEATURE_EXTRACTION")
    group.add_argument("--lora-base-model-path", action="store_false")


    return parser
from typing import Callable, Optional, Tuple
import torch.nn.functional as F

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
    num_layers: int = 1
    cross_attn_norm: bool = True
    qk_norm: Optional[str] = "rms_norm_across_heads"
    eps: float = 1e-6
    image_dim: int = 1280
    added_kv_proj_dim: int = 5120
    rope_max_seq_len: int = 1024
    has_image_input: bool = True
    has_image_pos_emb: bool = False


def model_provider(
    pre_process=True, post_process=True, add_encoder=True, add_decoder=True, parallel_output=True
):
    args = get_args()
    config = core_transformer_config_from_args(args)
    config_vast = load_config_vast()
    from teletron.models.wan.parallel_wan_model import  ParallelWanModel
    model = ParallelWanModel(
        dim=WanParams.num_attention_heads * WanParams.attention_head_dim,
        in_dim=WanParams.in_channels,
        ffn_dim=WanParams.ffn_dim,
        out_dim=WanParams.out_channels,
        text_dim=WanParams.text_dim,
        freq_dim=WanParams.freq_dim,
        eps=WanParams.eps,
        patch_size=WanParams.patch_size,
        num_heads=WanParams.num_attention_heads,
        num_layers=WanParams.num_layers,
        has_image_input=WanParams.has_image_input,
        has_image_pos_emb=WanParams.has_image_pos_emb
    )
    if args.debug: 
        from tensorwatch import watch_module_forward_backward
        watch_module_forward_backward(model.transformer, use_megatron=True)
    
    return model

def broadcast_timesteps(input: torch.Tensor):
    tp_cp_src_rank = mpu.get_tensor_context_parallel_src_rank()
    if mpu.get_tensor_context_parallel_world_size() > 1:
        dist.broadcast(input, tp_cp_src_rank, group=mpu.get_tensor_context_parallel_group())


def loss_func(output_tensor):
    loss = output_tensor[0].mean()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss = loss.unsqueeze(0)
    return loss, {"loss": averaged_loss[0]}

def forward_step(data_iterator, model):
    timers = get_timers()
    flow_scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
    flow_scheduler.set_timesteps(1000, training=True)
    prompt_emb = {}
    timers('batch-generator', log_level=2).start()
    batch = get_batch(data_iterator)
    timers('batch-generator').stop()
    latents = batch["latents"]
    noise = torch.randn_like(latents)

    timestep_id = torch.randint(0, flow_scheduler.num_train_timesteps, (1,))
    timestep = flow_scheduler.timesteps[timestep_id].to(
        dtype=torch.bfloat16, device=torch.cuda.current_device()
    )
    extra_input ={}       
    broadcast_timesteps(timestep)
    broadcast_timesteps(noise)
    prompt_emb["context"] = batch["context"]

    image_emb = {}
    image_emb["y"] = batch["image_emb_y"]
    print('y shape', image_emb['y'].shape)
    noisy_latents = flow_scheduler.add_noise(latents, noise, timestep)
    training_target = flow_scheduler.training_target(latents, noise, timestep)
    print('x shape', latents.shape)
    image_emb["clip_feature"] = batch["clip_feature"]

    print("batch[clip_feature].shape", batch["clip_feature"].shape)
    print("noisy_latents.shape", noisy_latents.shape)

    output_tensor_list = model(x=noisy_latents, 
                               timestep=timestep, 
                               context=prompt_emb["context"],
                               clip_feature=image_emb["clip_feature"],
                               y=image_emb["y"])

    return output_tensor_list, loss_func

import debugpy
def wait_for_debugger(rank_to_debug=0, port=5678):
    rank = int(os.environ.get("RANK", "0"))
    if rank == rank_to_debug:
        print(f"[Rank {rank}] Waiting for debugger on port {port}...")
        debugpy.listen(("0.0.0.0", port))
        debugpy.wait_for_client()
        print(f"[Rank {rank}] Debugger attached.")



if __name__ == "__main__":
    trainer = Trainer(model_provider_func=model_provider,
                      dataset_provide_func=train_valid_test_datasets_provider,
                      model_type=ModelType.encoder_or_decoder,
                      extra_args_provider=extra_args_provider,
                      args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
    trainer.pretrain(forward_step_func=forward_step,)

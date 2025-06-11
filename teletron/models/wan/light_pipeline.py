import torch 
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from megatron.core import mpu
from megatron.training import get_args
from vast.schedulers.flow_match import FlowMatchScheduler
from teletron.models.wan.light_model import TeletronWanModel
from typing import Callable, Optional, Tuple

def broadcast_timesteps(input: torch.Tensor):
    tp_cp_src_rank = mpu.get_tensor_context_parallel_src_rank()
    if mpu.get_tensor_context_parallel_world_size() > 1:
        dist.broadcast(input, tp_cp_src_rank, group=mpu.get_tensor_context_parallel_group())


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
    num_layers: int = 40
    cross_attn_norm: bool = True
    qk_norm: Optional[str] = "rms_norm_across_heads"
    eps: float = 1e-6
    image_dim: int = 1280
    added_kv_proj_dim: int = 5120
    rope_max_seq_len: int = 1024
    has_image_input: bool = True
    has_image_pos_emb: bool = False



class TeletronWanPipeline(nn.Module):
    def __init__(self, config, config_vast):
        super().__init__()
        self.config = config
        self.config_vast = config_vast
        self.post_process = True
        self.device = torch.cuda.current_device()

        args = get_args()
        wanConfig = WanParams()
        wanConfig.num_layers = args.num_layers

        if args.task_type == "wan_i2v_prone":
            wanConfig.has_image_input = True
            wanConfig.has_image_pos_emb = False
        elif args.task_type == "wan_flf":
            wanConfig.has_image_input = True
            wanConfig.has_image_pos_emb = True
        else:
            raise NotImplementedError(f"Unknown task type: {args.task_type}")

        self.transformer = TeletronWanModel(wanConfig)
        self.transformer.requires_grad_(True)

        # from tensorwatch import watch_module_forward_backward
        # watch_module_forward_backward(self.transformer, use_megatron=True)

        self.flow_scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.flow_scheduler.set_timesteps(1000, training=True)

        self.dtype = torch.bfloat16


    def __call__(self, batch):
        latents = batch["latents"]
        prompt_emb = {}
        prompt_emb["context"] = batch["context"]
        image_emb = {}
        image_emb["clip_feature"] = batch["clip_feature"]
        image_emb["y"] = batch["image_emb_y"]

        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.flow_scheduler.num_train_timesteps, (1,))
        timestep = self.flow_scheduler.timesteps[timestep_id].to(
            dtype=self.dtype, device=torch.cuda.current_device()
        )
        extra_input = self.prepare_extra_input(latents)
        broadcast_timesteps(timestep)
        broadcast_timesteps(noise)
        noisy_latents = self.flow_scheduler.add_noise(latents, noise, timestep)
        training_target = self.flow_scheduler.training_target(latents, noise, timestep)
        
        noise_pred = self.transformer(
            x=noisy_latents,  # [1, 2, 16, 28, 48] -> [1, 16, 2, 28, 48]
            timestep=timestep,  # [263]
            **prompt_emb,
            **extra_input,
            **image_emb,
            return_dict=False,
            use_gradient_checkpointing=True
        )
        if self.post_process:
            loss = torch.nn.functional.mse_loss(
                noise_pred.float(), training_target.float()
            )
            loss = loss * self.flow_scheduler.training_weight(timestep)
        print("loss", loss)
        return [loss]

    
    def prepare_extra_input(self, latents=None):
        return {}
    
        
    def set_input_tensor(self, input_tensor):
        # self.input_tensor = input_tensor
        # self.transformer.set_input_tensor(input_tensor)
        pass
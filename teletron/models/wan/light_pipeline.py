import torch 
import torch.distributed as dist
from einops import rearrange
from megatron.core import mpu
from vast.pipelines.wan.wan_video import WanVideoPipeline
from vast.schedulers.flow_match import FlowMatchScheduler
from teletron.models.wan.model import WanParams
import numpy as np 
from teletron.models.wan.light_model import TeletronWanModel


def broadcast_timesteps(input: torch.Tensor):
    tp_cp_src_rank = mpu.get_tensor_context_parallel_src_rank()
    if mpu.get_tensor_context_parallel_world_size() > 1:
        dist.broadcast(input, tp_cp_src_rank, group=mpu.get_tensor_context_parallel_group())


class TeletronWanPipeline(WanVideoPipeline):
    def __init__(self, wan_config, config, tokenizer_path=None):
        super().__init__(device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=tokenizer_path)
        self.post_process = True
        self.device = torch.cuda.current_device()
        
        wanConfig = WanParams()
        self.config=config
        self.wan_config = wanConfig
        # self.wan_config.num_layers = 1

        self.transformer = TeletronWanModel(wanConfig)
        # from tensorwatch import watch_module_forward_backward
        # watch_module_forward_backward(self.transformer, use_megatron=True)
        self.transformer.requires_grad_(True)

        self.flow_scheduler_config = wan_config.get("scheduler", dict())
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
        )[0]
        if self.post_process:
            loss = torch.nn.functional.mse_loss(
                noise_pred.float(), training_target.float()
            )
            loss = loss * self.flow_scheduler.training_weight(timestep)
        print("loss", loss)
        return [loss]

    
    def prepare_extra_input(self, latents=None):
        return {}
    
    
    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        """Customized state_dict"""
        return self.transformer.state_dict(prefix=prefix, keep_vars=keep_vars)


    def set_input_tensor(self, input_tensor):
        # self.input_tensor = input_tensor
        # self.transformer.set_input_tensor(input_tensor)
        pass
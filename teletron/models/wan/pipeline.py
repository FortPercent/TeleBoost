# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import os
from typing import List, Optional, Union, Any, Mapping

import functools
from einops import rearrange
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from torch import nn
from tqdm import tqdm
import logging
# from megatron.core.export.data_type import DataType
from vast.schedulers.flow_match import FlowMatchScheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from megatron.core import mpu
# TODO
from vast.models.dit.wan_dit import ModelManager
from vast.models.dit.wan_dit import WanModel
from vast.models.dit.wan_dit import WanTextEncoder
from vast.models.dit.wan_dit import WanVideoVAE
from vast.models.dit.wan_dit import WanImageEncoder
from vast.models.dit.wan_dit import WanPrompter

from teletron.models.wan.model import WanParams, WanVideoTransformer3DModel
from torchvision.transforms.functional import to_pil_image

from megatron.training import get_args
import torch.nn.functional as F
import torch.distributed as dist

from transformers import CLIPImageProcessor, CLIPVisionModel
from tensorwatch import watch_module_forward_backward

logger = logging.getLogger(__name__)

def broadcast_timesteps(input: torch.Tensor):
    tp_cp_src_rank = mpu.get_tensor_context_parallel_src_rank()
    if mpu.get_tensor_context_parallel_world_size() > 1:
        dist.broadcast(input, tp_cp_src_rank, group=mpu.get_tensor_and_context_parallel_group())

class WanPipeline(nn.Module):
    def __init__(self, wan_config,config,tokenizer_path=None):
        super().__init__()
        text_encoder_path = wan_config.text_encoder_path
        tokenizer_path = os.path.join(os.path.dirname(text_encoder_path), "google/umt5-xxl")
    
        self.pre_process = mpu.is_pipeline_first_stage()
        self.post_process = mpu.is_pipeline_last_stage()
        self.input_tensor = None

        print("Initializing Wanpipeline...")

        self.vae=WanVideoVAE()
        self.vae.model.encoder.conv1.weight.norm()
        #import ipdb;ipdb.set_trace()
        self.tiler_kwargs = {
            "tiled": wan_config.get("tiled", True), 
            "tile_size": wan_config.get("tile_size", (34, 34)), 
            "tile_stride": wan_config.get("tile_stride", (18, 16))
            }
        self.text_encoder=WanTextEncoder()
        self.image_encoder=WanImageEncoder()
        self.prompter = WanPrompter()
        self.prompter.fetch_models(self.text_encoder)
        self.prompter.fetch_tokenizer(tokenizer_path)

        

        wanConfig = WanParams()

        self.config = config
        self.wan_config = wan_config
        print("Initialize WanVideoTransformer3DModel")
        self.transformer = WanVideoTransformer3DModel(wanConfig,config)
        # from tensorwatch import watch_module_forward_backward
        # watch_module_forward_backward(self.transformer, use_megatron=True)
        watch_module_forward_backward(self.transformer, use_megatron=True)
        print("WanVideoTransformer3DModel Initialized")
        # self.vae_scale_factor = 2 ** (len(self.vae.params.ch_mult))
        
        self.flow_scheduler_config = wan_config.get("scheduler", dict())
        self.flow_scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.flow_scheduler.set_timesteps(1000, training=True)

        self.dtype = torch.bfloat16
        print(f"wan dtype: {self.dtype}")
        
    def forward(self, batch):
        DEBUG_FIX = True
        if DEBUG_FIX:
            torch.random.manual_seed(1234)
            batch['dense_prompt'] = ["a cute cat"]
            batch['images'] = torch.ones_like(batch["images"])
            batch['raw_first_image'] = torch.ones_like(batch["images"])
            batch['raw_last_image'] = torch.ones_like(batch["images"])
        #import ipdb; ipdb.set_trace()

        if self.pre_process:
            with torch.no_grad():
                torch.manual_seed(1234) 
                prompt_emb = self.encode_prompt(batch["dense_prompt"][0])
                latents = self.encode_video(rearrange(batch["images"], "b t c h w -> b c t h w").to(dtype=self.dtype, device=torch.cuda.current_device()), **self.tiler_kwargs)[0]

                _, num_frames, _, height, width = batch["images"].shape
                if 'raw_last_image' in batch:
                    raw_first_image = batch["raw_first_image"]
                    pil_first_image = to_pil_image(raw_first_image[0][0].cpu().permute(1,2,0).numpy().astype(np.uint8))
                    raw_last_image = batch['raw_last_image']
                    pil_last_image = to_pil_image(raw_last_image[0][0].cpu().permute(1,2,0).numpy().astype(np.uint8))
                    image_emb = self.encode_first_last_image(pil_first_image, pil_last_image, num_frames, height, width)
                else:
                    raw_first_image = batch["raw_first_image"]
                    pil_image = to_pil_image(raw_first_image[0][0].cpu().permute(1,2,0).numpy().astype(np.uint8))
                    image_emb = self.encode_image(pil_image, num_frames, height, width)
            latents = latents.unsqueeze(0).to(dtype=self.dtype, device=torch.cuda.current_device())

            # Data
            prompt_emb["context"] = prompt_emb["context"][0].to(dtype=self.dtype, device=torch.cuda.current_device())
            prompt_emb["context"] = prompt_emb["context"].unsqueeze(0)
            
            if "clip_feature" in image_emb:
                image_emb["clip_feature"] = image_emb["clip_feature"][0].to(dtype=self.dtype, device=torch.cuda.current_device()).unsqueeze(0)
            if "y" in image_emb:
                image_emb["y"] = image_emb["y"][0].to(dtype=self.dtype, device=torch.cuda.current_device()).unsqueeze(0)
        torch.manual_seed(1234)
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.flow_scheduler.num_train_timesteps, (1,))
        timestep = self.flow_scheduler.timesteps[timestep_id].to(dtype=self.dtype, device=torch.cuda.current_device())
        extra_input = self.prepare_extra_input(latents)
        broadcast_timesteps(timestep)
        broadcast_timesteps(noise)
        noisy_latents = self.flow_scheduler.add_noise(latents, noise, timestep)
        training_target = self.flow_scheduler.training_target(latents, noise, timestep)
        # noisy_latents, timestep=timestep, **prompt_emb, **extra_input, **image_emb,
        if DEBUG_FIX:
            prompt_emb['context'] = torch.ones_like(prompt_emb['context'])
            noisy_latents = torch.ones_like(noisy_latents)
            image_emb['clip_feature'] = torch.ones_like(image_emb['clip_feature'])
            image_emb['y'] = torch.ones_like(image_emb['y'])
            # timestep = [1]

        noise_pred = self.transformer(
                x=noisy_latents,  # [1, 2, 16, 28, 48] -> [1, 16, 2, 28, 48]
                timestep=timestep,  # [263]
                **prompt_emb, **extra_input,**image_emb,
                return_dict=False,
            )[0]
        
        if self.post_process:
            loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
            loss = loss * self.flow_scheduler.training_weight(timestep)

        # # offload
        # self.vae.to(device="cpu")
        print(loss.mean())
        import ipdb; ipdb.set_trace()
        return [loss]
    
    def forward_vae(self, images):
        images = images.to(self.vae.dtype)
        
        import torch.distributed as dist
        rank = dist.get_rank()       
        torch.save(images, f"images_{rank}.pt") 
        with torch.no_grad():

            images = rearrange(images, "b f c h w -> b c f h w")
            latents = self.vae.encode(images)
            latents = latents.latent_dist.sample()

        latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
        latents = (latents - latents_mean) * latents_std
        return latents

    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive, device=torch.cuda.current_device())
        return {"context": prompt_emb}
    
    
    def encode_image(self, image, num_frames, height, width, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        image = self.preprocess_image(image.resize((width, height))).to(torch.cuda.current_device())
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=torch.cuda.current_device())
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
        y = self.vae.encode([vae_input.to(dtype=self.dtype, device=torch.cuda.current_device())], device=torch.cuda.current_device(), 
                            tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=self.dtype, device=torch.cuda.current_device())
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.dtype, device=torch.cuda.current_device())
        y = y.to(dtype=self.dtype, device=torch.cuda.current_device())
        return {"clip_feature": clip_context, "y": y}
    
    def encode_first_last_image(self, pil_first_image, pil_last_image, num_frames, height, width, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        first_image = self.preprocess_image(pil_first_image.resize((width, height))).to(torch.cuda.current_device())
        last_image = self.preprocess_image(pil_last_image.resize((width, height))).to(torch.cuda.current_device())
        # if self.dit.has_image_pos_emb:
        #     clip_context = torch.cat([self.image_encoder.encode_image([first_image]), 
        #                             self.image_encoder.encode_image([last_image])], dim=1)
        # else:
        #     clip_context = self.image_encoder.encode_image([first_image])
        clip_context = torch.cat([self.image_encoder.encode_image([first_image]), 
                        self.image_encoder.encode_image([last_image])], dim=1)
        msk = torch.ones(1, num_frames, height//8, width//8, device=torch.cuda.current_device())
        msk[:, 1:-1] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]

        vae_input = torch.concat([first_image.transpose(0, 1), 
                                  torch.zeros(3, num_frames-2, height, width).to(first_image.device), 
                                  last_image.transpose(0, 1)], dim=1)
        y = self.vae.encode([vae_input.to(dtype=self.dtype, device=torch.cuda.current_device())], device=torch.cuda.current_device(), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=self.dtype, device=torch.cuda.current_device())
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.dtype, device=torch.cuda.current_device())
        y = y.to(dtype=self.dtype, device=torch.cuda.current_device())
        import ipdb; ipdb.set_trace()
        return {"clip_feature": clip_context, "y": y}

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames
    
    
    def prepare_extra_input(self, latents=None):
        return {}
    
    
    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=torch.cuda.current_device(), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    
    
    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(latents, device=torch.cuda.current_device(), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames

    def check_resize_height_width(self, height, width):
        if height % self.height_division_factor != 0:
            height = (height + self.height_division_factor - 1) // self.height_division_factor * self.height_division_factor
            print(f"The height cannot be evenly divided by {self.height_division_factor}. We round it up to {height}.")
        if width % self.width_division_factor != 0:
            width = (width + self.width_division_factor - 1) // self.width_division_factor * self.width_division_factor
            print(f"The width cannot be evenly divided by {self.width_division_factor}. We round it up to {width}.")
        return height, width


    def preprocess_image(self, image):
        image = torch.Tensor(np.array(image, dtype=np.float32) * (2 / 255) - 1).permute(2, 0, 1).unsqueeze(0)
        return image
    

    def preprocess_images(self, images):
        return [self.preprocess_image(image) for image in images]
    

    def vae_output_to_image(self, vae_output):
        image = vae_output[0].cpu().float().permute(1, 2, 0).numpy()
        image = Image.fromarray(((image / 2 + 0.5).clip(0, 1) * 255).astype("uint8"))
        return image
    

    def vae_output_to_video(self, vae_output):
        video = vae_output.cpu().permute(1, 2, 0).numpy()
        video = [Image.fromarray(((image / 2 + 0.5).clip(0, 1) * 255).astype("uint8")) for image in video]
        return video

    
    def merge_latents(self, value, latents, masks, scales, blur_kernel_size=33, blur_sigma=10.0):
        if len(latents) > 0:
            blur = GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma)
            height, width = value.shape[-2:]
            weight = torch.ones_like(value)
            for latent, mask, scale in zip(latents, masks, scales):
                mask = self.preprocess_image(mask.resize((width, height))).mean(dim=1, keepdim=True) > 0
                mask = mask.repeat(1, latent.shape[1], 1, 1).to(dtype=latent.dtype, device=latent.device)
                mask = blur(mask)
                value += latent * mask * scale
                weight += mask * scale
            value /= weight
        return value


    def control_noise_via_local_prompts(self, prompt_emb_global, prompt_emb_locals, masks, mask_scales, inference_callback, special_kwargs=None, special_local_kwargs_list=None):
        if special_kwargs is None:
            noise_pred_global = inference_callback(prompt_emb_global)
        else:
            noise_pred_global = inference_callback(prompt_emb_global, special_kwargs)
        if special_local_kwargs_list is None:
            noise_pred_locals = [inference_callback(prompt_emb_local) for prompt_emb_local in prompt_emb_locals]
        else:
            noise_pred_locals = [inference_callback(prompt_emb_local, special_kwargs) for prompt_emb_local, special_kwargs in zip(prompt_emb_locals, special_local_kwargs_list)]
        noise_pred = self.merge_latents(noise_pred_global, noise_pred_locals, masks, mask_scales)
        return noise_pred
    

    def extend_prompt(self, prompt, local_prompts, masks, mask_scales):
        local_prompts = local_prompts or []
        masks = masks or []
        mask_scales = mask_scales or []
        extended_prompt_dict = self.prompter.extend_prompt(prompt)
        prompt = extended_prompt_dict.get("prompt", prompt)
        local_prompts += extended_prompt_dict.get("prompts", [])
        masks += extended_prompt_dict.get("masks", [])
        mask_scales += [100.0] * len(extended_prompt_dict.get("masks", []))
        return prompt, local_prompts, masks, mask_scales

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        """Customized state_dict"""
        return self.transformer.state_dict(prefix=prefix, keep_vars=keep_vars)

    # def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
    #     """Customized load."""
    #     if not isinstance(state_dict, Mapping):
    #         raise TypeError(f"Expected state_dict to be dict-like, got {type(state_dict)}.")

    #     missing_keys, unexpected_keys = self.transformer.load_state_dict(state_dict, False)

    #     if missing_keys is not None:
    #         logger.info(f"Missing keys in state_dict: {missing_keys}.")
    #     if unexpected_keys is not None:
    #         logger.info(f"Unexpected key(s) in state_dict: {unexpected_keys}.")
    
    def set_input_tensor(self, input_tensor):
        # self.input_tensor = input_tensor
        # self.transformer.set_input_tensor(input_tensor)
        pass
import types
from teletron.models.teleai.models.dit.teleai_dit import ModelManager
from teletron.models.teleai import TeleaiModel
from teletron.models.teleai.teleai_model import sinusoidal_embedding_1d
from teletron.models.teleai.models.dit.teleai_dit import TeleaiTextEncoder
from teletron.models.teleai.models.dit.teleai_dit import TeleaiVideoVAE
from teletron.models.teleai.models.dit.teleai_dit import TeleaiImageEncoder
from teletron.models.teleai.models.dit.teleai_dit import TeleaiPrompter

from teletron.models.teleai.schedulers.flow_match import FlowMatchScheduler
# from scheduler import FlowMatchScheduler
from base_pipeline import BasePipeline
import torch, os
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional

# from teleai.models.dit.wan_dit.wan_video_dit import sinusoidal_embedding_1d

from PIL import Image

def resize_and_crop(image, target_size):
    original_width, original_height = image.size
    target_width, target_height = target_size
    
    scale = max(target_width / original_width, target_height / original_height)
    
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    left = (new_width - target_width) / 2
    top = (new_height - target_height) / 2
    right = (new_width + target_width) / 2
    bottom = (new_height + target_height) / 2
    
    cropped_image = resized_image.crop((left, top, right, bottom))
    
    return cropped_image



class TeleaiVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = TeleaiPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: TeleaiTextEncoder = None
        self.image_encoder: TeleaiImageEncoder = None
        self.dit: TeleaiModel = None
        self.vae: TeleaiVideoVAE = None
        self.model_names = ['text_encoder', 'dit', 'vae', 'image_encoder']
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.use_unified_sequence_parallel = False
        

    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model("teleai_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_manager.fetch_model("teleai_video_dit")
        self.vae = model_manager.fetch_model("teleai_video_vae")
        self.image_encoder = model_manager.fetch_model("teleai_video_image_encoder")


    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None,):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = TeleaiVideoPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        return pipe
    
    @staticmethod
    def load_moe_model_from_model_manager(
            model_manager: ModelManager, 
            moe_config,
            torch_dtype=None, 
            device=None,
            model_offload=None 
        ):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = TeleaiVideoPipeline(device=device, torch_dtype=torch_dtype)

        # load vae and encoders
        text_encoder_model_and_path = model_manager.fetch_model("teleai_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            pipe.text_encoder, tokenizer_path = text_encoder_model_and_path
            pipe.prompter.fetch_models(pipe.text_encoder)
            pipe.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        pipe.vae = model_manager.fetch_model("teleai_video_vae")
        pipe.image_encoder = model_manager.fetch_model("teleai_video_image_encoder")

        pipe.model_offload = model_offload
        if pipe.model_offload:
            pipe.current_gpu_model_name = None
            pipe.load_stream = torch.cuda.Stream()
            pipe.compute_stream = torch.cuda.Stream()
            pipe.cpu_stream = torch.cuda.Stream()
        # fetch moe models
        setattr(pipe, 'moe_config', moe_config)
        for model_name, model_info in moe_config.items():
            print(f"Init {model_name}")
            dit_path = model_info["dit_path"]
            setattr(pipe, model_name, model_manager.fetch_model("teleai_video_dit", file_path=dit_path))
        
        return pipe
    
    def denoising_model_name_list(self):
        if hasattr(self, 'moe_config'):
            model_list = []
            for model_name, _ in self.moe_config.items():
                model_list.append(model_name)
            return model_list
        return ["dit"]

    def denoising_model(self):
        if hasattr(self, 'moe_config'):
            model_dict = {}
            for model_name, _ in self.moe_config.items():
                model_dict[model_name] = getattr(self, model_name)
            return model_dict
        return self.dit

    def alloc_denoising_model(self, timestep, return_name=False):
        if not hasattr(self, 'moe_config'):
            return self.dit
        else:
            for model_name, model_info in self.moe_config.items():
                timestep_range = model_info["timestep_range"]
                if timestep >= timestep_range[0] and timestep <= timestep_range[1]:
                    if return_name:
                        return model_name
                    else:
                        return getattr(self, model_name)
            raise ValueError(f"Alloc Denoising Model Fail., TimeStep: {timestep}")

    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive, device=self.device)
        return {"context": prompt_emb}
    
    def encode_image(self, image, num_frames, height, width, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        image = self.preprocess_image(resize_and_crop(image, (width, height))).to(self.device)
        print("image dtype", image.dtype)
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
        y = self.vae.encode([vae_input.to(dtype=self.torch_dtype, device=self.device)], device=self.device, 
                            tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=self.torch_dtype, device=self.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}
    
    def encode_image_tensor(self, image, num_frames, height, width, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        b, t, c, h, w = image.shape
        assert h == height and w == width
        image = image.float() * (2 / 255) - 1
        
        msk = torch.ones(b, num_frames, height//8, width//8, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1) # (b, t + 3, h, w)
        msk = msk.view(b, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2) # b, c, t, h, w
        vae_input = torch.concat([image.transpose(1, 2), torch.zeros(b, 3, num_frames-1, height, width).to(image.device)], dim=2) # b, c, t, h, w
        
        self.vae.to(self.device)
        y = self.vae.encode(vae_input.to(dtype=self.torch_dtype, device=self.device), device=self.device, 
                            tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        self.vae.cpu() # offload
        
        y = y.to(dtype=self.torch_dtype, device=self.device)
        y = torch.concat([msk, y], dim=1) #, b, c, t, h, w
        
        self.image_encoder.to(self.device)
        clip_context = self.image_encoder.encode_tensor(image.to(self.device))
        self.image_encoder.cpu() # offload

        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}
    
    
    def encode_first_last_image(self, pil_first_image, pil_last_image, num_frames, height, width, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        first_image = self.preprocess_image(pil_first_image.resize((width, height))).to(self.device)
        last_image = self.preprocess_image(pil_last_image.resize((width, height))).to(self.device)
        # if self.dit.has_image_pos_emb:
        #     clip_context = torch.cat([self.image_encoder.encode_image([first_image]), 
        #                             self.image_encoder.encode_image([last_image])], dim=1)
        # else:
        #     clip_context = self.image_encoder.encode_image([first_image])
        clip_context = torch.cat([self.image_encoder.encode_image([first_image]), 
                        self.image_encoder.encode_image([last_image])], dim=1)
        msk = torch.ones(1, num_frames, height//8, width//8, device=self.device)
        msk[:, 1:-1] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]

        vae_input = torch.concat([first_image.transpose(0, 1), 
                                  torch.zeros(3, num_frames-2, height, width).to(first_image.device), 
                                  last_image.transpose(0, 1)], dim=1)
        y = self.vae.encode([vae_input.to(dtype=self.torch_dtype, device=self.device)], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=self.torch_dtype, device=self.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}
    
    def encode_image_with_mask(self, image, height, width, msk, ref_images, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        clip_context = self.image_encoder.encode_image([image])
        ref_images = rearrange(ref_images, 'b t c h w -> b c t h w')
        y = self.encode_video(ref_images.to(dtype=self.torch_dtype, device=self.device),
                            tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.unsqueeze(0)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        msk = msk.transpose(1, 2)
        y = torch.concat([msk, y], dim=1)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames
    
    def prepare_extra_input(self, latents=None):
        return {}
    
    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    
    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames
    
    def prepare_unified_sequence_parallel(self):
        return {"use_unified_sequence_parallel": self.use_unified_sequence_parallel}
    
    def prefetch_model_from_cpu(self, model_name):
        with torch.cuda.stream(self.load_stream):
            model = getattr(self, model_name).cuda()
            setattr(self, model_name, model)
        self.prefetch_gpu_model_name = model_name
        # return model
    
    def denoising_with_dit(self, latents, prompt_emb_posi, image_emb, extra_input, progress_bar_cmd, cfg_scale, prompt_emb_nega):
        if self.model_offload:
            self.prefetch_model_from_cpu(self.alloc_denoising_model(self.scheduler.timesteps[0], return_name=True))
        history_video = []
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            print("timestep: ",timestep)
            if progress_id % 10 == 0:
                history_video.append(latents)
            if self.model_offload:
                if self.prefetch_gpu_model_name == self.alloc_denoising_model(timestep, return_name=True):
                    print(f"switch to prefetched model: {self.prefetch_gpu_model_name}")
                    self.load_stream.synchronize()
                    current_dit = getattr(self, self.prefetch_gpu_model_name) # ToDO
                    current_gpu_model_name = self.prefetch_gpu_model_name
                dit = current_dit 
            else:
                dit = self.alloc_denoising_model(timestep)
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            if self.model_offload:
                with torch.cuda.stream(self.compute_stream):
                    # noise_pred = dit(latents, timestep=timestep, **prompt_emb_posi, **image_emb, **extra_input)
                    noise_pred_posi = dit(latents, timestep=timestep, **prompt_emb_posi, **image_emb, **extra_input)
                    if cfg_scale != 1.0:
                        noise_pred_nega = dit(latents, timestep=timestep, **prompt_emb_nega, **image_emb, **extra_input)
                        noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                    else:
                        noise_pred = noise_pred_posi
            else:
                noise_pred = dit(latents, timestep=timestep, **prompt_emb_posi, **image_emb, **extra_input)
            
            if self.model_offload:
                if progress_id != len(self.scheduler.timesteps) - 1:
                    next_timestep_model_name = self.alloc_denoising_model(self.scheduler.timesteps[progress_id + 1], return_name=True)
                    if current_gpu_model_name != next_timestep_model_name:
                        print(f"start prefetching model: {next_timestep_model_name}")
                        self.prefetch_model_from_cpu(next_timestep_model_name)
                    
                self.compute_stream.synchronize()
            
            if self.model_offload:
                if current_gpu_model_name != next_timestep_model_name:
                    print(f"offload model: {current_gpu_model_name}")
                    with torch.cuda.stream(self.cpu_stream):
                        model = getattr(self, current_gpu_model_name).cpu()
                        setattr(self, current_gpu_model_name, model)

            # Scheduler
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)
        return history_video, latents
    
    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image=None, # PIL Image
        end_image=None,
        input_video=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        tea_cache_l1_thresh=None,
        tea_cache_model_id="",
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
        cn_images=None, # control img, shape of (b c t h w) with norm(0.5)
        verbose=False
    ):
        # Parameter check
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")
        
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Initialize noise
        noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=rand_device, dtype=torch.float32)
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        if input_video is not None:
            self.load_models_to_device(['vae'])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2).to(dtype=self.torch_dtype, device=self.device)
            latents = self.encode_video(input_video.to(dtype=self.torch_dtype, device=self.device), **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = noise

        if cn_images is not None:
            self.load_models_to_device(['vae'])
            # no process here
            cn_images = self.encode_video(cn_images.to(dtype=self.torch_dtype, device=self.device), **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
            
        # Encode image
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            if end_image is not None:
                image_emb = self.encode_first_last_image(input_image, end_image, num_frames, height, width)
            else:
                image_emb = self.encode_image(input_image, num_frames, height, width) # without tilling
        else:
            image_emb = {}
        
        # Extra input
        extra_input = self.prepare_extra_input(latents)
        
        # TeaCache
        tea_cache_posi = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}
        tea_cache_nega = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}
        
        # Unified Sequence Parallel
        usp_kwargs = self.prepare_unified_sequence_parallel()

        # Denoise
        self.load_models_to_device(self.denoising_model_name_list())
        history_video, latents = self.denoising_with_dit(latents, prompt_emb_posi, image_emb, extra_input, progress_bar_cmd, cfg_scale, prompt_emb_nega)

        
        # history_video = []
        # for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            
        #     if progress_id % 10 == 0:
        #         history_video.append(latents)
        #     breakpoint()
        #     timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
        #     dit = self.alloc_denoising_model(timestep)
        #     # Inference
        #     # noise_pred_posi = model_fn_wan_video(
        #     #     dit, latents, cn_images=cn_images, timestep=timestep, 
        #     #     **prompt_emb_posi, **image_emb, **extra_input, **tea_cache_posi, **usp_kwargs
        #     #     )
        #     # # if cfg_scale != 1.0:
        #     #     noise_pred_nega = model_fn_wan_video(
        #     #         dit, latents, cn_images=cn_images, timestep=timestep, 
        #     #         **prompt_emb_nega, **image_emb, **extra_input, **tea_cache_nega, **usp_kwargs
        #     #         )
        #     #     noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        #     # else:
        #     #     noise_pred = noise_pred_posi
        #     noise_pred = dit(latents, timestep=timestep, **prompt_emb_posi, **image_emb, **extra_input)

        #     # Scheduler
        #     latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)

        # Decode
        self.load_models_to_device(['vae'])

        history_frames = []
        if verbose:
            for latent in history_video:
                frames = self.decode_video(latent, **tiler_kwargs)
                frames = self.tensor2video(frames[0])
                history_frames.append(frames)
        
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        history_frames.append(frames)

        return history_frames if verbose else frames


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: TeleaiModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states


def model_fn_wan_video(
    dit: TeleaiModel,
    x: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    cn_images=None, 
    **kwargs,
):
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)
    
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    if dit.has_image_input:
        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    if cn_images is not None:
        x = torch.cat([x, cn_images], dim=1)  # (b, c_x + c_y, f, h, w)
    
    x, (f, h, w) = dit.patchify(x)
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
    
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        # blocks
        if use_unified_sequence_parallel:
            if dist.is_initialized() and dist.get_world_size() > 1:
                x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
        for block in dit.blocks:
            x = block(x, context, t_mod, freqs)
        if tea_cache is not None:
            tea_cache.store(x)

    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
    x = dit.unpatchify(x, (f, h, w))
    return x

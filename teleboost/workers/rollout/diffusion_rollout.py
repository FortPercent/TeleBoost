# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single GPU model.
Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model to perform generation.
"""
import contextlib
import math
import random
import os

import torch
import torch.distributed
from diffusers.image_processor import VaeImageProcessor
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm.auto import tqdm
from transformers import GenerationConfig

from verl import DataProto
from verl.utils.device import get_device_id, get_device_name, get_nccl_backend
from verl.utils.torch_functional import get_response_mask
from wan.modules.vae import WanVAE

from verl.workers.rollout.base import BaseRollout  # import-verl: verl is pip-installed
from recipe.teleboost.algorithms.sigma_schedule import compute_sde_step
import logging

__all__ = ['DiffusionRollout']
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def _normalize_wan22_timestep(t, sigma):
    if sigma is not None:
        if torch.is_tensor(sigma):
            return sigma.detach().flatten()[0].float().item()
        return float(sigma)
    if t is None:
        return None
    if torch.is_tensor(t):
        t_val = t.detach().flatten()[0].float().item()
    else:
        t_val = float(t)
    if t_val > 1.0:
        t_val = t_val / 1000.0
    return t_val

def _select_wan22_guide_scale(guide_scale, t, sigma, boundary):
    if isinstance(guide_scale, (list, tuple)) and len(guide_scale) >= 2:
        t_val = _normalize_wan22_timestep(t, sigma)
        if t_val is None:
            return guide_scale[0]
        return guide_scale[1] if t_val >= boundary else guide_scale[0]
    return guide_scale


def _compute_flow_grpo_window(window_size: int, window_range: tuple[int, int], num_steps: int):
    if window_size <= 0:
        return None
    start_min, start_max = window_range
    start_max = min(start_max, num_steps)
    if start_max - window_size < start_min:
        start = start_min
    else:
        start = random.randint(start_min, start_max - window_size)
    end = start + window_size
    return (start, end)


class DiffusionRollout(BaseRollout):

    def __init__(self, module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = module
        # bfloat16 (not float16): the Wan VAE's intermediate activations
        # routinely exceed fp16's ~65504 max under autocast, producing NaN
        # decoded videos that silently propagate through the
        # ``(video_frames * 255).astype(uint8)`` cast (warns "invalid value
        # encountered in cast" but yields zeros), then NaN HPS scores ->
        # NaN advantages -> NaN grads.  bfloat16 has the same memory cost
        # as fp16 but the dynamic range of fp32, so VAE decode stays
        # finite.  Verified empirically on Wan2.2-T2V-A14B at sampling
        # steps in {4, 10}.
        vae_dtype = torch.bfloat16
        vae = WanVAE(
            vae_pth=os.path.join(self.config.model.vae_model_path),
            dtype=vae_dtype,
        )
        self.vae_module = vae

        # ----- VIPO: pixel-weight feature flag ---------------------------
        # Enabled via ``actor_rollout_ref.pixel_weight.enable`` in Hydra.
        # When False (the default) the rollout keeps the original scalar
        # log-probability behaviour.  When True it preserves spatial dims
        # so dense (T,H,W) log-probs can flow into a pixel-weighted loss.
        pixel_cfg = self.config.get("pixel_weight", {}) or {}
        self._pixel_enable = bool(pixel_cfg.get("enable", False))
        self._pixel_cfg = pixel_cfg

        # ----- σ_t SDE form (DanceGRPO vs Flow-GRPO) ---------------------
        # See ``recipe/teleboost/algorithms/sigma_schedule.py``.  Default
        # ``"dancegrpo"`` keeps the existing constant-eta noise schedule
        # byte-equivalent to the pre-registry implementation.
        self._sigma_form = self.config.actor.get("sigma_form", "dancegrpo")

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        torch.cuda.memory._set_allocator_settings(f"expandable_segments:{False}")
        context=prompts.batch['context']
        context_orig_lengths = prompts.batch['context_orig_lengths']
        # caption=prompts.non_tensor_batch['caption']
        neg_context = prompts.batch['null_context']
        sigma_schedule = prompts.batch["sigma_schedule"]
        input_latents = prompts.batch["input_latents"]
        latent_shape=input_latents[0].shape
        patch_size = [1, 2, 2]
        seq_len = math.ceil(
                (latent_shape[2] * latent_shape[3]) / (patch_size[1] * patch_size[2]) * latent_shape[1]
            )
        
        B = prompts.batch.batch_size[0]
        

        all_latents = []
        all_log_probs = []
        all_video_frames = []
        all_video_ids = []

        flow_cfg = self.config.get("flow_grpo", {})
        if not flow_cfg.get("enable", False):
            window_size = 0
        else:
            window_size = int(flow_cfg.get("sde_window_size", 0) or 0)
        window_range = tuple(flow_cfg.get("sde_window_range", (0, self.config.sampling_steps)))
        flow_window = _compute_flow_grpo_window(window_size, window_range, self.config.sampling_steps)

        batch_indices = torch.chunk(torch.arange(B), B // self.config.rollout.ulysses_sequence_parallel_size)
        
        grpo_sample = True
        # self.module.eval()
        self.vae_module.model.to(get_device_id(), dtype=torch.bfloat16)
        for index, batch_idx in enumerate(batch_indices):
            progress_bar = tqdm(range(0, self.config.sampling_steps), desc="WAN Sampling Progress")
            # batch_captions = [caption[i] for i in batch_idx]
            batch_contexts = [context[i].to(get_device_id()) for i in batch_idx]
            batch_neg_context = [neg_context[i].to(get_device_id()) for i in batch_idx]
            batch_context_orig_lengths = [context_orig_lengths[i] for i in batch_idx]
            batch_input_latents = [input_latents[i] for i in batch_idx]
            
            for i in range(len(batch_contexts)):
                batch_contexts[i] = batch_contexts[i][:batch_context_orig_lengths[i]]
            
            # ---- log info ----

            # with torch.no_grad(): 
            wan_outputs = self.run_wan_sample_step(
                batch_input_latents,
                progress_bar,
                sigma_schedule[0],
                self.module,
                batch_contexts,
                batch_neg_context,
                seq_len,
                grpo_sample,
                flow_window=flow_window,
            )

            if len(wan_outputs) == 5:
                _, final_latents, batch_latents, batch_log_probs, batch_prev_sample_mean = wan_outputs
            else:
                _, final_latents, batch_latents, batch_log_probs = wan_outputs
                batch_prev_sample_mean = None

            all_latents.append(batch_latents.unsqueeze(0))
            all_log_probs.append(batch_log_probs.unsqueeze(0))
            if batch_prev_sample_mean is not None:
                if "all_prev_sample_mean" not in locals():
                    all_prev_sample_mean = []
                all_prev_sample_mean.append(batch_prev_sample_mean.unsqueeze(0))
           
            
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # Cast final_latents to fp32 for the VAE decoder
                final_latents_vae = final_latents.to(dtype=torch.float32)

                decoded_videos = self.vae_module.decode([final_latents_vae])

                video_frames = decoded_videos[0]
                

                
                # Post-process: normalize from [-1, 1] to [0, 1]
                video_frames = (video_frames + 1.0) / 2.0
                video_frames = torch.clamp(video_frames, 0, 1)
                
                
                # Ensure video_frames is (C, T, H, W)
                if video_frames.dim() == 4:
                    # Subsample frames at 15 FPS for the preview
                    fps=15
                    video_id = video_frames[:, ::fps, :, :]
                    C, T, H, W = video_frames.shape
                    # print(video_frames.shape)
                        
                    # Convert to numpy (T, H, W, C)
                    video_id = video_id.permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, C)
                    # video_id = video_frames.cpu().numpy()
                    
                    import numpy as np
                    video_id = (video_id * 255).astype(np.uint8)
                        
                        # If single-channel, expand to 3 channels
                    if C == 1:
                        video_id = id.repeat(video_id, 3, axis=-1)
                        
                all_video_ids.append(video_id)
                
                video_frames = video_frames.unsqueeze(0)
                
            all_video_frames.append(video_frames)
            
        self.vae_module.model.to("cpu", dtype=torch.float32)
        torch.cuda.empty_cache()
        
        # Everything except all_video_paths is a single tensor
        if len(all_latents) > 1:
            all_latents = torch.cat(all_latents, dim=0)
            all_log_probs = torch.cat(all_log_probs, dim=0)
            all_video_frames = torch.cat(all_video_frames, dim=0)
            if "all_prev_sample_mean" in locals():
                all_prev_sample_mean = torch.cat(all_prev_sample_mean, dim=0)
        else:
            all_latents = all_latents[0]
            all_log_probs = all_log_probs[0]
            all_video_frames = all_video_frames[0]
            if "all_prev_sample_mean" in locals():
                all_prev_sample_mean = all_prev_sample_mean[0]

        if flow_window is None:
            timestep_value = [int(sigma * 1000) for sigma in sigma_schedule[0].squeeze()][:self.config.sampling_steps]
        else:
            window_indices = list(range(flow_window[0], flow_window[1]))
            timestep_value = [int(sigma_schedule[0].squeeze()[i] * 1000) for i in window_indices]
        
        timestep_values = [timestep_value[:] for _ in range(B)]

        timesteps =  torch.tensor(timestep_values, device=get_device_id(), dtype=torch.long)
       
        latents=all_latents[:, :-1]
        next_latents=all_latents[:, 1:]

        batch_dict = {
            "context_orig_lengths":context_orig_lengths,
            "contexts": context,
            "null_context":neg_context,
            "latents": latents,
            "next_latents": next_latents,
            "log_probs": all_log_probs,
            "video_frames": all_video_frames,
            "sigma_schedule": sigma_schedule,
            'timesteps':timesteps[:, :-1]
        }
        if "all_prev_sample_mean" in locals():
            batch_dict["prev_sample_mean"] = all_prev_sample_mean
        if flow_window is not None:
            window_indices = list(range(flow_window[0], flow_window[1]))
            timestep_indices = torch.tensor(window_indices, device=get_device_id(), dtype=torch.long)
            timestep_indices = timestep_indices.unsqueeze(0).repeat(B, 1)
            batch_dict["timestep_indices"] = timestep_indices

        batch = TensorDict(batch_dict, batch_size=B)


        non_tensor_batch = prompts.non_tensor_batch
        non_tensor_batch['video_ids'] = np.array(all_video_ids)

        # ----- VIPO: attach per-sample pixel-weight maps -----------------
        # Only executed when the feature flag is on.  We derive the
        # spatial/temporal target sizes from the produced latents so the
        # map aligns with whatever the actor will receive.  Failures in
        # DINOv2 fall back to an all-ones map inside the util (graceful
        # degradation to baseline GRPO for that sample).
        if self._pixel_enable:
            try:
                videos = all_video_frames
                if videos.ndim != 5:
                    raise ValueError(
                        f"Expected video_frames of shape (B, C, T, H, W); got {tuple(videos.shape)}"
                    )
                # latents here has shape (B, num_steps, C, T_lat, H_lat, W_lat);
                # we want T_lat, H_lat, W_lat from dims 3/4/5.
                target_time = int(latents.shape[3])
                target_size = (int(latents.shape[4]), int(latents.shape[5]))

                # Local import to keep the pixel-weight code path lazy -
                # we don't want to force DINOv2/transformers imports on
                # users who never enable VIPO.
                from recipe.teleboost.pixel_weight_utils import (
                    compute_batch_pixel_weight_maps,
                )

                pixel_weight_maps = compute_batch_pixel_weight_maps(
                    videos=videos,
                    target_size=target_size,
                    target_time=target_time,
                    device=videos.device,
                    model_path=self._pixel_cfg.get("model_path", "facebook/dinov2-large"),
                    pca_method=self._pixel_cfg.get("pca_method", "weighted"),
                    sigma=float(self._pixel_cfg.get("sigma", 1.0)),
                )
                batch["pixel_weight_maps"] = pixel_weight_maps
            except Exception as err:
                logger.warning(
                    "VIPO pixel_weight_maps computation failed: %s."
                    "  Falling back to baseline scalar GRPO for this batch.",
                    err,
                )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def run_wan_sample_step(
        self,
        latents,  # [(16, 7, 64, 64)]
        progress_bar, 
        sigma_schedule,
        transformer,
        context,
        neg_context,
        seq_len,
        grpo_sample,
        flow_window=None,
    ):
        """One Wan sampling step. Latent input layout is (C, T, H, W)."""
        if grpo_sample:
            all_latents = []
            
            all_log_probs = []
            return_prev_sample_mean = bool(
                self.config.actor.get("grpo_guard", {}).get("ratio_norm", False)
            )
            all_prev_sample_mean = [] if return_prev_sample_mean else None
            B = len(context) if isinstance(context, list) else context.shape[0]
            # ensure all tensors are on the same device
            device = latents[0].device
            boundary = getattr(self.config, "wan22_boundary", 0.9)
            base_guide_scale = getattr(self.config, "guide_scale", 5.0)

            if flow_window is None:
                window_start = 0
                window_end = self.config.sampling_steps
            else:
                window_start, window_end = flow_window
            
            for i in progress_bar:
                # Compute timestep from sigma value
                sigma = sigma_schedule[i]
                
                timestep_value = int(sigma * 1000)
                timestep = torch.full([B], timestep_value, device=device, dtype=torch.long)
                sample_guide_scale = _select_wan22_guide_scale(base_guide_scale, timestep, sigma, boundary)
                
                # timestep_cond = timestep
                # timestep_uncond = timestep

                with torch.autocast("cuda", torch.bfloat16):
                    # Wan model input: x is a list of (C, T, H, W) tensors
                    
                    
                    # import hashlib
                    # md5 = hashlib.md5(arr.tobytes()).hexdigest()
                    # print(
                    #     f"[Rollout] rank {torch.distributed.get_rank()} "
                    #     f"step {i}/{self.config.sampling_steps} "
                    #     f"shape={tuple(latents[0].shape)} "
                    #     f"norm={latents[0].norm().item():.4f} "
                    #     f"latents[0] md5={md5}"
                    #     f"timestep_cond norm={timestep_cond} "
                    #     f"context norm={context[0].norm().item():.4f} "
                    #     f"seq_len={seq_len} "
                    # )
                    # with torch.no_grad():
                    #     pred_cond = transformer(
                    #         x=latents,  # [(16, 7, 64, 64)]
                    #         t=timestep,
                    #         context=context,
                    #         seq_len=seq_len
                    #     )
                    with torch.no_grad():
                        pred_cond = transformer(
                            x=latents,  # [(16, 7, 64, 64)]
                            t=timestep,
                            context=context,
                            seq_len=seq_len
                        )
                    # with torch.no_grad():
                    #     pred_cond = transformer(
                    #         x=latents,  # [(16, 7, 64, 64)]
                    #         t=timestep,
                    #         context=context,
                    #         seq_len=seq_len
                    #     )
                        
                    # Unwrap conditional prediction
                    if isinstance(pred_cond, dict) and 'rgb' in pred_cond:
                        model_output_cond = pred_cond['rgb'][0]
                    elif isinstance(pred_cond, list):
                        model_output_cond = pred_cond[0]
                    else:
                        model_output_cond = pred_cond

                    # Unconditional prediction
                    
                    with torch.no_grad():
                        pred_uncond = transformer(
                            x=latents,  # [(16, 7, 64, 64)]
                            t=timestep,
                            context=neg_context,
                            seq_len=seq_len
                        )

                    if isinstance(pred_uncond, dict) and 'rgb' in pred_uncond:
                        model_output_uncond = pred_uncond['rgb'][0]
                    elif isinstance(pred_uncond, list):
                        model_output_uncond = pred_uncond[0]
                    else:
                        model_output_uncond = pred_uncond
                        
                    del pred_cond, pred_uncond

                    # CFG combine
                    model_output = model_output_uncond + sample_guide_scale * (model_output_cond - model_output_uncond)
                    del model_output_cond, model_output_uncond
                    torch.cuda.empty_cache()

                # Wan SDE sampling step
                in_window = window_start <= i < window_end
                if i == window_start:
                    all_latents.append(latents[0])

                if in_window:
                    if return_prev_sample_mean:
                        next_latents, pred_original, log_prob, prev_sample_mean = self.wan_step(
                            model_output,
                            latents[0].to(torch.float32),  # (16, 7, 64, 64)
                            self.config.actor.eta,
                            sigma_schedule,
                            i,
                            prev_sample=None,
                            grpo=True,
                            return_prev_sample_mean=True,
                        )
                        all_prev_sample_mean.append(prev_sample_mean)
                    else:
                        next_latents, pred_original, log_prob = self.wan_step(
                            model_output,
                            latents[0].to(torch.float32),  # (16, 7, 64, 64)
                            self.config.actor.eta,
                            sigma_schedule,
                            i,
                            prev_sample=None,
                            grpo=True,
                        )
                    all_log_probs.append(log_prob)
                    all_latents.append(next_latents.to(torch.float32))
                else:
                    # Outside the SDE window we want a deterministic (ODE
                    # Euler) step.  Setting ``eta=0.0`` zeros both the
                    # score-correction term *and* the Gaussian noise std
                    # in either σ_t form, so the step degenerates cleanly
                    # to ``latents + dsigma · model_output``.
                    next_latents, pred_original = self.wan_step(
                        model_output,
                        latents[0].to(torch.float32),
                        0.0,
                        sigma_schedule,
                        i,
                        prev_sample=None,
                        grpo=False,
                    )
                
                latents=[next_latents.to(torch.float32)]
            final_latents = pred_original

            # all_latents shape is (num_steps+1, 16, 7, 64, 64)
            all_latents = torch.stack(all_latents, dim=0)  # (9, 16, 7, 64, 64)
            all_log_probs = torch.stack(all_log_probs, dim=0)  # (8, B) -> (8,)
            if return_prev_sample_mean:
                all_prev_sample_mean = torch.stack(all_prev_sample_mean, dim=0)
                return latents, final_latents, all_latents, all_log_probs, all_prev_sample_mean
            
            return latents, final_latents, all_latents, all_log_probs

    def wan_step(
        self,
        model_output: torch.Tensor,  # model-predicted flow
        latents: torch.Tensor,       # current-timestep latents (16, 7, 64, 64)
        eta: float,                  # randomness strength
        sigmas: torch.Tensor,        # sigma schedule (FLUX-style)
        index: int,                  # current timestep index
        prev_sample: torch.Tensor,   # previous-step sample (for GRPO re-computation)
        grpo: bool,                  # True -> also return logprob
        return_prev_sample_mean: bool = False,
    ):
        """One Wan Flow-Matching sampling step, recast as an SDE solver for GRPO."""

        sigma = sigmas[index]
        sigma_next = sigmas[index + 1]

        # Predicted original sample (universal flow-matching geometry; used
        # by callers and by the DanceGRPO score-correction inside the
        # registry).
        pred_original_sample = latents - sigma * model_output

        # Dispatch to the σ_t form's SDE step (DanceGRPO constant-η or
        # Flow-GRPO t-dependent).  ``std_dev_t`` is the effective Gaussian
        # std for both noise injection and the log-prob density.  Pure
        # ODE Euler is reached via ``eta=0.0`` (e.g. outside the SDE
        # window above), which zeros both the score correction and the
        # noise std in either form.
        prev_sample_mean, std_dev_t, _sqrt_dt = compute_sde_step(
            form=self._sigma_form,
            model_output=model_output,
            latents=latents,
            eta=eta,
            sigma=sigma,
            sigma_next=sigma_next,
            pred_original_sample=pred_original_sample,
        )

        if grpo and prev_sample is None:
            prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

        if grpo:
            # log probability
            log_prob = (
                -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2)
                / (2 * (std_dev_t**2))
            ) - torch.log(std_dev_t + 1e-8) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))

            # When pixel-weighting is enabled, preserve the spatial dims so
            # the actor can form a dense advantage against the per-pixel map.
            # Here ``log_prob`` has
            # shape (C, T, H, W) for a single rollout sample, so summing
            # the channel axis leaves (T, H, W).  The original baseline
            # behaviour (mean over all non-batch dims -> scalar) is kept
            # when pixel-weighting is off.
            if self._pixel_enable:
                if log_prob.dim() == 4:
                    # (C, T, H, W) -> (T, H, W)
                    log_prob = log_prob.sum(dim=0)
                elif log_prob.dim() == 5:
                    # (B, C, T, H, W) -> (B, T, H, W)
                    log_prob = log_prob.sum(dim=1)
                else:
                    # Fallback: reduce everything except the leading dim.
                    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
            else:
                # Average over every non-batch dim
                log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
            if return_prev_sample_mean:
                return prev_sample, pred_original_sample, log_prob, prev_sample_mean
            return prev_sample, pred_original_sample, log_prob
        else:
            if return_prev_sample_mean:
                return prev_sample_mean, pred_original_sample, prev_sample_mean
            return prev_sample_mean, pred_original_sample

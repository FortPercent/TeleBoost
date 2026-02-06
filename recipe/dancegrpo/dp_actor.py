import logging
import math
import os

import numpy as np
import torch

from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id
from verl.utils.py_functional import append_to_dict
from verl.workers.actor import DataParallelPPOActor

__all__ = ["DiffusionDataParallelPPOActor"]

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


class DiffusionDataParallelPPOActor(DataParallelPPOActor):
    @staticmethod
    def _build_perms(timesteps: torch.Tensor, shuffle: bool = True) -> torch.Tensor:
        seq_len = len(timesteps[0])
        if shuffle:
            return torch.stack([torch.randperm(seq_len) for _ in range(timesteps.shape[0])])
        return torch.stack([torch.arange(seq_len) for _ in range(timesteps.shape[0])])

    @staticmethod
    def _broadcast_perms(perms: torch.Tensor) -> None:
        from verl.utils.ulysses import (
            get_ulysses_sequence_parallel_group,
            get_ulysses_sequence_parallel_world_size,
        )

        if get_ulysses_sequence_parallel_world_size() > 1:
            sp_size = get_ulysses_sequence_parallel_world_size()
            src_rank = (torch.distributed.get_rank() // sp_size) * sp_size
            torch.distributed.broadcast(perms, src=src_rank, group=get_ulysses_sequence_parallel_group())
            torch.distributed.barrier()

    @staticmethod
    def _reorder_batch_by_perms(data: DataProto, perms: torch.Tensor, keys) -> None:
        batch_idx = torch.arange(data.batch.batch_size[0])[:, None]
        for key in keys:
            data.batch[key] = data.batch[key][batch_idx, perms]

    @staticmethod
    def _prepare_contexts(td, device):
        ctx_lens = td["context_orig_lengths"].tolist() if torch.is_tensor(td["context_orig_lengths"]) else td["context_orig_lengths"]
        ctxs_cpu = [td["contexts"][i][:int(ctx_lens[i])] for i in range(len(ctx_lens))]
        nctx_cpu = [td["null_context"][i] for i in range(len(ctx_lens))]
        ctxs = [c.to(device) for c in ctxs_cpu]
        nctxs = [c.to(device) for c in nctx_cpu]
        return ctxs, nctxs

    @staticmethod
    def _calc_seq_len(latents: torch.Tensor) -> int:
        latent_shape = latents.shape
        return math.ceil((latent_shape[3] * latent_shape[4]) / (2 * 2) * latent_shape[2])

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        # GRPO-Guard options (Flow-GRPO RatioNorm)
        guard_cfg = self.config.get("grpo_guard", {})
        guard_enable = guard_cfg.get("enable", False)
        ratio_norm = guard_cfg.get("ratio_norm", guard_enable)
        ratio_norm_eps = guard_cfg.get("ratio_norm_eps", 1e-6)

        flow_cfg = self.config.get("flow_grpo", {})
        shuffle_timesteps = flow_cfg.get("shuffle_timesteps", True)
        if "timestep_indices" in data.batch and shuffle_timesteps:
            shuffle_timesteps = False

        perms = self._build_perms(data.batch["timesteps"], shuffle=shuffle_timesteps)
        self._broadcast_perms(perms)

        permute_keys = ["timesteps", "latents", "next_latents", "log_probs"]
        if ratio_norm:
            permute_keys.append("prev_sample_mean")
        if "timestep_indices" in data.batch:
            permute_keys.append("timestep_indices")
        self._reorder_batch_by_perms(data, perms, permute_keys)

        train_timesteps = int(len(data.batch["timesteps"][0]) * self.config.timestep_fraction)
        grad_norm = None

        select_keys = [
            "timesteps",
            "latents",
            "next_latents",
            "log_probs",
            "contexts",
            "sigma_schedule",
            "advantages",
            "context_orig_lengths",
            "null_context",
        ]
        if ratio_norm:
            select_keys.append("prev_sample_mean")
        if "timestep_indices" in data.batch:
            select_keys.append("timestep_indices")
        non_tensor_select_keys = ["caption"]

        self.gradient_accumulation = (
            self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
        )

        print("gradient_accumulation", self.gradient_accumulation)
        print("=" * 100)

        dataloader = data.select(select_keys, non_tensor_select_keys).chunk(data.batch.batch_size[0])

        device = torch.device(f"cuda:{get_device_id()}")
        move_keys = ["latents", "next_latents", "timesteps", "log_probs", "advantages", "sigma_schedule"]
        if ratio_norm:
            move_keys.append("prev_sample_mean")
        if "timestep_indices" in data.batch:
            move_keys.append("timestep_indices")
        perms = perms.to(device)

        metrics = {}
        for batch_idx, mini_batch in enumerate(dataloader):
            td = mini_batch.batch
            ctxs, nctxs = self._prepare_contexts(td, device)
            batch_on_device = mini_batch.pop(batch_keys=move_keys).to(device)

            self.actor_optimizer.zero_grad()

            for step_idx in range(train_timesteps):
                clip_range = self.config.clip_range
                adv_clip_max = self.config.adv_clip_max

                latent_t = batch_on_device.batch["latents"][:, step_idx]
                nlatent_t = batch_on_device.batch["next_latents"][:, step_idx]
                t_t = batch_on_device.batch["timesteps"][:, step_idx]
                sigma_0 = batch_on_device.batch["sigma_schedule"][0]

                seq_len = self._calc_seq_len(latent_t)

                if "timestep_indices" in batch_on_device.batch:
                    step_indices = batch_on_device.batch["timestep_indices"][0, step_idx]
                else:
                    step_indices = perms[batch_idx][step_idx]

                if ratio_norm:
                    (
                        new_log_probs,
                        prev_sample_mean,
                        std_dev_t,
                        sqrt_dt,
                    ) = self.grpo_wan_one_step(
                        latent_t,
                        nlatent_t,
                        ctxs,
                        nctxs,
                        seq_len,
                        self.actor_module,
                        t_t,
                        step_indices,
                        sigma_0,
                        return_stats=True,
                    )
                else:
                    new_log_probs = self.grpo_wan_one_step(
                        latent_t,
                        nlatent_t,
                        ctxs,
                        nctxs,
                        seq_len,
                        self.actor_module,
                        t_t,
                        step_indices,
                        sigma_0,
                    )

                advantages = torch.clamp(
                    batch_on_device.batch["advantages"],
                    -adv_clip_max,
                    adv_clip_max,
                )

                # 1. 拿到当前这一步的 old_log_prob (形状: Batch)
                old_log_probs_step = batch_on_device.batch["log_probs"][:, step_idx]

                # 2. 拿到当前这一步的 new_log_prob (形状: Batch)
                current_step = -train_timesteps + step_idx
                new_log_probs_step = new_log_probs[..., current_step]

                print(f"Step {step_idx}: New shape {new_log_probs_step.shape}, Old shape {old_log_probs_step.shape}")

                if ratio_norm:
                    prev_sample_mean_step = prev_sample_mean
                    prev_sample_mean_old = batch_on_device.batch["prev_sample_mean"][:, step_idx]
                    print(f"Step {step_idx}: RatioNorm scale {prev_sample_mean_step.shape}, mean bias {prev_sample_mean_old.shape}")
                    
                    ratio_mean_bias = (prev_sample_mean_step - prev_sample_mean_old).pow(2).mean(
                        dim=tuple(range(1, prev_sample_mean_step.ndim))
                    )
                    
                    sqrt_dt = sqrt_dt.mean()
                    std_dev_t_step = std_dev_t[..., current_step]
                    sigma_t = std_dev_t_step / (sqrt_dt + ratio_norm_eps)
                    scale = sqrt_dt * sigma_t
                    ratio_mean_bias = ratio_mean_bias / (2 * (scale**2 + ratio_norm_eps))
                    ratio = torch.exp((new_log_probs_step - old_log_probs_step + ratio_mean_bias) * scale) # 计算重要性采样权重，并进行RatioNorm调整(GRPO_Guard)
                else:
                    ratio = torch.exp(new_log_probs_step - old_log_probs_step)

                clipped_mask = (ratio < (1.0 - clip_range)) | (ratio > (1.0 + clip_range))
                clip_count = clipped_mask.sum().detach().item()
                clip_fraction = clipped_mask.float().mean().detach().item()

                unclipped_loss = -advantages * ratio
                clipped_loss = -advantages * torch.clamp(
                    ratio,
                    1.0 - clip_range,
                    1.0 + clip_range,
                )

                policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                if ratio_norm:
                    policy_loss = policy_loss / (sqrt_dt**2)

                loss = policy_loss / (self.gradient_accumulation * train_timesteps)

                data_dict = {
                    "actor/clip_count": clip_count,
                    "actor/clip_fraction": clip_fraction,
                    "actor/loss": loss.detach().item(),
                }
                if ratio_norm:
                    data_dict["actor/ratio_mean_bias"] = ratio_mean_bias.detach().mean().item()
                    data_dict["actor/ratio_scale"] = scale.detach().item()
                    data_dict["actor/sqrt_dt"] = sqrt_dt.detach().item()
                append_to_dict(metrics, data_dict)

                loss.backward()

                avg_loss = loss.detach()
                torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.AVG)

            if (batch_idx + 1) % self.gradient_accumulation == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor_module.parameters(), self.config.max_grad_norm
                )
                self.actor_optimizer.step()
                self.actor_optimizer.zero_grad()
                data_dict = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data_dict)

            del ctxs, nctxs
            for key in move_keys:
                if key in batch_on_device:
                    del batch_on_device[key]
            del batch_on_device, mini_batch
            torch.cuda.empty_cache()

        return metrics

    def grpo_wan_one_step(
        self,
        latents,
        pre_latents,
        context,
        context_null,
        seq_len,
        transformer,
        timesteps,
        i,
        sigma_schedule,
        guide_scale=5.0,
        return_stats: bool = False,
    ):
        """GRPO的单步训练，支持FP16优化"""
        transformer.train()

        # 确保latents维度正确：(16, 7, 64, 64)
        if latents.dim() == 5:
            latents = latents.squeeze(0)

        if pre_latents.dim() == 5:
            pre_latents = pre_latents.squeeze(0)

        if latents.shape[0] != 16:
            raise ValueError(f"Expected 16 channels, got {latents.shape[0]} channels")

        boundary = getattr(self.config, "wan22_boundary", 0.9)
        sigma = sigma_schedule[i] if sigma_schedule is not None else None
        sample_guide_scale = _select_wan22_guide_scale(guide_scale, timesteps, sigma, boundary)

        autocast_dtype = torch.bfloat16
        with torch.autocast("cuda", dtype=autocast_dtype):
            with torch.no_grad():
                pred_uncond = transformer(
                    x=[latents],
                    t=timesteps,
                    context=context_null,
                    seq_len=seq_len,
                )

            # 处理无条件预测输出
            if isinstance(pred_uncond, dict) and "rgb" in pred_uncond:
                model_output_uncond = pred_uncond["rgb"][0].detach()
            elif isinstance(pred_uncond, list):
                model_output_uncond = pred_uncond[0].detach()
            else:
                model_output_uncond = pred_uncond.detach()

            pred_cond = transformer(
                x=[latents],
                t=timesteps,
                context=context,
                seq_len=seq_len,
            )

            # 处理条件预测输出
            if isinstance(pred_cond, dict) and "rgb" in pred_cond:
                model_output_cond = pred_cond["rgb"][0]
            elif isinstance(pred_cond, list):
                model_output_cond = pred_cond[0]
            else:
                model_output_cond = pred_cond

            # CFG组合
            model_output = model_output_uncond + sample_guide_scale * (model_output_cond - model_output_uncond)

        if return_stats:
            _, _, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = self.wan_step(
                model_output,
                latents.to(torch.float32),
                self.config.eta,
                sigma_schedule,
                i,
                prev_sample=pre_latents,
                grpo=True,
                sde_solver=True,
                return_stats=True,
            )
            return log_prob, prev_sample_mean, std_dev_t, sqrt_dt

        _, _, log_prob = self.wan_step(
            model_output,
            latents.to(torch.float32),
            self.config.eta,
            sigma_schedule,
            i,
            prev_sample=pre_latents,
            grpo=True,
            sde_solver=True,
        )

        return log_prob

    def wan_step(
        self,
        model_output: torch.Tensor,
        latents: torch.Tensor,
        eta: float,
        sigmas: torch.Tensor,
        index: int,
        prev_sample: torch.Tensor,
        grpo: bool,
        sde_solver: bool,
        return_stats: bool = False,
    ):
        """WAN的Flow Matching采样步骤，转换为SDE求解器支持GRPO"""
        sigma = sigmas[index]
        dsigma = sigmas[index + 1] - sigma
        prev_sample_mean = latents + dsigma * model_output
        pred_original_sample = latents - sigma * model_output

        delta_t = sigma - sigmas[index + 1]
        std_dev_t = eta * torch.sqrt(delta_t)

        if sde_solver:
            score_estimate = -(latents - pred_original_sample * (1 - sigma)) / (sigma**2)
            log_term = -0.5 * eta**2 * score_estimate
            prev_sample_mean = prev_sample_mean + log_term * dsigma

        if grpo and prev_sample is None:
            prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

        if grpo:
            log_prob = (
                -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2)
                / (2 * (std_dev_t**2))
            ) - torch.log(std_dev_t + 1e-8) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))

            log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
            if return_stats:
                sqrt_dt = torch.sqrt(delta_t)
                return prev_sample, pred_original_sample, log_prob, prev_sample_mean, std_dev_t, sqrt_dt
            return prev_sample, pred_original_sample, log_prob

        if return_stats:
            sqrt_dt = torch.sqrt(delta_t)
            return prev_sample_mean, pred_original_sample, prev_sample_mean, std_dev_t, sqrt_dt
        return prev_sample_mean, pred_original_sample

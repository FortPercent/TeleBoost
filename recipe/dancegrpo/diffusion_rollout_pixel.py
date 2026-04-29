import math

import torch

from verl.workers.rollout.diffusion_rollout import DiffusionRollout

from .pixel_weight_utils_pixel import compute_batch_pixel_weight_maps_pixel

__all__ = ["DiffusionRolloutPixel"]


class DiffusionRolloutPixel(DiffusionRollout):
    def generate_sequences(self, prompts):
        data = super().generate_sequences(prompts)

        pixel_cfg = self.config.get("pixel_weight", {})
        if not pixel_cfg.get("enable", True):
            return data

        videos = data.batch["video_frames"]
        latents = data.batch["latents"]
        target_time = int(latents.shape[3])
        target_size = (int(latents.shape[4]), int(latents.shape[5]))

        pixel_weight_maps = compute_batch_pixel_weight_maps_pixel(
            videos=videos,
            target_size=target_size,
            target_time=target_time,
            device=videos.device,
            model_path=pixel_cfg.get("model_path", "./dinov2-large"),
            pca_method=pixel_cfg.get("pca_method", "weighted"),
            sigma=float(pixel_cfg.get("sigma", 1.0)),
        )
        data.batch["pixel_weight_maps"] = pixel_weight_maps
        return data

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
        return_prev_sample_mean: bool = False,
    ):
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
            ) - torch.log(std_dev_t + 1e-8) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi, device=latents.device)))
            log_prob = log_prob.sum(dim=0)
            if return_prev_sample_mean:
                return prev_sample, pred_original_sample, log_prob, prev_sample_mean
            return prev_sample, pred_original_sample, log_prob

        if return_prev_sample_mean:
            return prev_sample_mean, pred_original_sample, prev_sample_mean
        return prev_sample_mean, pred_original_sample

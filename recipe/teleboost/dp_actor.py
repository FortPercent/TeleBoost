import functools
import logging
import math
import os

import torch

from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id
from verl.utils.py_functional import append_to_dict
from verl.workers.actor import DataParallelPPOActor

from recipe.teleboost.algorithms.grpo_guard import (
    GRAD_REWEIGHT_FORMS,
    compute_grad_reweight_delta,
    compute_ratio_norm_bias,
)
from recipe.teleboost.algorithms.sigma_schedule import compute_sde_step

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
    # ------------------------------------------------------------------
    # VIPO (pixel-weighted advantage) helpers
    # ------------------------------------------------------------------
    # These small helpers avoid repeating the ``config.get("pixel_weight")``
    # boilerplate in every method and keep the flag name in one place.
    # When the flag is off the actor runs the original scalar GRPO path
    # bit-for-bit identical to the pre-merge baseline.

    def _pixel_cfg(self):
        return self.config.get("pixel_weight", {}) or {}

    def _pixel_enabled(self) -> bool:
        return bool(self._pixel_cfg().get("enable", False))

    @functools.cached_property
    def _sigma_form(self) -> str:
        """SDE σ_t form (cached at first access).

        Mirrors ``DiffusionRollout._sigma_form`` so the rollout and the
        actor's own ``wan_step`` agree on the σ_t convention.  See
        ``algorithms/sigma_schedule.py``.
        """
        return self.config.get("sigma_form", "dancegrpo")

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

        # ----- VIPO: runtime guards & feature detection -------------------
        pixel_enable = self._pixel_enabled()
        pixel_cfg = self._pixel_cfg()

        # GRPO-Guard options.  Paper: arxiv 2510.22319.  RatioNorm (Eq. 8)
        # rewrites the importance-sampling ratio with a Δμ-derived bias term
        # plus an outer ``σ_t · √Δt`` scale; grad-reweight ``δ`` rescales the
        # final policy loss so the gradient magnitude is dt-invariant.  The
        # paper's §4.3 ablation treats RatioNorm and grad-reweight as
        # **separable** levers (Mean-revised / RatioNorm / GRPO-Guard combined),
        # so we expose them as two flags:
        #
        #   ratio_norm       — apply Eq. 8 to the policy ratio
        #   grad_reweight    — rescale the policy loss by δ (= β/dt)
        #
        # ``grad_reweight_form`` selects between the paper's two δ shapes:
        #
        #   flow_grpo  (default): δ = 1/dt           (β ≈ const, Flow-GRPO style)
        #   dancegrpo:            δ = (1 + η²(1−t)/(2t)) / dt   (DanceGRPO form)
        #
        # Both ``ratio_norm`` and ``grad_reweight`` default to ``guard_enable``
        # so legacy ``grpo_guard.enable=true`` configs keep their behaviour
        # (the previous code bundled grad_reweight inside the ``if ratio_norm``
        # branch with a hardcoded ``policy_loss /= sqrt_dt^2`` — i.e. the
        # flow_grpo form).
        guard_cfg = self.config.get("grpo_guard", {})
        guard_enable = guard_cfg.get("enable", False)
        ratio_norm = guard_cfg.get("ratio_norm", guard_enable)
        ratio_norm_eps = guard_cfg.get("ratio_norm_eps", 1e-6)
        grad_reweight = guard_cfg.get("grad_reweight", guard_enable)
        grad_reweight_eps = guard_cfg.get("grad_reweight_eps", 1e-6)
        grad_reweight_form = guard_cfg.get("grad_reweight_form", "flow_grpo")
        if grad_reweight_form not in GRAD_REWEIGHT_FORMS:
            # Fail fast at config read so a typo doesn't silently pass on
            # ``grad_reweight=False`` runs (the helper would only catch it
            # on the first actual δ call).
            raise ValueError(
                f"Unsupported grpo_guard.grad_reweight_form={grad_reweight_form!r}; "
                f"valid forms: {sorted(GRAD_REWEIGHT_FORMS.keys())} "
                f"(see arxiv 2510.22319 §3.2.3)."
            )

        # Dense pixel-weighted advantages require a matching dense KL path.
        use_kl_loss = bool(self.config.get("use_kl_loss", False))
        if pixel_enable and use_kl_loss and not bool(pixel_cfg.get("kl_loss_compatible", False)):
            raise NotImplementedError(
                "VIPO pixel-weighted advantages are incompatible with the KL loss path. "
                "Set `actor_rollout_ref.actor.use_kl_loss=false` or flip "
                "`actor_rollout_ref.pixel_weight.kl_loss_compatible=true` once the "
                "dense-KL path has been implemented and tested."
            )

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

        # Number of denoising steps to train per minibatch. The diffusion
        # rollout already trims the final step (sigma -> 0 yields a peaked
        # log-prob and produces NaN gradients), so the available pool is
        # ``len(timesteps) = sampling_steps - 1``.  Use ``max(1, ...)`` so
        # that small smoke configs (sampling_steps=2,3) still execute at
        # least one policy-gradient step instead of silently no-op'ing.
        timestep_count = len(data.batch["timesteps"][0])
        if timestep_count <= 0:
            raise RuntimeError(
                "No trainable timesteps in batch. The rollout produced "
                "len(timesteps)=0; bump actor_rollout_ref.sampling_steps to "
                ">=2 (the rollout drops the final sigma->0 step)."
            )
        train_timesteps = max(1, int(timestep_count * self.config.timestep_fraction))
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

                # ----- log-prob / advantage shape reconciliation ---------
                # Baseline (pixel_enable=False): both log_probs are scalar
                # per sample -> flatten to (B,).  The matching advantages
                # tensor is also (B,).
                #
                # VIPO (pixel_enable=True): log_probs have dense spatial
                # shape (B, T_lat, H_lat, W_lat).  We keep those dims and
                # broadcast a matching dense advantages tensor - which the
                # trainer already expanded via `_apply_vipo_broadcast` -
                # so the policy loss is computed per spatial location.
                if pixel_enable:
                    old_log_probs_step = batch_on_device.batch["log_probs"][:, step_idx]
                    new_log_probs_step = new_log_probs
                    # Normalise rank so both tensors are (B, T, H, W).
                    # ``wan_step`` returns (T, H, W) for a single-sample
                    # mini-batch; add a leading batch dim so broadcasting
                    # against old_log_probs works.
                    if new_log_probs_step.dim() + 1 == old_log_probs_step.dim():
                        new_log_probs_step = new_log_probs_step.unsqueeze(0)
                    if new_log_probs_step.dim() == old_log_probs_step.dim() + 1 and new_log_probs_step.shape[0] == 1:
                        new_log_probs_step = new_log_probs_step.squeeze(0)
                    if new_log_probs_step.shape != old_log_probs_step.shape:
                        raise RuntimeError(
                            f"VIPO log-prob shape mismatch: new={tuple(new_log_probs_step.shape)} "
                            f"vs old={tuple(old_log_probs_step.shape)}.  Both should be (B, T, H, W)."
                        )
                else:
                    # 1. old_log_prob for this denoising step (shape: Batch)
                    old_log_probs_step = batch_on_device.batch["log_probs"][:, step_idx].flatten()

                    # 2. new_log_probs is already the current step's log_prob (shape: Batch)
                    new_log_probs_step = new_log_probs.flatten()

                print(f"Step {step_idx}: New shape {new_log_probs_step.shape}, Old shape {old_log_probs_step.shape}")

                if ratio_norm:
                    prev_sample_mean_step = prev_sample_mean
                    prev_sample_mean_old = batch_on_device.batch["prev_sample_mean"][:, step_idx]

                    # RatioNorm bias + outer scale (paper arxiv 2510.22319 Eq. 8).
                    # Logic-preserving: ``compute_ratio_norm_bias`` performs
                    # the same reduction order, scalar collapse, and eps
                    # placement as the previous inline implementation.
                    ratio_mean_bias, scale, sqrt_dt_scalar = compute_ratio_norm_bias(
                        prev_sample_mean_step,
                        prev_sample_mean_old,
                        sqrt_dt,
                        std_dev_t,
                        eps=ratio_norm_eps,
                    )

                    print(f"Step {step_idx}: ratio_mean_bias {ratio_mean_bias.shape}, scale {scale}")

                    # VIPO: broadcast the per-sample bias scalar over the
                    # dense spatial dims when the actor is running in
                    # pixel mode.  In scalar mode this is a no-op (shapes
                    # already match).
                    if pixel_enable:
                        ratio_mean_bias_bcast = ratio_mean_bias.view(-1, 1, 1, 1)
                    else:
                        ratio_mean_bias_bcast = ratio_mean_bias

                    # Importance-sampling ratio with RatioNorm adjustment (GRPO_Guard)
                    ratio = torch.exp((new_log_probs_step - old_log_probs_step + ratio_mean_bias_bcast) * scale)
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
                # GRPO-Guard grad-reweight (paper arxiv 2510.22319 §3.2.3,
                # Eq. 12).  ``sqrt_dt_scalar`` is computed inside the
                # ``if ratio_norm`` branch above; recompute it here when
                # grad_reweight is on but ratio_norm is off.
                if grad_reweight:
                    if not ratio_norm:
                        sqrt_dt_scalar = sqrt_dt.mean() if sqrt_dt.ndim > 0 else sqrt_dt
                    dt_scalar = sqrt_dt_scalar ** 2
                    eta = float(self.config.get("eta", 0.25))
                    # Per-sample t_t reduced to a scalar so δ is uniform
                    # across the mini-batch (paper writes δ as a function
                    # of t at the outer step level, not per-sample).
                    t_scalar = t_t.float().mean()
                    delta = compute_grad_reweight_delta(
                        grad_reweight_form,
                        t_scalar,
                        dt_scalar,
                        eta,
                        eps=grad_reweight_eps,
                    )
                    policy_loss = policy_loss * delta

                loss = policy_loss / (self.gradient_accumulation * train_timesteps)

                data_dict = {
                    "actor/clip_count": clip_count,
                    "actor/clip_fraction": clip_fraction,
                    "actor/loss": loss.detach().item(),
                }
                if ratio_norm:
                    data_dict["actor/ratio_mean_bias"] = ratio_mean_bias.detach().mean().item()
                    data_dict["actor/ratio_scale"] = scale if isinstance(scale, float) else scale.item()
                    data_dict["actor/sqrt_dt"] = sqrt_dt_scalar if isinstance(sqrt_dt_scalar, float) else sqrt_dt_scalar.item()
                if grad_reweight:
                    data_dict["actor/grad_reweight_delta"] = delta if isinstance(delta, float) else delta.item()
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
        """One GRPO training step (with FP16 support)."""
        transformer.train()

        # Ensure latents shape (16, 7, 64, 64)
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

            # Handle unconditional prediction output
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

            # Handle conditional prediction output
            if isinstance(pred_cond, dict) and "rgb" in pred_cond:
                model_output_cond = pred_cond["rgb"][0]
            elif isinstance(pred_cond, list):
                model_output_cond = pred_cond[0]
            else:
                model_output_cond = pred_cond

            # CFG combine
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
        return_stats: bool = False,
    ):
        """One Wan Flow-Matching sampling step, recast as an SDE solver for GRPO."""
        sigma = sigmas[index]
        sigma_next = sigmas[index + 1]
        pred_original_sample = latents - sigma * model_output

        # Dispatch to σ_t form's SDE step (DanceGRPO / Flow-GRPO).  The
        # returned ``std_dev_t`` and ``sqrt_dt`` keep the contract the
        # GRPO-Guard RatioNorm path expects:
        #   sigma_t = std_dev_t / sqrt_dt
        # which equals η for DanceGRPO form and η·√(t/(1−t)) for
        # Flow-GRPO form (see ``algorithms/sigma_schedule.py``).
        prev_sample_mean, std_dev_t, sqrt_dt = compute_sde_step(
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
            log_prob = (
                -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2)
                / (2 * (std_dev_t**2))
            ) - torch.log(std_dev_t + 1e-8) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))

            # Align the reduction with the rollout path.
            # Both this actor's ``wan_step`` and the rollout's ``wan_step``
            # must sum the **channel** axis in pixel mode, so the shapes
            # match at loss time.  In baseline mode we mean-reduce over all
            # non-batch dims to a scalar, identical to pre-merge behaviour.
            if self._pixel_enabled():
                if log_prob.dim() == 4:
                    # (C, T, H, W) -> (T, H, W): channel is dim 0 when
                    # there is no explicit batch dim.
                    log_prob = log_prob.sum(dim=0)
                elif log_prob.dim() == 5:
                    # (B, C, T, H, W) -> (B, T, H, W): channel is dim 1.
                    log_prob = log_prob.sum(dim=1)
                else:
                    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
            else:
                log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
            if return_stats:
                return prev_sample, pred_original_sample, log_prob, prev_sample_mean, std_dev_t, sqrt_dt
            return prev_sample, pred_original_sample, log_prob

        if return_stats:
            return prev_sample_mean, pred_original_sample, prev_sample_mean, std_dev_t, sqrt_dt
        return prev_sample_mean, pred_original_sample

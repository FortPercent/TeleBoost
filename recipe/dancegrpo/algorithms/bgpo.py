"""BGPO: Bayesian-Prior Group Optimization.

Two branches share the ``algorithm.bgpo.enable`` flag:

* **CRT** (``use_rerange``): rearranges per-sample rewards before the
  advantage computation, using a prior-based sign and sigmoid.
* **RAS** (``adaptive_weight_*``): scales the scalar advantage by a
  per-group weight derived from posterior statistics.

When ``enable=false`` the mixin is a no-op and the trainer follows the
baseline GRPO path bit-for-bit.

The implementation is exposed as :class:`BGPOMixin` so the trainer can
inherit it and keep ``self.config`` / ``self.global_steps`` /
``self._std_running_accumulation`` shared with the rest of the training
loop. Pure helpers are module-level functions so they can be unit-tested
without spinning up a full trainer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch

from verl import DataProto

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers (no trainer state)
# ---------------------------------------------------------------------------


def compute_joint_task_weights(advantages: torch.Tensor) -> torch.Tensor:
    """Compute per-sample convex weights from multi-reward advantages.

    Used when BGPO is layered on top of a joint reward (multiple reward
    models). Each row of ``advantages`` is one sample's per-task advantage
    vector; the returned weights pick the (lo, hi) endpoints that bracket
    zero so that the convex combination is signed correctly.
    """
    if advantages.numel() == 0:
        return torch.zeros_like(advantages)
    if advantages.dim() != 2:
        raise ValueError(f"advantages must be 2D, got shape {tuple(advantages.shape)}")

    weights = torch.zeros_like(advantages)
    for i in range(advantages.shape[0]):
        a = advantages[i].detach().cpu().numpy().astype(np.float32)
        n = int(a.shape[0])
        if n == 0:
            continue

        if a.min() <= 0 <= a.max():
            idx_lo, idx_hi = int(np.argmin(a)), int(np.argmax(a))
            if np.isclose(a[idx_hi], a[idx_lo]):
                c = np.ones(n, dtype=np.float32) / n
            else:
                t = -a[idx_lo] / (a[idx_hi] - a[idx_lo])
                c = np.zeros(n, dtype=np.float32)
                c[idx_lo] = 1.0 - t
                c[idx_hi] = t
        elif a.min() > 0:
            c = np.zeros(n, dtype=np.float32)
            c[int(np.argmin(a))] = 1.0
        else:
            c = np.zeros(n, dtype=np.float32)
            c[int(np.argmax(a))] = 1.0

        weights[i] = torch.from_numpy(c).to(device=advantages.device, dtype=advantages.dtype)

    return weights


def rerange_group_rewards(
    group_rewards: torch.Tensor,
    prior: float,
    method: str,
    a: float,
    temperature: float,
    exp_clamp: float = 30.0,
) -> torch.Tensor:
    """CRT reward rearrangement (binary prior-threshold method)."""
    if method != "binary":
        return group_rewards

    flag = group_rewards - prior
    positive_sign = torch.clamp(torch.sign(flag), min=0.0)
    numerator = a * flag + positive_sign
    exponent = torch.clamp(-group_rewards / max(temperature, 1e-8), min=-exp_clamp, max=exp_clamp)
    denom = 1.0 + torch.exp(exponent)
    coef = numerator / denom
    return coef * group_rewards


# ---------------------------------------------------------------------------
# Trainer mixin
# ---------------------------------------------------------------------------


class BGPOMixin:
    """Trainer mixin for BGPO. Mix into ``RayDanceGRPOTrainer``.

    Attributes referenced from the trainer:
        - ``self.config`` (Hydra DictConfig)
        - ``self.global_steps`` (int, current training step)
        - ``self._std_running_accumulation`` (float, RAS std running mean)
    """

    # -- config accessors ---------------------------------------------------

    def _get_bgpo_config(self) -> Dict[str, Any]:
        algorithm_cfg = self.config.get("algorithm", {})
        return algorithm_cfg.get("bgpo", {}) or {}

    def _is_bgpo_enabled(self) -> bool:
        return bool(self._get_bgpo_config().get("enable", False))

    def _get_prior_array(self, source_batch: DataProto) -> Optional[np.ndarray]:
        """Return one prior per prompt group (rollout_n samples per group)."""
        if source_batch is None or "prior" not in source_batch.non_tensor_batch:
            return None
        prior_arr = np.asarray(source_batch.non_tensor_batch["prior"]).reshape(-1).astype(np.float32)
        rollout_n = max(int(self.config.actor_rollout_ref.rollout.n), 1)
        if prior_arr.size >= rollout_n and prior_arr.size % rollout_n == 0:
            prior_arr = prior_arr[::rollout_n]
        return prior_arr

    # -- adaptive weight (RAS) ---------------------------------------------

    def _calculate_adaptive_weight(
        self,
        group_rewards: torch.Tensor,
        prior: float,
        step: int,
    ) -> float:
        """RAS weight; supports ``no | random | bayes | cvpr`` modes."""
        bgpo_cfg = self._get_bgpo_config()

        method = bgpo_cfg.get("adaptive_weight_method", "no")
        if method == "no":
            return 0.0
        if method == "random":
            return torch.rand(1).item() * 2 - 1
        if method == "bayes":
            n = int(group_rewards.shape[0])
            if n <= 1:
                sample_var = torch.tensor(1.0, device=group_rewards.device, dtype=group_rewards.dtype)
            else:
                sample_var = group_rewards.var(unbiased=True) + 1e-8

            prior_var = float(bgpo_cfg.get("prior_var", 1.0))
            weight_range = bgpo_cfg.get("bayes_weight_range", [0.5, 1.5])
            if len(weight_range) != 2:
                weight_range = [0.5, 1.5]
            w_min, w_max = float(weight_range[0]), float(weight_range[1])

            sample_mean = group_rewards.mean()
            posterior_var = 1.0 / (n / sample_var + 1.0 / prior_var)
            posterior_mean = posterior_var * (n * sample_mean / sample_var + prior / prior_var)
            posterior_std = torch.sqrt(posterior_var)
            z = (prior - posterior_mean) / posterior_std
            sqrt_2 = torch.sqrt(torch.tensor(2.0, device=group_rewards.device, dtype=group_rewards.dtype))
            prob_better = 1.0 - 0.5 * (1 + torch.erf(z / sqrt_2))
            weight = w_min + (w_max - w_min) * prob_better

            # Match dancegrpo-combine convention: weight is centered on 1.0.
            return float(weight.item() - 1.0)

        if method == "cvpr":
            discriminate_method = bgpo_cfg.get("adaptive_weight_discriminate_method", "normal")
            weight_method = bgpo_cfg.get("adaptive_weight_weight_method", "std_pos")
            fixed_weight = float(bgpo_cfg.get("adaptive_weight_fix_weight", 0.0))

            group_mean = float(group_rewards.mean().item())
            if discriminate_method == "normal":
                sign = 1.0 if group_mean > prior else -1.0
            elif discriminate_method == "reverse":
                sign = 1.0 if group_mean < prior else -1.0
            else:
                raise ValueError(f"Unsupported adaptive_weight_discriminate_method: {discriminate_method}")

            if weight_method == "fix":
                magnitude = fixed_weight
            elif weight_method == "random":
                magnitude = torch.rand(1).item()
            elif "std" in weight_method:
                use_running_mean = bool(bgpo_cfg.get("use_std_runningmean", False))
                std_group = float(group_rewards.std().item() + 1e-8)
                self._std_running_accumulation += std_group
                std = self._std_running_accumulation / max(step, 1) if use_running_mean else std_group
                if weight_method == "std_pos":
                    magnitude = std
                elif weight_method == "std_neg":
                    neg_base = float(bgpo_cfg.get("neg_base", 1.005))
                    scale_neg = float(bgpo_cfg.get("scale_neg", 0.01))
                    magnitude = neg_base ** (scale_neg / max(std, 1e-8)) - 1
                else:
                    raise ValueError(f"Unsupported adaptive_weight_weight_method: {weight_method}")
            else:
                raise ValueError(f"Unsupported adaptive_weight_weight_method: {weight_method}")

            return float(sign * magnitude)

        raise ValueError(f"Unsupported adaptive_weight_method: {method}")

    # -- reward / advantage application ------------------------------------

    def _apply_bgpo_on_rewards(
        self,
        gen_batch_output: DataProto,
        source_batch: DataProto,
        metrics: dict,
    ) -> DataProto:
        """CRT branch: rearrange per-sample rewards using the binary method."""
        if "rewards" not in gen_batch_output.batch:
            return gen_batch_output

        prior_arr = self._get_prior_array(source_batch)
        if prior_arr is None:
            logger.warning("BGPO enabled but 'prior' is missing in dataset; skipping BGPO reward postprocess")
            return gen_batch_output

        bgpo_cfg = self._get_bgpo_config()
        rewards = gen_batch_output.batch["rewards"]
        rollout_n = max(int(self.config.actor_rollout_ref.rollout.n), 1)

        num_groups = int(prior_arr.shape[0])
        max_groups_by_reward = int(rewards.shape[0] // rollout_n)
        num_groups = min(num_groups, max_groups_by_reward)
        if num_groups <= 0:
            return gen_batch_output

        group_rewards_len = num_groups * rollout_n
        rewards_for_groups = rewards[:group_rewards_len]

        use_rerange = bool(bgpo_cfg.get("use_rerange", False))
        rerange_method = bgpo_cfg.get("rerange_method", "binary")
        rerange_a = float(bgpo_cfg.get("rerange_a", 50.0))
        rerange_temperature = float(bgpo_cfg.get("rerange_temperature", 5.0))
        exp_clamp = float(bgpo_cfg.get("exp_clamp", 30.0))
        reranged_rewards = rewards_for_groups.clone()

        if use_rerange:
            for i in range(num_groups):
                start = i * rollout_n
                end = start + rollout_n
                reranged_rewards[start:end] = rerange_group_rewards(
                    rewards_for_groups[start:end],
                    prior=float(prior_arr[i]),
                    method=rerange_method,
                    a=rerange_a,
                    temperature=rerange_temperature,
                    exp_clamp=exp_clamp,
                )
        else:
            return gen_batch_output

        append_rerange = bool(bgpo_cfg.get("append_rerange_samples", False))
        if append_rerange:
            try:
                reranged_proto = gen_batch_output.select(
                    batch_keys=["rewards"],
                    non_tensor_batch_keys=[],
                    deepcopy=True,
                )
                reranged_proto.batch["rewards"][:group_rewards_len] = reranged_rewards
                gen_batch_output = DataProto.concat([gen_batch_output, reranged_proto])
                metrics["train/bgpo_samples"] = float(gen_batch_output.batch.batch_size[0])
            except Exception as exc:
                logger.warning("BGPO append_rerange_samples failed (%s); falling back to in-place rerange", exc)
                gen_batch_output.batch["rewards"][:group_rewards_len] = reranged_rewards
        else:
            gen_batch_output.batch["rewards"][:group_rewards_len] = reranged_rewards

        metrics["train/rewards_reranged"] = reranged_rewards.mean().item()
        return gen_batch_output

    def _apply_bgpo_on_advantages(
        self,
        gen_batch_output: DataProto,
        source_batch: DataProto,
        metrics: dict,
    ) -> DataProto:
        """RAS branch: scale scalar advantages by an adaptive group weight."""
        if "advantages" not in gen_batch_output.batch or "rewards" not in gen_batch_output.batch:
            return gen_batch_output

        prior_arr = self._get_prior_array(source_batch)
        if prior_arr is None:
            return gen_batch_output

        bgpo_cfg = self._get_bgpo_config()
        alpha = float(bgpo_cfg.get("regularization_term_alpha", 1.0))
        max_scale = float(bgpo_cfg.get("max_adv_scale", 10.0))
        min_scale = float(bgpo_cfg.get("min_adv_scale", 0.01))

        rewards = gen_batch_output.batch["rewards"]
        rollout_n = max(int(self.config.actor_rollout_ref.rollout.n), 1)

        num_groups = min(int(prior_arr.shape[0]), int(rewards.shape[0] // rollout_n))
        if num_groups <= 0:
            return gen_batch_output

        group_rewards_len = num_groups * rollout_n
        per_sample_weight = torch.zeros_like(rewards, dtype=torch.float32)

        for i in range(num_groups):
            start = i * rollout_n
            end = start + rollout_n
            weight = self._calculate_adaptive_weight(
                rewards[start:end].float(),
                float(prior_arr[i]),
                self.global_steps,
            )
            per_sample_weight[start:end] = weight

        # Mirror group weights onto an appended rerange block (CRT + RAS combo).
        append_rerange = bool(bgpo_cfg.get("append_rerange_samples", False))
        use_rerange = bool(bgpo_cfg.get("use_rerange", False))
        if use_rerange and append_rerange and rewards.shape[0] >= 2 * group_rewards_len:
            per_sample_weight[group_rewards_len : 2 * group_rewards_len] = per_sample_weight[:group_rewards_len]

        gen_batch_output.batch["bgpo_weight"] = per_sample_weight

        scale = torch.clamp(1.0 + alpha * per_sample_weight, min=min_scale, max=max_scale)
        advantages = gen_batch_output.batch["advantages"]
        # Dense advantages (e.g. VIPO already broadcast) need scale reshaped
        # so we only multiply along the batch axis.
        if advantages.ndim > 1:
            scale_view = scale.view(scale.shape[0], *([1] * (advantages.ndim - 1)))
        else:
            scale_view = scale
        gen_batch_output.batch["advantages"] = advantages * scale_view

        metrics["train/bgpo_weight"] = per_sample_weight[:group_rewards_len].mean().item()
        metrics["train/bgpo_adv_scale"] = scale.mean().item()
        return gen_batch_output


__all__ = [
    "BGPOMixin",
    "compute_joint_task_weights",
    "rerange_group_rewards",
]

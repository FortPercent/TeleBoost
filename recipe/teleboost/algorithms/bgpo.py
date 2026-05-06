"""BGPO: Bayesian-Prior Group Optimization (arxiv 2511.18919).

Paper formulas implemented:

* **CRT** — Contrastive Reward Transformation, Eq. 4::

      R̃ᵢ⁽ʲ⁾ = [λ·(Rᵢ⁽ʲ⁾ − R_prior) + 𝟙{Rᵢ⁽ʲ⁾ > R_prior}] · exp(Rᵢ⁽ʲ⁾)

* **RAS** — Reliability-Adaptive Scaling, Eq. 2::

      wᵢ = 1 + α · [2·σ(k·(R̄ᵢ − R_prior)) − 1]

  Output range is ``[1−α, 1+α]``; the paper recommends ``α=0.5`` (Table 3).

* **Loss coupling**, Eq. 3::

      ℒ_RAS,i = w_group,i · ℒ_GRPO,i

  We apply ``w_group,i`` to the *advantage* (since the GRPO loss is linear
  in the advantage, ``A · w · log_ratio · ...``, this is equivalent to
  applying it to the loss directly).

Two branches share the ``algorithm.bgpo.enable`` flag:

* **CRT** (``use_rerange``): rearranges per-sample rewards before the
  advantage computation.
* **RAS** (``adaptive_weight_method=paper``): scales the scalar advantage
  by a per-group weight ``w_g`` from Eq. 2.

When ``enable=false`` the mixin is a no-op and the trainer follows the
baseline GRPO path bit-for-bit.

The implementation is exposed as :class:`BGPOMixin` so the trainer can
inherit it and keep ``self.config`` / ``self.global_steps`` shared with
the rest of the training loop. Pure helpers are module-level functions
so they can be unit-tested without spinning up a full trainer.

Pre-paper-faithfulness audit (2026-05) the code carried two non-paper
modes: ``rerange_method=binary`` (sigmoid-weighted reward, NOT Eq. 4)
and ``adaptive_weight_method=bayes|cvpr|random`` (closed-form Bayesian
posterior CDF / per-sample heuristic ablations, NOT Eq. 2).  Those have
been removed in favour of the paper-verbatim formulas above; the legacy
config keys (``rerange_method``, ``rerange_temperature``, ``prior_var``,
``bayes_weight_range``, ``adaptive_weight_*`` sub-keys) no longer exist.
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


def rerange_group_rewards(
    group_rewards: torch.Tensor,
    prior: float,
    lambda_contrast: float,
    exp_clamp: float = 30.0,
) -> torch.Tensor:
    """CRT reward rearrangement (BGPO paper Eq. 4, verbatim).

    .. math::

        \\tilde{R} = \\big[\\lambda \\cdot (R - R_{prior})
                     + \\mathbb{1}\\{R > R_{prior}\\}\\big]
                     \\cdot \\exp(R)

    Args:
        group_rewards: raw rewards within a single rollout group, shape
            ``(rollout_n,)``.
        prior: ``R_prior`` for this prompt.
        lambda_contrast: paper's ``λ`` — the contrast factor on the
            ``(R − R_prior)`` term.
        exp_clamp: numerical-safety clamp on the input to ``exp``.  The
            paper does not clamp ``exp(R)`` but in practice rewards can
            be unbounded (e.g. HPS-v2 producing ~30+); we clamp at ±30
            so ``exp`` stays within fp32 range.  Set to a large value
            (e.g. 1e9) to disable the clamp entirely.
    """
    flag = group_rewards - prior
    indicator = (flag > 0).to(group_rewards.dtype)  # paper's 𝟙{R > R_prior}
    bracket = lambda_contrast * flag + indicator
    exponent = torch.clamp(group_rewards, min=-exp_clamp, max=exp_clamp)
    return bracket * torch.exp(exponent)


def paper_ras_centered_weight(
    group_rewards: torch.Tensor,
    prior: float,
    k_sharpness: float,
) -> float:
    """RAS centered weight (paper Eq. 2 inner term).

    Returns ``2·σ(k·(R̄ − R_prior)) − 1``.  The outer ``1 + α·(...)``
    yielding paper Eq. 2's ``w = 1 + α·[2σ(k·(R̄ − R_prior)) − 1]`` is
    applied by the caller (see ``BGPOMixin._apply_bgpo_on_advantages``).

    Output range is ``(−1, 1)``; multiplied by ``α`` and added to 1 it
    becomes the paper's ``w ∈ (1−α, 1+α)``.

    Args:
        group_rewards: rewards in a single rollout group, shape
            ``(rollout_n,)`` or ``(B,)``.
        prior: ``R_prior`` for the prompt.
        k_sharpness: paper's ``k`` — sigmoid sharpness on the mean
            deviation ``(R̄ − R_prior)``.
    """
    group_mean = group_rewards.mean()
    x = k_sharpness * (group_mean - prior)
    return float(2.0 * torch.sigmoid(x).item() - 1.0)


# ---------------------------------------------------------------------------
# Trainer mixin
# ---------------------------------------------------------------------------


class BGPOMixin:
    """Trainer mixin for BGPO. Mix into ``RayDanceGRPOTrainer``.

    Attributes referenced from the trainer:
        - ``self.config`` (Hydra DictConfig)
        - ``self.global_steps`` (int, current training step)
    """

    # -- config accessors ---------------------------------------------------

    def _get_bgpo_config(self) -> Dict[str, Any]:
        algorithm_cfg = self.config.get("algorithm", {})
        return algorithm_cfg.get("bgpo", {}) or {}

    def _is_bgpo_enabled(self) -> bool:
        enabled = bool(self._get_bgpo_config().get("enable", False))
        if enabled:
            # Fail-loud guard: BGPO paper (arxiv 2511.18919) specifies
            # single-scalar-reward optimization only.  The joint reward path in
            # this codebase uses an in-house multi-reward aggregation
            # (``compute_joint_task_weights`` in
            # ``algorithms/multi_reward_aggregation.py``) that is **not** part
            # of the BGPO paper and has not been independently validated.
            # Stacking BGPO on top of the joint path therefore mixes a
            # paper-faithful BGPO with an in-house multi-reward scheme, which
            # is misleading at best and possibly incorrect.  Refuse the
            # combination explicitly so that nobody silently trains a
            # "BGPO + joint" run thinking it is BGPO-paper-validated.
            reward_type = self.config.get("reward_model", {}).get("type", "single")
            if reward_type == "joint":
                raise ValueError(
                    "Configuration not supported: algorithm.bgpo.enable=true "
                    "with reward_model.type=joint. The BGPO paper "
                    "(arxiv 2511.18919) specifies single-scalar-reward "
                    "optimization only; combining BGPO with the joint reward "
                    "path runs an in-house multi-reward aggregation "
                    "(compute_joint_task_weights) that is not paper-validated. "
                    "Either set algorithm.bgpo.enable=false (use joint without "
                    "BGPO scaling/CRT) or set reward_model.type=single (use "
                    "BGPO with a single scalar reward, the paper-supported "
                    "configuration)."
                )

            # Soft warning: BGPO + VIPO is structurally compatible (BGPO's
            # per-group scalar weight broadcasts cleanly across VIPO's dense
            # ``[B, T, H, W]`` advantage), but neither paper studies the
            # combination.  BGPO (arxiv 2511.18919) operates on scalar
            # advantages; VIPO (arxiv 2511.18719) reshapes advantages to
            # per-pixel.  Stacking them turns BGPO's per-group weight into a
            # uniform multiplier across VIPO's pixel pattern — mathematically
            # benign, research-wise unvalidated.
            pixel_enable = bool(
                self.config.get("actor_rollout_ref", {})
                .get("pixel_weight", {})
                .get("enable", False)
            )
            if pixel_enable:
                logger.warning(
                    "BGPO + VIPO are both enabled.  This combination is "
                    "structurally compatible (briefly tested) but not validated "
                    "by either paper: BGPO (arxiv 2511.18919) and VIPO "
                    "(arxiv 2511.18719) were developed independently.  BGPO's "
                    "per-group scalar weight becomes a uniform multiplier "
                    "across VIPO's pixel pattern, which preserves VIPO's "
                    "shaping but is not a paper-supported configuration."
                )
        return enabled

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
        step: int,  # kept for signature compat; unused in paper mode
    ) -> float:
        """RAS per-group weight (BGPO paper Eq. 2).

        Returns the *centered* weight ``2σ(k·(R̄ − R_prior)) − 1``;
        ``_apply_bgpo_on_advantages`` then multiplies by ``α`` and adds
        ``1`` to get the paper's ``w = 1 + α·[2σ(k·(R̄ − R_prior)) − 1]``.

        Modes:
          * ``"paper"`` (default): paper Eq. 2 verbatim.
          * ``"no"``: returns ``0`` so the outer ``1 + α·0 = 1`` leaves
            the advantage unchanged.  Useful when running CRT alone.
        """
        del step  # signature compat; paper Eq. 2 has no step dependence.
        bgpo_cfg = self._get_bgpo_config()
        method = bgpo_cfg.get("adaptive_weight_method", "paper")

        if method == "no":
            return 0.0

        if method == "paper":
            return paper_ras_centered_weight(
                group_rewards,
                prior=prior,
                k_sharpness=float(bgpo_cfg.get("k_sharpness", 1.0)),
            )

        raise ValueError(
            f"Unsupported adaptive_weight_method={method!r}; "
            f"valid modes: 'paper' (Eq. 2), 'no'.  Pre-2026-05 the code "
            f"carried 'bayes', 'cvpr', and 'random' modes that were not "
            f"paper-faithful and have been removed; switch to 'paper'."
        )

    # -- reward / advantage application ------------------------------------

    def _apply_bgpo_on_rewards(
        self,
        gen_batch_output: DataProto,
        source_batch: DataProto,
        metrics: dict,
    ) -> DataProto:
        """CRT branch: rearrange per-sample rewards via paper Eq. 4."""
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
        lambda_contrast = float(bgpo_cfg.get("lambda_contrast", 1.0))
        exp_clamp = float(bgpo_cfg.get("exp_clamp", 30.0))
        reranged_rewards = rewards_for_groups.clone()

        if use_rerange:
            for i in range(num_groups):
                start = i * rollout_n
                end = start + rollout_n
                reranged_rewards[start:end] = rerange_group_rewards(
                    rewards_for_groups[start:end],
                    prior=float(prior_arr[i]),
                    lambda_contrast=lambda_contrast,
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
    "rerange_group_rewards",
    "paper_ras_centered_weight",
]

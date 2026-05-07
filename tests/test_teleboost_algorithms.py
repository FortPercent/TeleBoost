"""Unit tests for the pure functions extracted into ``recipe/teleboost/algorithms/``.

These cover the refactor's contract surface: ``compute_joint_task_weights``
and ``rerange_group_rewards``. They run CPU-only and need ``torch`` but
not a GPU.

Run from the repo root:

    pytest tests/test_teleboost_algorithms.py -v
"""
from __future__ import annotations

import math

import pytest
import torch

from recipe.teleboost.algorithms import (
    compute_joint_task_weights,
    paper_ras_centered_weight,
    rerange_group_rewards,
)


# ---------------------------------------------------------------------------
# compute_joint_task_weights
# ---------------------------------------------------------------------------


class TestComputeJointTaskWeights:
    def test_empty_input_returns_zeros(self):
        out = compute_joint_task_weights(torch.empty(0, 3))
        assert out.shape == (0, 3)
        assert out.numel() == 0

    def test_single_sample_straddling_zero(self):
        # One sample, advantages span [-2, 1, 3]. min=-2 (idx 0), max=3 (idx 2).
        # t = -(-2) / (3 - (-2)) = 0.4. So c[0] = 0.6, c[2] = 0.4, c[1] = 0.
        adv = torch.tensor([[-2.0, 1.0, 3.0]])
        weights = compute_joint_task_weights(adv)
        assert weights.shape == (1, 3)
        assert torch.allclose(weights[0], torch.tensor([0.6, 0.0, 0.4]), atol=1e-6)

    def test_all_positive_picks_argmin(self):
        # All advantages > 0 -> single-mass at the smallest one.
        adv = torch.tensor([[1.0, 2.0, 3.0]])
        weights = compute_joint_task_weights(adv)
        assert torch.allclose(weights[0], torch.tensor([1.0, 0.0, 0.0]))

    def test_all_negative_picks_argmax(self):
        adv = torch.tensor([[-3.0, -1.0, -2.0]])
        weights = compute_joint_task_weights(adv)
        # argmax = idx 1 (value -1.0)
        assert torch.allclose(weights[0], torch.tensor([0.0, 1.0, 0.0]))

    def test_uniform_when_max_equals_min(self):
        # All-zero advantages: min == max == 0, near-zero -> uniform 1/n.
        adv = torch.zeros(1, 4)
        weights = compute_joint_task_weights(adv)
        assert torch.allclose(weights[0], torch.full((4,), 0.25))

    def test_per_row_independence(self):
        # Mix three independent rows; weights should be computed per-row.
        adv = torch.tensor([
            [-1.0, 1.0],     # straddle
            [2.0, 5.0],      # all positive
            [-4.0, -1.0],    # all negative
        ])
        weights = compute_joint_task_weights(adv)
        # Row 0: t = 1/2 -> [0.5, 0.5]
        assert torch.allclose(weights[0], torch.tensor([0.5, 0.5]))
        # Row 1: argmin = 0 -> [1, 0]
        assert torch.allclose(weights[1], torch.tensor([1.0, 0.0]))
        # Row 2: argmax = 1 -> [0, 1]
        assert torch.allclose(weights[2], torch.tensor([0.0, 1.0]))

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2D"):
            compute_joint_task_weights(torch.tensor([1.0, 2.0]))


# ---------------------------------------------------------------------------
# rerange_group_rewards (BGPO CRT branch — paper Eq. 4)
# ---------------------------------------------------------------------------
#
# Paper Eq. 4 (arxiv 2511.18919):
#   R̃ = [λ·(R − R_prior) + 𝟙{R > R_prior}] · exp(R)
#
# These tests pin the formula verbatim.


class TestRerangeGroupRewardsPaperEq4:
    def test_at_prior_indicator_is_zero(self):
        # R == R_prior: flag = 0, indicator = 0 → bracket = 0, output = 0.
        rewards = torch.tensor([0.5])
        out = rerange_group_rewards(rewards, prior=0.5, lambda_contrast=1.0)
        assert torch.allclose(out, torch.zeros_like(rewards))

    def test_above_prior_paper_formula(self):
        # R = 1.0, R_prior = 0.5, λ = 50 → bracket = 50·0.5 + 1 = 26
        # output = 26 · exp(1.0) ≈ 70.685
        rewards = torch.tensor([1.0])
        out = rerange_group_rewards(rewards, prior=0.5, lambda_contrast=50.0)
        expected = (50 * 0.5 + 1.0) * math.exp(1.0)
        assert torch.allclose(out, torch.tensor([expected]), rtol=1e-5)

    def test_below_prior_paper_formula(self):
        # R = 0.1, R_prior = 0.5, λ = 50 → bracket = 50·(-0.4) + 0 = -20
        # output = -20 · exp(0.1) ≈ -22.103
        rewards = torch.tensor([0.1])
        out = rerange_group_rewards(rewards, prior=0.5, lambda_contrast=50.0)
        expected = (50 * -0.4 + 0.0) * math.exp(0.1)
        assert torch.allclose(out, torch.tensor([expected]), rtol=1e-5)

    def test_clamps_extreme_reward(self):
        # R = -1e6 with default exp_clamp=30 → exp(R) clamped to exp(-30) > 0.
        rewards = torch.tensor([-1e6])
        out = rerange_group_rewards(rewards, prior=0.0, lambda_contrast=1.0)
        assert torch.isfinite(out).all()
        # bracket = 1·(-1e6) + 0 = -1e6.  exp(R) clamped to exp(-30).
        assert out.item() == pytest.approx(-1e6 * math.exp(-30.0), rel=1e-5)

    def test_preserves_dtype_and_device(self):
        rewards = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)
        out = rerange_group_rewards(rewards, prior=0.5, lambda_contrast=10.0)
        assert out.dtype == rewards.dtype
        assert out.device == rewards.device

    def test_batched_paper_formula(self):
        # Element-wise paper Eq. 4 on a 3-element batch.
        rewards = torch.tensor([0.2, 0.5, 0.8])
        out = rerange_group_rewards(rewards, prior=0.5, lambda_contrast=10.0)
        # idx 0: R=0.2, flag=-0.3, ind=0, bracket=-3, output=-3*exp(0.2)
        # idx 1: R=0.5, flag=0, ind=0, bracket=0, output=0
        # idx 2: R=0.8, flag=0.3, ind=1, bracket=4, output=4*exp(0.8)
        expected = torch.tensor([
            -3.0 * math.exp(0.2),
            0.0,
            4.0 * math.exp(0.8),
        ])
        assert torch.allclose(out, expected, rtol=1e-5)
        # Sign sanity:
        assert out[0].item() < 0
        assert out[1].item() == 0.0
        assert out[2].item() > 0

    def test_indicator_is_strict_inequality(self):
        # Paper specifies 𝟙{R > R_prior}, strict.  At R == R_prior the
        # indicator is zero, not one.  Pin to catch any future drift to
        # >= or rounded sign.
        rewards = torch.tensor([0.5, 0.5 + 1e-7])
        out = rerange_group_rewards(rewards, prior=0.5, lambda_contrast=0.0)
        # λ=0 isolates the indicator term: bracket = 0 or 1 only.
        # R = 0.5 → indicator = 0 → output = 0
        # R = 0.5 + 1e-7 → indicator = 1 → output = exp(0.5 + 1e-7) > 0
        assert out[0].item() == 0.0
        assert out[1].item() > 0.0


# ---------------------------------------------------------------------------
# paper_ras_centered_weight (BGPO RAS branch — paper Eq. 2 inner term)
# ---------------------------------------------------------------------------
#
# Paper Eq. 2 (arxiv 2511.18919):
#   w = 1 + α · [2·σ(k·(R̄ − R_prior)) − 1]
#
# ``paper_ras_centered_weight`` returns the term in brackets,
# ``2·σ(k·(R̄ − R_prior)) − 1``, so the BGPO trainer's outer
# ``advantage *= clamp(1 + α·w_centered, ...)`` reconstructs paper Eq. 2.


class TestPaperRASCenteredWeight:
    def test_at_prior_is_zero(self):
        # R̄ == R_prior → sigmoid(0) = 0.5 → 2·0.5 − 1 = 0 (centered).
        w = paper_ras_centered_weight(torch.tensor([0.5, 0.5, 0.5]), prior=0.5, k_sharpness=1.0)
        assert w == pytest.approx(0.0, abs=1e-7)

    def test_above_prior_is_positive(self):
        # R̄ > R_prior → sigmoid > 0.5 → centered weight > 0.
        w = paper_ras_centered_weight(torch.tensor([0.8, 0.9, 1.0]), prior=0.5, k_sharpness=1.0)
        assert w > 0.0

    def test_below_prior_is_negative(self):
        w = paper_ras_centered_weight(torch.tensor([0.0, 0.1, 0.2]), prior=0.5, k_sharpness=1.0)
        assert w < 0.0

    def test_paper_formula_explicit(self):
        # Pin formula numerically.
        # R̄ = 0.7, R_prior = 0.3, k = 2.0
        # x = 2·(0.7 − 0.3) = 0.8
        # σ(0.8) = 1/(1+exp(-0.8)) = 0.6900...
        # centered = 2·0.6900 − 1 = 0.3800...
        w = paper_ras_centered_weight(torch.tensor([0.6, 0.7, 0.8]), prior=0.3, k_sharpness=2.0)
        sigmoid_val = 1.0 / (1.0 + math.exp(-0.8))
        expected = 2.0 * sigmoid_val - 1.0
        assert w == pytest.approx(expected, rel=1e-6)

    def test_output_range_is_minus_one_to_one(self):
        # Sigmoid is mathematically in (0,1), so 2σ−1 is in (−1,1) — but
        # float32 saturates at the boundaries for extreme inputs (e.g.
        # sigmoid(-100) underflows to 0.0).  In practice the range is
        # the closed interval [-1, 1].
        for k in [0.1, 1.0, 10.0]:
            for mean_val in [-1e3, -1.0, 0.0, 1.0, 1e3]:
                w = paper_ras_centered_weight(
                    torch.tensor([mean_val]), prior=0.0, k_sharpness=k,
                )
                assert -1.0 <= w <= 1.0, f"k={k}, mean={mean_val}, w={w}"

    def test_k_zero_yields_zero(self):
        # k=0 → x=0 → σ(0)=0.5 → centered = 0 (regardless of mean).
        w = paper_ras_centered_weight(torch.tensor([100.0]), prior=0.0, k_sharpness=0.0)
        assert w == pytest.approx(0.0, abs=1e-7)

    def test_paper_w_reconstruction_in_alpha_range(self):
        # Verify the *outer* ``w = 1 + α · centered`` falls in [1−α, 1+α]
        # for α=0.5 (paper Table 3 recommended) — the trainer's
        # ``advantage *= clamp(1 + α·w_centered, min, max)`` then matches
        # paper Eq. 2 exactly.
        alpha = 0.5
        for rewards, prior in [
            (torch.tensor([0.9, 1.0, 1.1]), 0.0),  # well above prior
            (torch.tensor([-0.9, -1.0, -1.1]), 0.0),  # well below prior
            (torch.tensor([0.0, 0.0, 0.0]), 0.0),  # at prior
        ]:
            w_centered = paper_ras_centered_weight(rewards, prior=prior, k_sharpness=5.0)
            w_paper = 1.0 + alpha * w_centered  # = paper Eq. 2
            assert 1.0 - alpha <= w_paper <= 1.0 + alpha


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_algorithms_namespace_exports_mixins():
    """The trainer relies on these names from algorithms.__init__."""
    from recipe.teleboost.algorithms import BGPOMixin, JointRewardMixin, VIPOMixin

    bgpo_methods = {m for m in dir(BGPOMixin) if not m.startswith("__")}
    vipo_methods = {m for m in dir(VIPOMixin) if not m.startswith("__")}
    joint_methods = {m for m in dir(JointRewardMixin) if not m.startswith("__")}

    assert {"_get_bgpo_config", "_is_bgpo_enabled", "_apply_bgpo_on_rewards",
            "_apply_bgpo_on_advantages", "_calculate_adaptive_weight",
            "_get_prior_array"} <= bgpo_methods
    assert {"_is_pixel_weight_enabled", "_apply_vipo_broadcast"} <= vipo_methods
    assert {"_maybe_create_joint_reward_runner", "_compute_joint_reward",
            "_compute_joint_parallel_reward", "_precompute_joint_advantages"} <= joint_methods


def test_trainer_inherits_all_three_mixins():
    """The driver trainer must mix in BGPO, VIPO, and JointReward."""
    from recipe.teleboost.algorithms import BGPOMixin, JointRewardMixin, VIPOMixin
    from recipe.teleboost.teleboost_ray_trainer import RayTeleBoostTrainer

    mro = RayTeleBoostTrainer.__mro__
    assert BGPOMixin in mro
    assert VIPOMixin in mro
    assert JointRewardMixin in mro


def test_joint_runner_helpers_callable():
    """Module-level helpers exposed for unit tests / external use."""
    from recipe.teleboost.algorithms import _JointRewardRunner, merge_worker_results

    assert callable(merge_worker_results)
    assert hasattr(_JointRewardRunner, "compute")


def test_pixel_weight_utils_shim_reexports():
    """Diffusion rollout still imports compute_batch_pixel_weight_maps from
    the legacy path."""
    from recipe.teleboost.pixel_weight_utils import compute_batch_pixel_weight_maps as legacy
    from recipe.teleboost.algorithms.vipo import compute_batch_pixel_weight_maps as canonical

    assert legacy is canonical

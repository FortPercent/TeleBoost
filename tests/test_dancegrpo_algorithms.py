"""Unit tests for the pure functions extracted into ``recipe/dancegrpo/algorithms/``.

These cover the refactor's contract surface: ``compute_joint_task_weights``
and ``rerange_group_rewards``. They run CPU-only and need ``torch`` but
not a GPU.

Run from the repo root:

    pytest tests/test_dancegrpo_algorithms.py -v
"""
from __future__ import annotations

import math

import pytest
import torch

from recipe.dancegrpo.algorithms import (
    compute_joint_task_weights,
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
# rerange_group_rewards (BGPO CRT branch)
# ---------------------------------------------------------------------------


class TestRerangeGroupRewards:
    def test_unknown_method_is_passthrough(self):
        rewards = torch.tensor([0.1, 0.5, 0.9])
        out = rerange_group_rewards(rewards, prior=0.5, method="unknown", a=1.0, temperature=1.0)
        assert torch.equal(out, rewards)

    def test_binary_zero_for_reward_at_prior(self):
        # When reward == prior, flag = 0 and positive_sign = 0 -> coef = 0.
        rewards = torch.tensor([0.5])
        out = rerange_group_rewards(rewards, prior=0.5, method="binary", a=1.0, temperature=1.0)
        assert torch.allclose(out, torch.zeros_like(rewards))

    def test_binary_above_prior_is_amplified(self):
        # reward = 1.0, prior = 0.5, a = 50.0 -> numerator = 50*0.5 + 1 = 26.
        # denom = 1 + exp(-1.0/5.0) ~= 1 + 0.8187 ~= 1.8187.
        # coef = 26 / 1.8187 ~= 14.30.
        # output = coef * 1.0 ~= 14.30.
        rewards = torch.tensor([1.0])
        out = rerange_group_rewards(
            rewards, prior=0.5, method="binary", a=50.0, temperature=5.0
        )
        denom = 1.0 + math.exp(-1.0 / 5.0)
        expected = (50 * 0.5 + 1.0) / denom
        assert torch.allclose(out, torch.tensor([expected]), atol=1e-5)

    def test_binary_below_prior_has_negative_coef(self):
        # flag = -0.4, positive_sign = 0, numerator = -20, output is negative.
        rewards = torch.tensor([0.1])
        out = rerange_group_rewards(
            rewards, prior=0.5, method="binary", a=50.0, temperature=5.0
        )
        assert out.item() < 0.0

    def test_binary_clamps_extreme_exponent(self):
        # reward = -1e6 with temperature = 1: -reward/T -> 1e6 -> overflow.
        # The function clamps to ``exp_clamp`` so exp() stays finite.
        rewards = torch.tensor([-1e6])
        out = rerange_group_rewards(
            rewards,
            prior=0.0,
            method="binary",
            a=1.0,
            temperature=1.0,
            exp_clamp=30.0,
        )
        assert torch.isfinite(out).all()

    def test_binary_preserves_dtype_and_device(self):
        rewards = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)
        out = rerange_group_rewards(
            rewards, prior=0.5, method="binary", a=10.0, temperature=2.0
        )
        assert out.dtype == rewards.dtype
        assert out.device == rewards.device

    def test_binary_batched_input(self):
        # The function operates element-wise; verify a 1-D batch.
        rewards = torch.tensor([0.2, 0.5, 0.8])
        out = rerange_group_rewards(
            rewards, prior=0.5, method="binary", a=10.0, temperature=2.0
        )
        assert out.shape == rewards.shape
        # Reward at prior (idx 1) is exactly zero.
        assert out[1].item() == pytest.approx(0.0, abs=1e-7)
        # Above prior is positive, below is negative.
        assert out[2].item() > 0
        assert out[0].item() < 0


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_algorithms_namespace_exports_mixins():
    """The trainer relies on these names from algorithms.__init__."""
    from recipe.dancegrpo.algorithms import BGPOMixin, VIPOMixin

    # Both should be plain mixin classes (no abstract methods, no required init).
    bgpo_methods = {m for m in dir(BGPOMixin) if not m.startswith("__")}
    vipo_methods = {m for m in dir(VIPOMixin) if not m.startswith("__")}

    assert {"_get_bgpo_config", "_is_bgpo_enabled", "_apply_bgpo_on_rewards",
            "_apply_bgpo_on_advantages", "_calculate_adaptive_weight",
            "_get_prior_array"} <= bgpo_methods
    assert {"_is_pixel_weight_enabled", "_apply_vipo_broadcast"} <= vipo_methods


def test_pixel_weight_utils_shim_reexports():
    """Diffusion rollout still imports compute_batch_pixel_weight_maps from
    the legacy path."""
    from recipe.dancegrpo.pixel_weight_utils import compute_batch_pixel_weight_maps as legacy
    from recipe.dancegrpo.algorithms.vipo import compute_batch_pixel_weight_maps as canonical

    assert legacy is canonical

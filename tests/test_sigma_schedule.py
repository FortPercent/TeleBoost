"""Paper-equation level tests for ``algorithms/sigma_schedule.py``.

References:
    * DanceGRPO (arxiv 2505.07818) — constant-η σ_t form
    * Flow-GRPO (arxiv 2505.05470) — t-dependent σ_t = η·√(t/(1−t))
    * Flow-GRPO upstream ``sd3_sde_with_logprob.py`` for the verbatim
      mean-update coefficients.

These tests pin two things:

1. **Byte-equivalence** with the pre-registry inline ``wan_step`` math
   for ``sigma_form="dancegrpo"`` (the default).  This guards against
   silent drift in any future refactor — default smokes will stay
   identical to the pre-merge baseline.

2. **Paper-faithfulness** for ``sigma_form="flow_grpo"`` against the
   upstream formulas (std and mean-update coefficients).

Run with::

    pytest tests/test_sigma_schedule.py -v
"""

from __future__ import annotations

import math

import pytest
import torch

from recipe.teleboost.algorithms.sigma_schedule import (
    SIGMA_FORMS,
    compute_sde_step,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_inputs(seed: int = 0):
    """Reproducible input bundle for SDE step tests."""
    g = torch.Generator().manual_seed(seed)
    latents = torch.randn(4, 3, 16, 16, generator=g, dtype=torch.float64)
    model_output = torch.randn(4, 3, 16, 16, generator=g, dtype=torch.float64)
    sigma = torch.tensor(0.7, dtype=torch.float64)
    sigma_next = torch.tensor(0.45, dtype=torch.float64)
    eta = 0.3
    pred_original = latents - sigma * model_output
    return latents, model_output, sigma, sigma_next, eta, pred_original


# ---------------------------------------------------------------------------
# DanceGRPO form: byte-equivalence to legacy inline math
# ---------------------------------------------------------------------------


def _legacy_dancegrpo_step(model_output, latents, eta, sigma, sigma_next):
    """Verbatim copy of the pre-registry inline ``wan_step`` math (with the
    score-correction term, which was always reached: every production
    call site passed ``sde_solver=True``).

    Used as the reference for byte-equivalence: the registry must agree
    with this exactly when ``form="dancegrpo"``.
    """
    dsigma = sigma_next - sigma
    prev_sample_mean = latents + dsigma * model_output
    pred_original_sample = latents - sigma * model_output
    delta_t = sigma - sigma_next
    std_dev_t = eta * torch.sqrt(delta_t)
    score_estimate = -(latents - pred_original_sample * (1 - sigma)) / (sigma ** 2)
    log_term = -0.5 * (eta ** 2) * score_estimate
    prev_sample_mean = prev_sample_mean + log_term * dsigma
    return prev_sample_mean, std_dev_t, torch.sqrt(delta_t)


def test_dancegrpo_form_byte_equivalent_to_legacy():
    """Registry's dancegrpo form must match the pre-refactor inline math
    bit-for-bit (in float64) so existing rollouts stay identical."""
    latents, model_output, sigma, sigma_next, eta, pred_original = _make_inputs()

    new_mean, new_std, new_sqrt_dt = compute_sde_step(
        form="dancegrpo",
        model_output=model_output,
        latents=latents,
        eta=eta,
        sigma=sigma,
        sigma_next=sigma_next,
        pred_original_sample=pred_original,
    )
    ref_mean, ref_std, ref_sqrt_dt = _legacy_dancegrpo_step(
        model_output, latents, eta, sigma, sigma_next
    )

    assert torch.equal(new_mean, ref_mean), "prev_sample_mean drifted"
    assert torch.equal(new_std, ref_std), "std_dev_t drifted"
    assert torch.equal(new_sqrt_dt, ref_sqrt_dt), "sqrt_dt drifted"


def test_dancegrpo_std_equals_eta_times_sqrt_dt():
    """DanceGRPO σ_t = η constant ⇒ std_dev_t / sqrt_dt = η."""
    latents, model_output, sigma, sigma_next, eta, pred_original = _make_inputs()
    _, std_dev_t, sqrt_dt = compute_sde_step(
        "dancegrpo", model_output, latents, eta, sigma, sigma_next, pred_original
    )
    assert std_dev_t.item() == pytest.approx(eta * sqrt_dt.item(), rel=1e-12)


def test_eta_zero_degenerates_to_pure_ode_in_both_forms():
    """``eta=0`` must zero both the score correction *and* the noise std,
    leaving the pure ODE Euler mean.  This is the actual mechanism the
    rollout uses to disable noise outside the SDE window — both forms
    must support it cleanly (otherwise the rollout's outside-window
    branch would produce different trajectories per form)."""
    latents, model_output, sigma, sigma_next, _eta, pred_original = _make_inputs()
    expected_ode = latents + (sigma_next - sigma) * model_output

    for form in ("dancegrpo", "flow_grpo"):
        mean, std, _ = compute_sde_step(
            form, model_output, latents, eta=0.0,
            sigma=sigma, sigma_next=sigma_next,
            pred_original_sample=pred_original,
        )
        assert torch.allclose(mean, expected_ode, rtol=1e-12, atol=1e-14), (
            f"form={form!r} did not degenerate to ODE Euler at eta=0"
        )
        assert torch.equal(std, torch.zeros_like(std)), (
            f"form={form!r} std must be zero at eta=0"
        )


# ---------------------------------------------------------------------------
# Flow-GRPO form: paper-faithfulness
# ---------------------------------------------------------------------------


def test_flow_grpo_std_matches_paper_formula():
    """Flow-GRPO: σ_t = η·√(t/(1−t)), and the effective Gaussian std is
    σ_t·√Δt.  Pinned to the paper formula explicitly.
    """
    latents, model_output, sigma, sigma_next, eta, pred_original = _make_inputs()
    _, std_dev_t, sqrt_dt = compute_sde_step(
        "flow_grpo", model_output, latents, eta, sigma, sigma_next, pred_original
    )

    expected_sigma_t = eta * math.sqrt(sigma.item() / (1.0 - sigma.item()))
    expected_std = expected_sigma_t * math.sqrt(sigma.item() - sigma_next.item())
    assert std_dev_t.item() == pytest.approx(expected_std, rel=1e-12)


def test_flow_grpo_sigma_t_recoverable_via_grpo_guard_contract():
    """GRPO-Guard reads ``sigma_t = std_dev_t / sqrt_dt``; check that
    Flow-GRPO form produces σ_t = η·√(t/(1−t)) under that contract."""
    latents, model_output, sigma, sigma_next, eta, pred_original = _make_inputs()
    _, std_dev_t, sqrt_dt = compute_sde_step(
        "flow_grpo", model_output, latents, eta, sigma, sigma_next, pred_original
    )
    sigma_t_recovered = std_dev_t / sqrt_dt
    expected = eta * math.sqrt(sigma.item() / (1.0 - sigma.item()))
    assert sigma_t_recovered.item() == pytest.approx(expected, rel=1e-12)


def test_flow_grpo_sde_mean_matches_upstream_formula():
    """Flow-GRPO upstream (yifan123/flow_grpo sd3_sde_with_logprob.py)::

        prev_sample_mean = sample · (1 + std² /(2σ) · dt)
                         + model_output · (1 + std²·(1−σ)/(2σ)) · dt

    where ``dt = σ_next − σ`` (negative) and
    ``std = √(σ/(1−σ)) · noise_level``.  Pinned verbatim.
    """
    latents, model_output, sigma, sigma_next, eta, pred_original = _make_inputs()
    new_mean, _, _ = compute_sde_step(
        "flow_grpo", model_output, latents, eta, sigma, sigma_next, pred_original
    )

    dt = sigma_next - sigma  # negative
    std = torch.sqrt(sigma / (1.0 - sigma)) * eta
    expected = (
        latents * (1.0 + (std ** 2) / (2.0 * sigma) * dt)
        + model_output * (1.0 + (std ** 2) * (1.0 - sigma) / (2.0 * sigma)) * dt
    )
    assert torch.allclose(new_mean, expected, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# Cross-form sanity (interaction with downstream BGPO/VIPO/joint layers)
# ---------------------------------------------------------------------------


def test_forms_differ_at_nonzero_eta():
    """The two forms produce different prev_sample_mean / std_dev_t
    when η > 0 — a regression check that one form was not silently
    routed to the other."""
    latents, model_output, sigma, sigma_next, eta, pred_original = _make_inputs()
    dance_mean, dance_std, _ = compute_sde_step(
        "dancegrpo", model_output, latents, eta, sigma, sigma_next, pred_original
    )
    flow_mean, flow_std, _ = compute_sde_step(
        "flow_grpo", model_output, latents, eta, sigma, sigma_next, pred_original
    )
    assert not torch.allclose(dance_mean, flow_mean)
    assert not torch.allclose(dance_std, flow_std)


def test_both_forms_share_output_shape_and_dtype():
    """sigma_form is *orthogonal* to BGPO/VIPO/joint: switching the form
    must not change the rank, shape, or dtype of the SDE-step outputs,
    so downstream advantage / pixel-weight / aggregation layers are
    unaffected by the form choice.  This pins the contract.
    """
    latents, model_output, sigma, sigma_next, eta, pred_original = _make_inputs()
    dance_mean, dance_std, dance_sqrt_dt = compute_sde_step(
        "dancegrpo", model_output, latents, eta, sigma, sigma_next, pred_original
    )
    flow_mean, flow_std, flow_sqrt_dt = compute_sde_step(
        "flow_grpo", model_output, latents, eta, sigma, sigma_next, pred_original
    )
    assert dance_mean.shape == flow_mean.shape == latents.shape
    assert dance_mean.dtype == flow_mean.dtype == latents.dtype
    assert dance_std.shape == flow_std.shape  # both scalar tensors
    assert dance_std.dtype == flow_std.dtype
    assert dance_sqrt_dt.shape == flow_sqrt_dt.shape
    assert dance_sqrt_dt.dtype == flow_sqrt_dt.dtype


def test_unknown_form_raises():
    latents, model_output, sigma, sigma_next, eta, pred_original = _make_inputs()
    with pytest.raises(ValueError, match=r"unknown sigma_form"):
        compute_sde_step(
            "not_a_real_form",
            model_output, latents, eta, sigma, sigma_next, pred_original,
        )


def test_flow_grpo_sigma_one_edge_case_no_nan():
    """``σ=1`` is the start of the Wan/SD3 schedule and a pole of the
    Flow-GRPO formula (``1-σ`` in the denominator).  Mirrors the
    Flow-GRPO upstream substitution: replace ``σ=1`` with ``σ_next``
    (paper's ``sigma_max``) so ``std_dev_t`` stays finite.

    Without this, the first denoise step produces ``inf`` σ_t →
    NaN log-probs → NaN gradients (verified in remote smoke 2026-05-05).
    """
    latents, model_output, _, _, eta, _ = _make_inputs()
    sigma = torch.tensor(1.0, dtype=torch.float64)
    sigma_next = torch.tensor(0.95, dtype=torch.float64)
    pred_original = latents - sigma * model_output

    mean, std_dev_t, sqrt_dt = compute_sde_step(
        "flow_grpo", model_output, latents, eta, sigma, sigma_next, pred_original
    )
    assert torch.isfinite(mean).all(), "prev_sample_mean must be finite at σ=1"
    assert torch.isfinite(std_dev_t).all(), "std_dev_t must be finite at σ=1"
    assert torch.isfinite(sqrt_dt).all(), "sqrt_dt must be finite at σ=1"

    # The substituted σ_t should equal the formula at σ_next, not σ=1.
    expected_sigma_t = eta * math.sqrt(sigma_next.item() / (1.0 - sigma_next.item()))
    expected_std = expected_sigma_t * math.sqrt(sigma.item() - sigma_next.item())
    assert std_dev_t.item() == pytest.approx(expected_std, rel=1e-12), (
        "std_dev_t at σ=1 should use sigma_next as the denominator offset "
        "(matches Flow-GRPO upstream sd3_sde_with_logprob.py)"
    )


def test_registry_keys_match_grpo_guard():
    """sigma_schedule and grpo_guard registries must share keys: each
    σ_t form on the SDE side has its matching δ form on the loss side."""
    from recipe.teleboost.algorithms.grpo_guard import GRAD_REWEIGHT_FORMS

    assert set(SIGMA_FORMS.keys()) == set(GRAD_REWEIGHT_FORMS.keys()), (
        "σ_t form keys must match grad-reweight form keys "
        "(SDE-side and loss-side σ_t conventions must align)."
    )

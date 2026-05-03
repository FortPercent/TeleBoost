"""Multi-reward aggregation helpers.

These helpers are used by ``algorithms/joint.py`` to combine the per-task
advantages of a *joint reward* (e.g. aesthetic + RAFT + VideoCLIP-XL +
VideoPhy) into a single scalar advantage.

**Origin and scope.**  The convex weight scheme implemented here
(:func:`compute_joint_task_weights`) is **not** part of the BGPO paper
(arxiv 2511.18919) — that paper specifies only single-scalar-reward
optimization with a per-prompt-group scalar weight
``w_g = 1 + α [2σ(k(R̄_g − R_prior)) − 1]`` (see :mod:`bgpo`).  The
function here was previously housed in ``algorithms/bgpo.py`` for
namespace convenience, which made it look like a paper-described BGPO
construct; it is in fact an in-house design choice for multi-reward
joint training and is not validated by any published reference at the
time of writing.

Pulling it out into a separate module makes that boundary explicit:
``algorithms/bgpo.py`` should hold only the paper-faithful BGPO code
path (CRT reward rearrangement + RAS scalar weight); experimental /
in-house extensions that operate on multi-reward matrices live here.
"""

from __future__ import annotations

import numpy as np
import torch


def compute_joint_task_weights(advantages: torch.Tensor) -> torch.Tensor:
    """Compute per-sample convex weights from multi-reward advantages.

    Each row of ``advantages`` (shape ``(B, K)`` for ``B`` samples and
    ``K`` reward heads) is one sample's per-task advantage vector.  The
    function returns a ``(B, K)`` matrix where each row is a convex
    weight vector (non-negative entries that sum to 1) selecting which
    task heads to credit for that sample's final scalar advantage.

    Per-row rule:

    * Mixed signs (``min(a) <= 0 <= max(a)``) — pick the (argmin, argmax)
      pair and choose ``t`` such that ``(1-t) a_lo + t a_hi = 0``;
      everything else gets weight zero.  Intuition: when the reward
      heads disagree on the sample, return the conservative
      zero-bracketing combination so the joint advantage does not
      strongly push the policy in either direction until the heads
      align.
    * All positive (``min(a) > 0``) — full weight on ``argmin``.
      Intuition: when every head agrees the sample is "good", pick the
      *least* enthusiastic head as a conservative representative.
    * All negative (``max(a) < 0``) — full weight on ``argmax``.
      Intuition: dual of the previous case.

    **Not from the BGPO paper.**  See the module docstring; this is an
    in-house multi-reward aggregation rule whose behaviour has not been
    independently validated against a published reference.
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


__all__ = ["compute_joint_task_weights"]

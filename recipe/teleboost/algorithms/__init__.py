"""DanceGRPO supported algorithms.

One module per algorithm so additions / removals stay visible in the
directory listing.

* :mod:`bgpo`  — Bayesian-Prior Group Optimization. CRT (reward
  rearrangement) and RAS (adaptive advantage scaling) branches.
* :mod:`vipo`  — Pixel-weighted dense advantage broadcast via DINOv2.
* :mod:`joint` — Multi-head joint reward (worker-side parallel groups,
  legacy fixed 4-model runner, dynamic driver-side runner).

Each module exposes:

* A pure-function compute / helper API at module level.
* A ``*Mixin`` class that the trainer (``RayTeleBoostTrainer``) inherits
  from to gain the algorithm's hooks.

When the algorithm's enable flag is False the mixin is a no-op and
training falls back to baseline GRPO bit-for-bit.
"""

from recipe.teleboost.algorithms.bgpo import (
    BGPOMixin,
    paper_ras_centered_weight,
    rerange_group_rewards,
)
from recipe.teleboost.algorithms.grpo_advantage import (
    per_prompt_zscore_advantage,
)
from recipe.teleboost.algorithms.grpo_guard import (
    GRAD_REWEIGHT_FORMS,
    compute_grad_reweight_delta,
    compute_ratio_norm_bias,
)
from recipe.teleboost.algorithms.joint import (
    JointRewardMixin,
    _JointRewardRunner,
    merge_worker_results,
)
from recipe.teleboost.algorithms.multi_reward_aggregation import (
    compute_joint_task_weights,
)
from recipe.teleboost.algorithms.sigma_schedule import (
    SIGMA_FORMS,
    compute_sde_step,
)
from recipe.teleboost.algorithms.vipo import (
    VIPOMixin,
    compute_batch_pixel_weight_maps,
    compute_dinov2_feature_map_reverse,
)

__all__ = [
    # GRPO advantage (paper arxiv 2402.03300 + 2505.07818)
    "per_prompt_zscore_advantage",
    # BGPO (paper arxiv 2511.18919)
    "BGPOMixin",
    "rerange_group_rewards",
    "paper_ras_centered_weight",
    # GRPO-Guard (paper arxiv 2510.22319)
    "compute_ratio_norm_bias",
    "compute_grad_reweight_delta",
    "GRAD_REWEIGHT_FORMS",
    # Joint reward
    "JointRewardMixin",
    "_JointRewardRunner",
    "merge_worker_results",
    # Multi-reward aggregation (in-house, NOT from BGPO paper)
    "compute_joint_task_weights",
    # SDE σ_t schedule registry (DanceGRPO 2505.07818 vs Flow-GRPO 2505.05470)
    "SIGMA_FORMS",
    "compute_sde_step",
    # VIPO
    "VIPOMixin",
    "compute_batch_pixel_weight_maps",
    "compute_dinov2_feature_map_reverse",
]

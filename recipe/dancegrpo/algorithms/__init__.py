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
* A ``*Mixin`` class that the trainer (``RayDanceGRPOTrainer``) inherits
  from to gain the algorithm's hooks.

When the algorithm's enable flag is False the mixin is a no-op and
training falls back to baseline GRPO bit-for-bit.
"""

from recipe.dancegrpo.algorithms.bgpo import (
    BGPOMixin,
    rerange_group_rewards,
)
from recipe.dancegrpo.algorithms.grpo_guard import (
    GRAD_REWEIGHT_FORMS,
    compute_grad_reweight_delta,
    compute_ratio_norm_bias,
)
from recipe.dancegrpo.algorithms.joint import (
    JointRewardMixin,
    _JointRewardRunner,
    merge_worker_results,
)
from recipe.dancegrpo.algorithms.multi_reward_aggregation import (
    compute_joint_task_weights,
)
from recipe.dancegrpo.algorithms.vipo import (
    VIPOMixin,
    compute_batch_pixel_weight_maps,
    compute_dinov2_feature_map_reverse,
)

__all__ = [
    # BGPO (paper arxiv 2511.18919)
    "BGPOMixin",
    "rerange_group_rewards",
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
    # VIPO
    "VIPOMixin",
    "compute_batch_pixel_weight_maps",
    "compute_dinov2_feature_map_reverse",
]

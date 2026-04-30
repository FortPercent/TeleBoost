"""Backward-compat shim. Implementation moved to ``algorithms/vipo.py``.

Kept as a thin re-export so the diffusion rollout's
``from recipe.dancegrpo.pixel_weight_utils import compute_batch_pixel_weight_maps``
import keeps working.  New code should import directly from
``recipe.dancegrpo.algorithms.vipo``.
"""

from recipe.dancegrpo.algorithms.vipo import (
    compute_batch_pixel_weight_maps,
    compute_batch_pixel_weight_maps_pixel,
    compute_dinov2_feature_map_reverse,
    compute_dinov2_feature_map_reverse_pixel,
)

__all__ = [
    "compute_batch_pixel_weight_maps",
    "compute_batch_pixel_weight_maps_pixel",
    "compute_dinov2_feature_map_reverse",
    "compute_dinov2_feature_map_reverse_pixel",
]

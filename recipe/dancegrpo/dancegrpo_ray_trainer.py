# Thin wrapper to avoid duplicate trainer implementations.
from verl.utils.dataset.dancegrpo_ray_trainer import (
    RayDanceGRPOTrainer,
    compute_advantage,
    merge_worker_results,
)

__all__ = ["RayDanceGRPOTrainer", "compute_advantage", "merge_worker_results"]

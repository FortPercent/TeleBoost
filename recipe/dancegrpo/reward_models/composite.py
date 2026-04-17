# Copyright 2024 Dance-GRPO Team
"""
Composite Reward Manager - Manages multiple reward models and aggregates their scores.

This module provides a unified interface for running multiple reward models
in parallel and combining their results with configurable weights.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
from tensordict import TensorDict

from verl import DataProto

from .base import BaseRewardModel, RewardConfig, zscore_normalize, make_reward_batch
from .registry import RewardRegistry

logger = logging.getLogger(__name__)


@dataclass
class CompositeRewardConfig:
    """
    Configuration for composite reward computation.
    
    Attributes:
        models: List of individual reward model configurations
        aggregation: How to combine rewards ("weighted_sum", "mean", "max", "min")
        normalize_final: Whether to normalize the final aggregated reward
        parallel_compute: Whether to compute rewards in parallel
    """
    models: List[RewardConfig] = field(default_factory=list)
    aggregation: str = "weighted_sum"
    normalize_final: bool = False
    parallel_compute: bool = True
    
    @classmethod
    def from_dict(cls, d: Dict) -> "CompositeRewardConfig":
        """Create from a dictionary (e.g., from YAML config)."""
        models = [
            RewardConfig.from_dict(m) if isinstance(m, dict) else m
            for m in d.get("models", [])
        ]
        return cls(
            models=models,
            aggregation=d.get("aggregation", "weighted_sum"),
            normalize_final=d.get("normalize_final", False),
            parallel_compute=d.get("parallel_compute", True),
        )


class CompositeRewardManager:
    """
    Manages multiple reward models and aggregates their results.
    
    This class provides:
    - Unified initialization of multiple reward models
    - Parallel or sequential reward computation
    - Configurable reward aggregation strategies
    - Proper handling of dummy results from inactive workers
    
    Example usage:
        config = CompositeRewardConfig(
            models=[
                RewardConfig(name="aesthetic", weight=1.0, dp_fraction=0.25),
                RewardConfig(name="raft", weight=1.0, dp_fraction=0.5),
            ],
            aggregation="weighted_sum"
        )
        manager = CompositeRewardManager(config, rank=0, world_size=8)
        manager.init_all_models()
        
        result = manager.compute_and_aggregate(data)
    """
    
    def __init__(
        self, 
        config: CompositeRewardConfig, 
        rank: int, 
        world_size: int
    ):
        """
        Initialize the composite reward manager.
        
        Args:
            config: Configuration for composite rewards
            rank: Current process rank
            world_size: Total number of processes
        """
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.models: Dict[str, BaseRewardModel] = {}
        self._executor: Optional[ThreadPoolExecutor] = None
        
        # Create reward model instances
        for model_config in config.models:
            if not model_config.enabled:
                logger.info(f"Skipping disabled model: {model_config.name}")
                continue
                
            try:
                model = RewardRegistry.create(
                    model_config.name,
                    model_config,
                    rank,
                    world_size
                )
                self.models[model_config.name] = model
                logger.info(f"Created reward model: {model_config.name}")
            except Exception as e:
                logger.error(f"Failed to create model {model_config.name}: {e}")
                raise
    
    def init_all_models(self) -> None:
        """Initialize all registered reward models."""
        for name, model in self.models.items():
            if model.is_active:
                logger.info(f"Initializing model: {name}")
                model.init_model()
            else:
                logger.info(f"Skipping init for inactive model: {name}")
    
    def compute_rewards(self, data: DataProto) -> Dict[str, DataProto]:
        """
        Compute rewards from all models.
        
        Args:
            data: Input data with video frames and captions
            
        Returns:
            Dictionary mapping model names to their reward DataProtos
        """
        if self.config.parallel_compute and len(self.models) > 1:
            return self._compute_parallel(data)
        return self._compute_sequential(data)
    
    def _compute_sequential(self, data: DataProto) -> Dict[str, DataProto]:
        """Compute rewards sequentially."""
        results = {}
        for name, model in self.models.items():
            start = time.time()
            results[name] = model.compute_batch_score(data)
            logger.info(f"Model {name} took {time.time() - start:.2f}s")
        return results
    
    def _compute_parallel(self, data: DataProto) -> Dict[str, DataProto]:
        """Compute rewards in parallel using threads."""
        results = {}
        threads = {}
        
        def compute_one(name: str, model: BaseRewardModel) -> tuple:
            return name, model.compute_batch_score(data)
        
        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            futures = {
                executor.submit(compute_one, name, model): name
                for name, model in self.models.items()
            }
            
            for future in as_completed(futures):
                name, result = future.result()
                results[name] = result
        
        return results
    
    def aggregate_rewards(
        self, 
        rewards: Dict[str, DataProto],
        batch_size: int
    ) -> torch.Tensor:
        """
        Aggregate rewards from multiple models.
        
        Args:
            rewards: Dictionary of reward DataProtos from each model
            batch_size: Expected batch size
            
        Returns:
            Aggregated reward tensor
        """
        method = self.config.aggregation
        
        # Build weight mapping
        weight_map = {
            cfg.name: cfg.weight 
            for cfg in self.config.models 
            if cfg.enabled
        }
        
        if method == "weighted_sum":
            return self._aggregate_weighted_sum(rewards, weight_map)
        elif method == "mean":
            return self._aggregate_mean(rewards)
        elif method == "max":
            return self._aggregate_max(rewards)
        elif method == "min":
            return self._aggregate_min(rewards)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def _aggregate_weighted_sum(
        self, 
        rewards: Dict[str, DataProto],
        weight_map: Dict[str, float]
    ) -> torch.Tensor:
        """Compute weighted sum of rewards."""
        total = None
        
        for name, reward_proto in rewards.items():
            # Get the reward tensor (key might be model-specific)
            reward_key = self.models[name].REWARD_KEY
            reward_tensor = reward_proto.batch[reward_key]
            
            weight = weight_map.get(name, 1.0)
            weighted = reward_tensor * weight
            
            if total is None:
                total = weighted
            else:
                total = total + weighted
        
        return total
    
    def _aggregate_mean(self, rewards: Dict[str, DataProto]) -> torch.Tensor:
        """Compute mean of all rewards."""
        tensors = []
        for name, reward_proto in rewards.items():
            reward_key = self.models[name].REWARD_KEY
            tensors.append(reward_proto.batch[reward_key])
        
        stacked = torch.stack(tensors, dim=0)
        return stacked.mean(dim=0)
    
    def _aggregate_max(self, rewards: Dict[str, DataProto]) -> torch.Tensor:
        """Take maximum reward across models."""
        tensors = []
        for name, reward_proto in rewards.items():
            reward_key = self.models[name].REWARD_KEY
            tensors.append(reward_proto.batch[reward_key])
        
        stacked = torch.stack(tensors, dim=0)
        return stacked.max(dim=0).values
    
    def _aggregate_min(self, rewards: Dict[str, DataProto]) -> torch.Tensor:
        """Take minimum reward across models."""
        tensors = []
        for name, reward_proto in rewards.items():
            reward_key = self.models[name].REWARD_KEY
            tensors.append(reward_proto.batch[reward_key])
        
        stacked = torch.stack(tensors, dim=0)
        return stacked.min(dim=0).values
    
    def compute_and_aggregate(self, data: DataProto) -> DataProto:
        """
        Compute all rewards and return aggregated result.
        
        This is the main entry point for reward computation.
        
        Args:
            data: Input DataProto with video frames and captions
            
        Returns:
            DataProto with aggregated rewards and individual reward tensors
        """
        start_time = time.time()
        
        # Compute all individual rewards
        individual_rewards = self.compute_rewards(data)
        
        # Determine batch size from first result
        first_name = next(iter(individual_rewards))
        first_key = self.models[first_name].REWARD_KEY
        batch_size = individual_rewards[first_name].batch[first_key].shape[0]
        
        # Aggregate
        aggregated = self.aggregate_rewards(individual_rewards, batch_size)
        
        # Optionally normalize final result
        if self.config.normalize_final:
            aggregated = zscore_normalize(aggregated)
        
        # Build result with all rewards
        result_dict = {"rewards": aggregated}
        for name, reward_proto in individual_rewards.items():
            reward_key = self.models[name].REWARD_KEY
            result_dict[reward_key] = reward_proto.batch[reward_key]
        
        batch = TensorDict(result_dict, batch_size=batch_size)
        
        elapsed = time.time() - start_time
        logger.info(f"Total reward computation took {elapsed:.2f}s")
        
        return DataProto(batch=batch)
    
    def get_model(self, name: str) -> Optional[BaseRewardModel]:
        """Get a specific reward model by name."""
        return self.models.get(name)
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self.models.keys())
    
    def __len__(self) -> int:
        """Return number of registered models."""
        return len(self.models)


def merge_worker_results(
    data_list: List[DataProto], 
    use_valid_tensor_only: bool = True
) -> DataProto:
    """
    Merge results from multiple DP workers.
    
    This replaces the original merge_worker_results with improved logic
    that uses explicit validity tracking instead of zero-checking.
    
    Args:
        data_list: List of DataProto from different workers
        use_valid_tensor_only: If True, skip tensors that are marked as dummy
        
    Returns:
        Merged DataProto
    """
    if not data_list:
        return DataProto()
    if isinstance(data_list, DataProto):
        return data_list
    if len(data_list) == 1:
        return data_list[0]
    
    # Collect all keys
    all_batch_keys = set()
    all_non_tensor_keys = set()
    
    for dp in data_list:
        if dp.batch is not None:
            all_batch_keys.update(dp.batch.keys())
        if dp.non_tensor_batch is not None:
            all_non_tensor_keys.update(dp.non_tensor_batch.keys())
    
    # Merge batch tensors
    batch_dict = {}
    for key in all_batch_keys:
        tensors = []
        for dp in data_list:
            if dp.batch is not None and key in dp.batch:
                tensor = dp.batch[key]
                # Check for validity using meta info or tensor properties
                is_valid = True
                if use_valid_tensor_only:
                    # Use meta info if available, otherwise check if any non-zero
                    if hasattr(dp, 'meta_info') and dp.meta_info:
                        is_valid = dp.meta_info.get(f'{key}_valid', True)
                    # Fallback: check if tensor has meaningful values
                    # This is a heuristic - better to use explicit validity flags
                
                if is_valid:
                    tensors.append(tensor)
        
        if tensors:
            batch_dict[key] = torch.cat(tensors, dim=0)
    
    # Merge non-tensor batches
    non_tensor_dict = {}
    for key in all_non_tensor_keys:
        arrays = []
        for dp in data_list:
            if dp.non_tensor_batch is not None and key in dp.non_tensor_batch:
                arrays.append(dp.non_tensor_batch[key])
        
        if arrays:
            non_tensor_dict[key] = np.concatenate(arrays, axis=0)
    
    return DataProto.from_dict(tensors=batch_dict, non_tensors=non_tensor_dict)

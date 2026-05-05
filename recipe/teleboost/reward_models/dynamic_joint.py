# Copyright 2024 Dance-GRPO Team
"""
Dynamic Joint Reward Runner

This module provides a flexible joint reward runner that can dynamically
load and manage any number of reward models based on configuration.
Supports configurable MPS allocation for GPU resource management.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
from tensordict import TensorDict

from verl import DataProto

from .base import BaseRewardModel, RewardConfig, MPSConfig, zscore_normalize, make_reward_batch
from .registry import RewardRegistry

logger = logging.getLogger(__name__)


@dataclass
class JointRewardConfig:
    """
    Configuration for dynamic joint reward computation.
    
    Attributes:
        models: List of reward model configurations
        aggregation: How to aggregate rewards ("weighted_sum", "mean", "max")
        normalize_individual: Whether to normalize each model's output
        normalize_final: Whether to normalize the final aggregated output
        parallel_compute: Whether to compute rewards in parallel threads
        mps: MPS configuration for GPU allocation
    """
    models: List[RewardConfig] = field(default_factory=list)
    aggregation: str = "weighted_sum"
    normalize_individual: bool = True
    normalize_final: bool = False
    parallel_compute: bool = True
    mps: MPSConfig = field(default_factory=MPSConfig)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JointRewardConfig":
        """Create from a dictionary (e.g., from YAML config)."""
        if d is None:
            return cls()
        
        # Parse MPS config
        mps = MPSConfig.from_dict(d.get("mps", {}))
        
        # Parse model configs with MPS inheritance
        models = []
        for model_dict in d.get("models", []):
            config = RewardConfig.from_dict(model_dict) if isinstance(model_dict, dict) else model_dict
            # Apply MPS percentage from global config if not set on model
            if config.mps_percentage == 0 and mps.enabled:
                config.mps_percentage = mps.get_percentage_for_model(config.name)
            models.append(config)
        
        return cls(
            models=models,
            aggregation=d.get("aggregation", "weighted_sum"),
            normalize_individual=d.get("normalize_individual", True),
            normalize_final=d.get("normalize_final", False),
            parallel_compute=d.get("parallel_compute", True),
            mps=mps,
        )


class DynamicJointRewardRunner:
    """
    A flexible reward runner that can handle any number of reward models.
    
    Unlike the static _JointRewardRunner, this class:
    - Loads models dynamically from configuration
    - Supports arbitrary number of reward models
    - Allows MPS configuration per model
    - Provides configurable aggregation strategies
    
    Usage:
        config = JointRewardConfig.from_dict(yaml_config["reward_model"]["joint"])
        runner = DynamicJointRewardRunner(config, rank=0, world_size=8)
        runner.init_all_models()
        
        result = runner.compute(batch_data)
    """
    
    def __init__(
        self, 
        config: JointRewardConfig, 
        rank: int, 
        world_size: int
    ):
        """
        Initialize the dynamic joint reward runner.
        
        Args:
            config: Configuration for joint rewards
            rank: Current process rank
            world_size: Total number of processes
        """
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.models: Dict[str, BaseRewardModel] = {}
        
        # Threading infrastructure for parallel computation
        self._thread_inputs: Dict[str, Any] = {}
        self._reward_results: Dict[str, Any] = {}
        self._ready_events: Dict[str, threading.Event] = {}
        self._done_events: Dict[str, threading.Event] = {}
        self._threads_started = False
        
        # Create model instances from config
        self._create_models()
    
    def _create_models(self) -> None:
        """Create reward model instances from configuration."""
        for model_config in self.config.models:
            if not model_config.enabled:
                logger.info(f"Skipping disabled model: {model_config.name}")
                continue
            
            try:
                model = RewardRegistry.create(
                    model_config.name,
                    model_config,
                    self.rank,
                    self.world_size
                )
                self.models[model_config.name] = model
                logger.info(
                    f"Created reward model: {model_config.name} "
                    f"(weight={model_config.weight}, dp_fraction={model_config.dp_fraction}, "
                    f"mps={model_config.mps_percentage}%)"
                )
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
        
        # Start background threads if using parallel computation
        if self.config.parallel_compute and len(self.models) > 1:
            self._start_threads()
    
    def _start_threads(self) -> None:
        """Start background threads for parallel reward computation."""
        if self._threads_started:
            return
        
        for name in self.models:
            self._thread_inputs[name] = None
            self._reward_results[name] = None
            self._ready_events[name] = threading.Event()
            self._done_events[name] = threading.Event()
            
            t = threading.Thread(
                target=self._thread_loop, 
                args=(name,), 
                daemon=True
            )
            t.start()
        
        self._threads_started = True
        logger.info(f"Started {len(self.models)} background threads for reward computation")
    
    def _thread_loop(self, name: str) -> None:
        """Background thread loop for a single reward model."""
        model = self.models[name]
        while True:
            self._ready_events[name].wait()
            self._ready_events[name].clear()
            
            try:
                result = model.compute_batch_score(self._thread_inputs[name])
                self._reward_results[name] = result
            except Exception as e:
                logger.error(f"Error computing {name} reward: {e}")
                self._reward_results[name] = None
            
            self._done_events[name].set()
    
    def compute(self, batch: DataProto) -> Dict[str, DataProto]:
        """
        Compute rewards from all models.
        
        Args:
            batch: Input DataProto with video frames and captions
            
        Returns:
            Dictionary mapping model names to their reward DataProtos
        """
        if self.config.parallel_compute and self._threads_started:
            return self._compute_parallel_threaded(batch)
        return self._compute_sequential(batch)
    
    def _compute_sequential(self, batch: DataProto) -> Dict[str, DataProto]:
        """Compute rewards sequentially."""
        results = {}
        for name, model in self.models.items():
            start = time.time()
            results[name] = model.compute_batch_score(batch)
            logger.debug(f"Model {name} took {time.time() - start:.2f}s")
        return results
    
    def _compute_parallel_threaded(self, batch: DataProto) -> Dict[str, DataProto]:
        """Compute rewards in parallel using pre-started threads."""
        # Dispatch work to all threads
        for name in self.models:
            self._thread_inputs[name] = batch
            self._done_events[name].clear()
            self._ready_events[name].set()
        
        # Wait for all threads to complete
        for name in self.models:
            self._done_events[name].wait()
        
        # Collect results
        return {name: self._reward_results[name] for name in self.models}
    
    def compute_and_aggregate(self, batch: DataProto) -> DataProto:
        """
        Compute all rewards and return aggregated result.
        
        Args:
            batch: Input DataProto with video frames and captions
            
        Returns:
            DataProto with aggregated rewards and individual reward tensors
        """
        start_time = time.time()
        
        # Compute all individual rewards
        individual_rewards = self.compute(batch)
        
        # Filter out None results (from errors)
        valid_rewards = {k: v for k, v in individual_rewards.items() if v is not None}
        
        if not valid_rewards:
            logger.error("No valid rewards computed!")
            raise RuntimeError("All reward computations failed")
        
        # Determine batch size from first result
        first_name = next(iter(valid_rewards))
        first_key = self.models[first_name].REWARD_KEY
        batch_size = valid_rewards[first_name].batch[first_key].shape[0]
        
        # Aggregate rewards
        aggregated = self._aggregate_rewards(valid_rewards, batch_size)
        
        # Apply final normalization if configured
        if self.config.normalize_final:
            aggregated = zscore_normalize(aggregated)
        
        # Build result with all rewards
        result_dict = {"rewards": aggregated}
        for name, reward_proto in valid_rewards.items():
            reward_key = self.models[name].REWARD_KEY
            result_dict[reward_key] = reward_proto.batch[reward_key]
        
        batch_result = TensorDict(result_dict, batch_size=batch_size)
        
        elapsed = time.time() - start_time
        logger.info(f"Total joint reward computation took {elapsed:.2f}s")
        
        return DataProto(batch=batch_result)
    
    def _aggregate_rewards(
        self, 
        rewards: Dict[str, DataProto], 
        batch_size: int
    ) -> torch.Tensor:
        """Aggregate rewards from multiple models."""
        method = self.config.aggregation
        
        if method == "weighted_sum":
            return self._aggregate_weighted_sum(rewards)
        elif method == "mean":
            return self._aggregate_mean(rewards)
        elif method == "max":
            return self._aggregate_max(rewards)
        elif method == "min":
            return self._aggregate_min(rewards)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def _get_weight(self, model_name: str) -> float:
        """Get the weight for a specific model."""
        for cfg in self.config.models:
            if cfg.name == model_name:
                return cfg.weight
        return 1.0
    
    def _aggregate_weighted_sum(self, rewards: Dict[str, DataProto]) -> torch.Tensor:
        """Compute weighted sum of rewards."""
        total = None
        total_weight = 0.0
        
        for name, reward_proto in rewards.items():
            reward_key = self.models[name].REWARD_KEY
            reward_tensor = reward_proto.batch[reward_key]
            weight = self._get_weight(name)
            total_weight += weight
            
            weighted = reward_tensor * weight
            if total is None:
                total = weighted
            else:
                total = total + weighted
        
        # Optionally normalize by total weight
        # return total / total_weight if total_weight > 0 else total
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
    
    def get_model(self, name: str) -> Optional[BaseRewardModel]:
        """Get a specific reward model by name."""
        return self.models.get(name)
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self.models.keys())
    
    def get_metrics(self, rewards: Dict[str, DataProto]) -> Dict[str, float]:
        """
        Get metrics for logging from reward results.
        
        Args:
            rewards: Dictionary of rewards from compute()
            
        Returns:
            Dictionary of metric name -> value
        """
        metrics = {}
        for name, reward_proto in rewards.items():
            if reward_proto is not None:
                reward_key = self.models[name].REWARD_KEY
                metric_key = f"train/rewards_{name}"
                metrics[metric_key] = reward_proto.batch[reward_key].mean().item()
        return metrics
    
    def __len__(self) -> int:
        """Return number of registered models."""
        return len(self.models)


def create_joint_runner_from_config(
    config_dict: Dict[str, Any],
    rank: int,
    world_size: int
) -> DynamicJointRewardRunner:
    """
    Factory function to create a DynamicJointRewardRunner from config dict.
    
    Args:
        config_dict: Configuration dictionary (from YAML)
        rank: Current process rank
        world_size: Total number of processes
        
    Returns:
        Configured DynamicJointRewardRunner instance
    """
    joint_config = JointRewardConfig.from_dict(config_dict)
    return DynamicJointRewardRunner(joint_config, rank, world_size)

# Copyright 2024 Dance-GRPO Team
"""
Base classes and utilities for Reward Models.

This module provides the abstract base class that all reward models should inherit from,
along with common utilities for data parallel processing and result aggregation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import time
import os

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl import DataProto
from verl.utils.device import get_device_id

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """
    Configuration for a single reward model.
    
    Attributes:
        name: Unique identifier for the reward model (must match registry name)
        model_path: Path to the model weights
        weight: Weight in composite reward calculation (default: 1.0)
        dp_fraction: Fraction of GPUs to use (e.g., 0.25 = 1/4 of all GPUs)
        rank_offset: Starting rank for this model's active workers
        enabled: Whether this model is enabled
        normalize: Whether to apply z-score normalization to rewards
        mps_percentage: CUDA MPS active thread percentage (0-100, 0 means disabled)
        extra_config: Model-specific configuration dictionary
    """
    name: str
    model_path: str = ""
    weight: float = 1.0
    dp_fraction: float = 1.0
    rank_offset: int = 0
    enabled: bool = True
    normalize: bool = True
    mps_percentage: int = 0  # 0 means MPS disabled, e.g., 25 means 25% GPU threads
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RewardConfig":
        """Create a RewardConfig from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def get_mps_env_value(self) -> Optional[str]:
        """Get the MPS environment variable value, or None if disabled."""
        if self.mps_percentage > 0:
            return str(self.mps_percentage)
        return None


@dataclass
class MPSConfig:
    """
    Configuration for CUDA MPS (Multi-Process Service) allocation.
    
    This allows fine-grained control over GPU resource allocation 
    when running multiple reward models concurrently.
    
    Attributes:
        enabled: Whether MPS is enabled globally
        default_percentage: Default percentage for models without explicit config
        model_percentages: Per-model MPS percentage overrides
    """
    enabled: bool = False
    default_percentage: int = 25  # Default 25% per model
    model_percentages: Dict[str, int] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MPSConfig":
        """Create from a dictionary (e.g., from YAML config)."""
        if d is None:
            return cls()
        return cls(
            enabled=d.get("enabled", False),
            default_percentage=d.get("default_percentage", 25),
            model_percentages=d.get("model_percentages", {}),
        )
    
    def get_percentage_for_model(self, model_name: str) -> int:
        """Get MPS percentage for a specific model."""
        return self.model_percentages.get(model_name, self.default_percentage)
    
    def apply_mps_env(self, model_name: str) -> None:
        """Apply MPS environment variable for a model."""
        if self.enabled:
            percentage = self.get_percentage_for_model(model_name)
            os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(percentage)
            logger.info(f"Set MPS percentage for {model_name}: {percentage}%")


def zscore_normalize(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Apply z-score normalization to a tensor.
    
    Args:
        values: Input tensor
        eps: Small value to prevent division by zero
        
    Returns:
        Normalized tensor with mean 0 and std 1
    """
    mean = values.mean()
    std = values.std() + eps
    return (values - mean) / std


def split_batch_for_dp(
    data: DataProto, 
    dp_size: int, 
    dp_rank: int
) -> DataProto:
    """
    Split a DataProto batch for data parallelism.
    
    Args:
        data: The full batch
        dp_size: Number of data parallel workers
        dp_rank: Current worker's rank within the DP group
        
    Returns:
        A subset of the data for this worker
    """
    batch_size = data.batch.batch_size[0]
    
    # Calculate start and end indices for this rank
    base = batch_size // dp_size
    remainder = batch_size % dp_size
    start = dp_rank * base + min(dp_rank, remainder)
    end = start + base + (1 if dp_rank < remainder else 0)
    
    # Slice the data
    local_batch = data.batch[start:end]
    local_non_tensor = {k: v[start:end] for k, v in data.non_tensor_batch.items()}
    
    return DataProto(batch=local_batch, non_tensor_batch=local_non_tensor)


def split_video_frames(data: DataProto, permute_to_tchw: bool = True) -> List[torch.Tensor]:
    """
    Split video frames from a DataProto batch into individual samples.
    
    Args:
        data: DataProto containing 'video_frames' in batch
        permute_to_tchw: If True, permute from (C, T, H, W) to (T, C, H, W)
        
    Returns:
        List of video tensors, one per sample
    """
    video_frames = data.batch['video_frames']  # Shape: (B, C, T, H, W)
    batch_size = data.batch.batch_size[0]
    
    # Split into individual samples
    frames_list = video_frames.chunk(batch_size, dim=0)
    frames_list = [f.squeeze(0) for f in frames_list]  # Remove batch dim
    
    if permute_to_tchw:
        # (C, T, H, W) -> (T, C, H, W)
        frames_list = [f.permute(1, 0, 2, 3) for f in frames_list]
    
    return frames_list


def split_captions(captions: np.ndarray, batch_size: int) -> List[str]:
    """
    Split captions array into a list of strings.
    
    Args:
        captions: Array of captions
        batch_size: Expected batch size
        
    Returns:
        List of caption strings
    """
    if isinstance(captions, np.ndarray):
        return [str(c) for c in captions[:batch_size]]
    return list(captions[:batch_size])


def make_reward_batch(
    reward_key: str, 
    rewards: torch.Tensor, 
    batch_size: int
) -> DataProto:
    """
    Create a DataProto containing rewards.
    
    Args:
        reward_key: Key name for the rewards (e.g., "aes_rewards")
        rewards: Tensor of reward values
        batch_size: Batch size for the TensorDict
        
    Returns:
        DataProto with the rewards
    """
    batch = TensorDict(
        {reward_key: rewards},
        batch_size=batch_size
    )
    return DataProto(batch=batch)


class BaseRewardModel(ABC):
    """
    Abstract base class for all reward models.
    
    Subclasses must implement:
        - init_model(): Initialize the model weights
        - compute_single_score(): Compute reward for a single sample
        
    The base class provides:
        - Data parallel batch splitting
        - Z-score normalization
        - Device management
        - MPS configuration
        - Timing and logging
    """
    
    # Subclasses should set this to specify the reward key name
    REWARD_KEY: str = "rewards"
    
    def __init__(
        self, 
        config: RewardConfig, 
        global_rank: int, 
        world_size: int
    ):
        """
        Initialize the reward model.
        
        Args:
            config: Configuration for this reward model
            global_rank: Current process's global rank
            world_size: Total number of processes
        """
        self.config = config
        self.global_rank = global_rank
        self.world_size = world_size
        self.device = None  # Set during init_model
        
        # Calculate DP info
        self.dp_size = max(1, int(world_size * config.dp_fraction))
        self.is_active = self._check_active()
        
        if self.is_active:
            self.local_dp_rank = (global_rank - config.rank_offset) % self.dp_size
            # Apply MPS settings if configured
            self._apply_mps_settings()
        else:
            self.local_dp_rank = -1
            
        logger.info(
            f"[{self.config.name}] Rank {global_rank}: "
            f"active={self.is_active}, dp_size={self.dp_size}, "
            f"local_rank={self.local_dp_rank}, mps={config.mps_percentage}%"
        )
    
    def _check_active(self) -> bool:
        """Check if this rank should be active for this reward model."""
        start_rank = self.config.rank_offset
        end_rank = start_rank + self.dp_size
        return start_rank <= self.global_rank < end_rank
    
    def _apply_mps_settings(self) -> None:
        """Apply MPS environment settings if configured."""
        mps_value = self.config.get_mps_env_value()
        if mps_value:
            os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = mps_value
            logger.info(
                f"[{self.config.name}] Set CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={mps_value}"
            )
    
    def get_mps_percentage(self) -> int:
        """Get the configured MPS percentage."""
        return self.config.mps_percentage
    
    def get_device(self) -> torch.device:
        """Get the device for this worker."""
        if self.device is None:
            self.device = get_device_id()
        return self.device
    
    @abstractmethod
    def init_model(self) -> None:
        """
        Initialize the reward model.
        
        Subclasses should load model weights here.
        This method is only called on active ranks.
        """
        pass
    
    @abstractmethod
    def compute_single_score(
        self, 
        video_frames: torch.Tensor, 
        caption: str
    ) -> float:
        """
        Compute the reward score for a single sample.
        
        Args:
            video_frames: Video frames tensor, shape depends on model
            caption: Text caption for the video
            
        Returns:
            Scalar reward value
        """
        pass
    
    def compute_batch_score(self, data: DataProto) -> DataProto:
        """
        Compute reward scores for a batch of samples.
        
        This method handles:
        - Checking if this worker is active
        - Splitting data for data parallelism
        - Calling compute_single_score for each sample
        - Normalizing results if configured
        - Packaging results into DataProto
        
        Args:
            data: Input DataProto with 'video_frames' and 'caption'
            
        Returns:
            DataProto with computed rewards
        """
        start_time = time.time()
        
        # Extract data
        extracted = data.pop(
            batch_keys=['video_frames'],
            non_tensor_batch_keys=['caption'],
        )
        
        # Handle inactive workers
        if not self.is_active:
            batch_size = extracted.batch.batch_size[0] // self.dp_size
            logger.info(f"[{self.config.name}] Rank {self.global_rank} inactive, returning zeros")
            dummy_rewards = torch.zeros(batch_size, device='cpu')
            return make_reward_batch(self.REWARD_KEY, dummy_rewards, batch_size)
        
        # Split data for this DP rank
        local_data = split_batch_for_dp(extracted, self.dp_size, self.local_dp_rank)
        batch_size = local_data.batch.batch_size[0]
        
        logger.info(f"[{self.config.name}] Rank {self.global_rank} processing batch_size={batch_size}")
        
        # Get video frames and captions
        video_frames_list = split_video_frames(local_data, permute_to_tchw=True)
        captions = split_captions(local_data.non_tensor_batch['caption'], batch_size)
        
        # Compute scores for each sample
        rewards = []
        for i, (frames, caption) in enumerate(zip(video_frames_list, captions)):
            score = self.compute_single_score(frames, caption)
            rewards.append(torch.tensor(score, device=self.get_device()))
        
        rewards = torch.stack(rewards)
        
        # Normalize if configured
        if self.config.normalize:
            rewards = zscore_normalize(rewards)
        
        # Move to CPU for return
        rewards = rewards.cpu()
        
        elapsed = time.time() - start_time
        logger.info(f"[{self.config.name}] compute time: {elapsed:.2f}s")
        
        return make_reward_batch(self.REWARD_KEY, rewards, batch_size)
    
    def move_model_to_device(self, model: torch.nn.Module) -> None:
        """Move a model to the compute device."""
        model.to(self.get_device())
    
    def move_model_to_cpu(self, model: torch.nn.Module) -> None:
        """Move a model back to CPU to free GPU memory."""
        model.to('cpu')

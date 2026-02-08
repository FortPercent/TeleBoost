# Copyright 2024 Dance-GRPO Team
"""
Reward Models Module

This module provides a unified abstraction for various reward models used in video generation RL.
Supports dynamic loading of reward models from configuration.
"""

from .registry import RewardRegistry
from .base import BaseRewardModel, RewardConfig, MPSConfig
from .composite import CompositeRewardManager, CompositeRewardConfig
from .dynamic_joint import (
    DynamicJointRewardRunner, 
    JointRewardConfig,
    create_joint_runner_from_config
)

# Import all reward models to trigger registration
from . import aesthetic
from . import raft
from . import videoclip
from . import videophy
from . import hps

__all__ = [
    "RewardRegistry",
    "BaseRewardModel", 
    "RewardConfig",
    "MPSConfig",
    "CompositeRewardManager",
    "CompositeRewardConfig",
    "DynamicJointRewardRunner",
    "JointRewardConfig",
    "create_joint_runner_from_config",
]


def list_available_models():
    """List all available reward model names."""
    return RewardRegistry.list_available()


def create_reward_model(name: str, config: RewardConfig, rank: int, world_size: int):
    """
    Create a reward model instance by name.
    
    Args:
        name: Registered name of the reward model
        config: Configuration for the model
        rank: Current process rank
        world_size: Total number of processes
        
    Returns:
        BaseRewardModel instance
    """
    return RewardRegistry.create(name, config, rank, world_size)

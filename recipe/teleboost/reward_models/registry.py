# Copyright 2024 Dance-GRPO Team
"""
Reward Model Registry - A registry pattern for managing reward models.

Usage:
    @RewardRegistry.register("my_reward")
    class MyRewardModel(BaseRewardModel):
        ...
    
    # Create instance
    model = RewardRegistry.create("my_reward", config, rank, world_size)
"""

from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseRewardModel, RewardConfig


class RewardRegistry:
    """
    Central registry for all reward models.
    
    Provides a decorator-based registration mechanism and factory method
    for creating reward model instances.
    """
    
    _registry: Dict[str, Type["BaseRewardModel"]] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a reward model class.
        
        Args:
            name: Unique identifier for the reward model
            
        Example:
            @RewardRegistry.register("aesthetic")
            class AestheticRewardModel(BaseRewardModel):
                pass
        """
        def decorator(model_cls: Type["BaseRewardModel"]) -> Type["BaseRewardModel"]:
            if name in cls._registry:
                raise ValueError(f"Reward model '{name}' is already registered")
            cls._registry[name] = model_cls
            return model_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type["BaseRewardModel"]:
        """
        Get a reward model class by name.
        
        Args:
            name: The registered name of the reward model
            
        Returns:
            The reward model class
            
        Raises:
            ValueError: If the name is not registered
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown reward model: '{name}'. Available models: {available}"
            )
        return cls._registry[name]
    
    @classmethod
    def create(
        cls, 
        name: str, 
        config: "RewardConfig", 
        rank: int, 
        world_size: int
    ) -> "BaseRewardModel":
        """
        Factory method to create a reward model instance.
        
        Args:
            name: The registered name of the reward model
            config: Configuration for the reward model
            rank: Current process rank
            world_size: Total number of processes
            
        Returns:
            An instance of the requested reward model
        """
        model_cls = cls.get(name)
        return model_cls(config, rank, world_size)
    
    @classmethod
    def list_available(cls) -> list:
        """List all registered reward model names."""
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a reward model name is registered."""
        return name in cls._registry

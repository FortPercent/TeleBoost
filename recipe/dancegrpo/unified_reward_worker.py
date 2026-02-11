# Copyright 2024 Dance-GRPO Team
"""
Unified Reward Model Worker

This module provides a unified worker that can dynamically load any reward model
from the Registry based on configuration. This eliminates the need for separate
worker classes for each reward model type.

Usage:
    In config:
        reward_model:
            type: single
            model_name: hps  # Any registered model name
            model_path: /path/to/model
            extra_config:
                model_type: ViT-H-14
    
    Or for joint mode (dict format - recommended):
        reward_model:
            type: joint
            joint:
                models:
                    aesthetic:
                      weight: 1.0
                      extra_config:
                        clip_model_path: /path/to/model
                      ...
    
    This allows Hydra overrides like:
        reward_model.joint.models.aesthetic.extra_config.clip_model_path=/new/path
"""

import logging
import os
import time
from typing import Dict, Any, List, Optional

import torch
import torch.distributed as dist
from tensordict import TensorDict
from omegaconf import DictConfig, OmegaConf

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.debug import WorkerProfiler
from verl.utils.device import get_device_id

from .reward_models import RewardRegistry, create_reward_model
from .reward_models.base import RewardConfig, split_batch_for_dp, zscore_normalize

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

from verl.utils.debug import ProfilerConfig, WorkerProfiler, WorkerProfilerExtension


class UnifiedRewardModelWorker(Worker, WorkerProfilerExtension):
    """
    Unified Reward Model Worker that dynamically loads models from Registry.
    
    This worker can handle:
    1. Single model mode: Loads one model based on `model_name`
    2. Joint mode: Loads multiple models from `joint.models` list
    
    All models are loaded from the RewardRegistry, ensuring consistent
    initialization and interface.
    
    Note: This class inherits from Worker (not RewardModelWorker) because
    it manages Registry-based models that have their own initialization logic.
    """
    
    def __init__(self, config, cuda_visible_devices=None):
        """Initialize the UnifiedRewardModelWorker.
        
        Args:
            config: Reward model configuration (OmegaConf DictConfig)
            cuda_visible_devices: CUDA visible devices configuration
        """
        Worker.__init__(self, cuda_visible_devices=cuda_visible_devices)
        WorkerProfilerExtension.__init__(
            self, 
            WorkerProfiler(rank=self.rank, config=ProfilerConfig(**OmegaConf.to_object(config.get("profiler", DictConfig({})))))
        )
        self.config = config
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize the reward model(s) based on configuration."""
        # Initialize torch.distributed if not already done
        if not dist.is_initialized():
            from verl.utils.device import get_nccl_backend
            dist.init_process_group(backend=get_nccl_backend())
        
        self.reward_models: Dict[str, Any] = {}
        self.model_weights: Dict[str, float] = {}
        self.model_configs: Dict[str, RewardConfig] = {}
        
        # Get global rank and world size
        # Use different names to avoid conflict with parent class properties
        self._global_rank = dist.get_rank() if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        rm_type = self.config.get("type", "single")
        
        # Store mode for compute_rm_score to use
        self._mode = "joint" if rm_type == "joint" else "single"
        
        if rm_type == "joint":
            self._init_joint_models()
        else:
            # single, qwen, or any other type -> treat as single model
            self._init_single_model()
        
        # Log available models
        available = RewardRegistry.list_available()
        logger.info(f"[UnifiedRM] Mode: {self._mode}, Available models: {available}")
        logger.info(f"[UnifiedRM] Initialized models: {list(self.reward_models.keys())}")
    
    def _init_single_model(self):
        """Initialize a single reward model."""
        model_name = self.config.get("model_name", None)
        
        if model_name is None:
            # Fallback: try to infer from type
            rm_type = self.config.get("type", "single")
            if rm_type in RewardRegistry.list_available():
                model_name = rm_type
            else:
                raise ValueError(
                    f"No model_name specified and type '{rm_type}' is not a registered model. "
                    f"Available models: {RewardRegistry.list_available()}"
                )
        
        # Build config
        # Support both config.model.path (original format) and config.model_path (joint mode format)
        if hasattr(self.config, "model") and self.config.model:
            model_path = self.config.model.get("path", "")
        else:
            model_path = self.config.get("model_path", "")
        
        extra_config = OmegaConf.to_container(
            self.config.get("extra_config", {}), resolve=True
        ) if self.config.get("extra_config") else {}
        
        config = RewardConfig(
            name=model_name,
            model_path=model_path,
            weight=1.0,
            dp_fraction=1.0,
            rank_offset=0,
            enabled=True,
            normalize=self.config.get("normalize", True),
            mps_percentage=self.config.get("mps_percentage", 0),
            extra_config=extra_config,
        )
        
        self._create_and_init_model(model_name, config)
    
    def _init_joint_models(self):
        """Initialize multiple reward models from joint configuration.
        
        Supports both dict format (recommended) and legacy list format:
        
        Dict format (allows Hydra name-based overrides):
            joint:
              models:
                aesthetic:
                  weight: 1.0
                  extra_config: {...}
        
        Legacy list format:
            joint:
              models:
                - name: aesthetic
                  weight: 1.0
        """
        joint_config = self.config.get("joint", {})
        models_config = joint_config.get("models", {})
        
        if not models_config:
            raise ValueError("Joint mode requires at least one model in 'joint.models'")
        
        # Detect format: dict (new) vs list (legacy)
        if isinstance(models_config, (list, tuple)):
            # Legacy list format
            model_items = []
            for model_cfg in models_config:
                model_name = model_cfg.get("name")
                if model_name:
                    model_items.append((model_name, model_cfg))
                else:
                    logger.warning("[UnifiedRM] Skipping model without 'name' in list format")
        else:
            # New dict format - models_config is DictConfig or dict
            model_items = list(OmegaConf.to_container(models_config, resolve=True).items())
        
        for model_name, model_cfg in model_items:
            if not model_cfg.get("enabled", True):
                continue
            
            if not RewardRegistry.is_registered(model_name):
                logger.warning(
                    f"[UnifiedRM] Model '{model_name}' not registered, skipping. "
                    f"Available: {RewardRegistry.list_available()}"
                )
                continue
            
            # Build RewardConfig
            extra_config = model_cfg.get("extra_config", {})
            if hasattr(extra_config, 'items'):
                # It's already a dict-like object
                extra_config = dict(extra_config)
            
            config = RewardConfig(
                name=model_name,
                model_path=model_cfg.get("model_path", ""),
                weight=model_cfg.get("weight", 1.0),
                dp_fraction=model_cfg.get("dp_fraction", 1.0),
                rank_offset=model_cfg.get("rank_offset", 0),
                enabled=True,
                normalize=model_cfg.get("normalize", True),
                mps_percentage=model_cfg.get("mps_percentage", 0),
                extra_config=extra_config,
            )
            
            self._create_and_init_model(model_name, config)
    
    def _create_and_init_model(self, model_name: str, config: RewardConfig):
        """Create and initialize a reward model from the registry."""
        try:
            model = create_reward_model(
                name=model_name,
                config=config,
                rank=self._global_rank,
                world_size=self._world_size,
            )
            model.init_model()
            
            self.reward_models[model_name] = model
            self.model_weights[model_name] = config.weight
            self.model_configs[model_name] = config
            
            logger.info(f"[UnifiedRM] Initialized '{model_name}' with weight {config.weight}")
            
        except Exception as e:
            logger.error(f"[UnifiedRM] Failed to initialize '{model_name}': {e}")
            raise
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @WorkerProfiler.annotate(color="blue")
    def compute_rm_score(self, data: DataProto) -> DataProto:
        """
        Compute reward scores using the initialized model(s).
        
        For single mode: Uses DP_COMPUTE_PROTO (data already split by framework)
        For joint mode: Uses DP_COMPUTE_PROTO but models handle their own DP splitting
        """
        start_time = time.time()
        data = data.to(get_device_id())
        
        if self._mode == "single":
            return self._compute_single_mode(data, start_time)
        else:
            return self._compute_joint_mode(data, start_time)
    
    def _compute_single_mode(self, data: DataProto, start_time: float) -> DataProto:
        """Compute scores for single model mode.
        
        Note: In joint parallel mode (multiple UnifiedRewardModelWorkers),
        we return model-specific keys (e.g., 'aesthetic_rewards') and let
        the driver handle aggregation. We don't add 'rewards' key here to
        avoid conflicts when union-ing results from multiple models.
        """
        model_name = list(self.reward_models.keys())[0]
        model = self.reward_models[model_name]
        
        # Single mode: data is already split per worker by DP_COMPUTE_PROTO
        result = model.compute_batch_score(data)
        
        # Return result with model-specific key (e.g., 'aesthetic_rewards')
        # Don't rename to 'rewards' - let driver aggregate multiple models
        
        elapsed = time.time() - start_time
        logger.info(f"[UnifiedRM] {model_name} compute time: {elapsed:.2f}s")
        return result
    
    def _compute_joint_mode(self, data: DataProto, start_time: float) -> DataProto:
        """Compute scores for joint model mode with multiple models."""
        all_rewards = {}
        batch_size = None
        
        for model_name, model in self.reward_models.items():
            try:
                # Joint mode: use compute_batch_score_for_joint which handles DP splitting
                # since multiple models may have different dp_fraction configurations
                result = model.compute_batch_score_for_joint(data)
                reward_key = model.REWARD_KEY
                rewards = result.batch[reward_key]
                all_rewards[model_name] = rewards
                
                if batch_size is None:
                    batch_size = result.batch.batch_size[0]
                    
            except Exception as e:
                logger.error(f"[UnifiedRM] Error computing {model_name}: {e}")
                continue
        
        if not all_rewards:
            raise RuntimeError("No rewards computed from any model")
        
        # Aggregate rewards
        aggregation = self.config.get("joint", {}).get("aggregation", "weighted_sum")
        final_rewards = self._aggregate_rewards(all_rewards, aggregation)
        
        # Optionally normalize final rewards
        if self.config.get("joint", {}).get("normalize_final", False):
            final_rewards = zscore_normalize(final_rewards)
        
        # Build result
        batch = TensorDict(
            {"rewards": final_rewards},
            batch_size=batch_size
        )
        
        # Also include individual rewards for logging
        for model_name, rewards in all_rewards.items():
            batch[f"{model_name}_rewards"] = rewards
        
        elapsed = time.time() - start_time
        logger.info(f"[UnifiedRM] Joint compute time: {elapsed:.2f}s")
        
        return DataProto(batch=batch, non_tensor_batch=data.non_tensor_batch)
    
    def _aggregate_rewards(
        self, 
        all_rewards: Dict[str, torch.Tensor], 
        aggregation: str
    ) -> torch.Tensor:
        """Aggregate rewards from multiple models."""
        if aggregation == "weighted_sum":
            weighted = []
            for model_name, rewards in all_rewards.items():
                weight = self.model_weights.get(model_name, 1.0)
                weighted.append(rewards * weight)
            return sum(weighted)
        
        elif aggregation == "mean":
            stacked = torch.stack(list(all_rewards.values()))
            return stacked.mean(dim=0)
        
        elif aggregation == "max":
            stacked = torch.stack(list(all_rewards.values()))
            return stacked.max(dim=0)[0]
        
        elif aggregation == "min":
            stacked = torch.stack(list(all_rewards.values()))
            return stacked.min(dim=0)[0]
        
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")


class JointRewardModelWorker(Worker, WorkerProfilerExtension):
    """
    Joint Reward Model Worker for combining multiple reward models.
    
    Uses ALL_TO_ALL dispatch so each worker receives the full data and handles
    its own DP splitting. This allows different models to run on different
    subsets of ranks (via dp_fraction configuration).
    
    Key difference from UnifiedRewardModelWorker in single mode:
    - Single mode: Uses DP_COMPUTE_PROTO, data pre-split by framework
    - Joint mode: Uses ALL_TO_ALL, data not pre-split, models handle splitting
    
    Note: This class inherits from Worker (not RewardModelWorker) because
    it manages Registry-based models that have their own initialization logic.
    """
    
    def __init__(self, config, cuda_visible_devices=None):
        """Initialize the JointRewardModelWorker.
        
        Args:
            config: Reward model configuration (OmegaConf DictConfig)
            cuda_visible_devices: CUDA visible devices configuration
        """
        Worker.__init__(self, cuda_visible_devices=cuda_visible_devices)
        WorkerProfilerExtension.__init__(
            self, 
            WorkerProfiler(rank=self.rank, config=ProfilerConfig(**OmegaConf.to_object(config.get("profiler", DictConfig({})))))
        )
        self.config = config
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize joint reward models from configuration."""
        # Initialize torch.distributed if not already done
        if not dist.is_initialized():
            from verl.utils.device import get_nccl_backend
            dist.init_process_group(backend=get_nccl_backend())
        
        self.reward_models: Dict[str, Any] = {}
        self.model_weights: Dict[str, float] = {}
        self.model_configs: Dict[str, RewardConfig] = {}
        
        self._global_rank = dist.get_rank() if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # Initialize models from joint configuration
        joint_cfg = self.config.get("joint", {})
        models_cfg = joint_cfg.get("models", {})
        
        if not models_cfg:
            raise ValueError("JointRewardModelWorker requires 'joint.models' configuration")
        
        # Detect format: dict (new) vs list (legacy)
        if isinstance(models_cfg, (list, tuple)):
            # Legacy list format
            model_items = []
            for model_cfg in models_cfg:
                model_name = model_cfg.get("name")
                if model_name:
                    model_items.append((model_name, model_cfg))
                else:
                    logger.warning("[JointRM] Skipping model config without 'name'")
        else:
            # New dict format
            model_items = list(OmegaConf.to_container(models_cfg, resolve=True).items())
        
        for model_name, model_cfg in model_items:
            if not model_cfg.get("enabled", True):
                continue
            
            # Build RewardConfig for this model
            extra_config = model_cfg.get("extra_config", {})
            if hasattr(extra_config, 'items'):
                extra_config = dict(extra_config)
            
            # Each model can have its own DP configuration:
            # - dp_fraction: fraction of workers that will run this model
            # - rank_offset: starting rank for the model's active workers
            # - mps_percentage: GPU resource allocation via MPS
            config = RewardConfig(
                name=model_name,
                model_path=model_cfg.get("model_path", "") or model_cfg.get("model", {}).get("path", ""),
                extra_config=extra_config,
                weight=model_cfg.get("weight", 1.0),
                normalize=model_cfg.get("normalize", True),
                dp_fraction=model_cfg.get("dp_fraction", 1.0),
                rank_offset=model_cfg.get("rank_offset", 0),
                mps_percentage=model_cfg.get("mps_percentage", 0),
            )
            
            self._create_and_init_model(model_name, config)
        
        logger.info(f"[JointRM] Initialized {len(self.reward_models)} models: "
                    f"{list(self.reward_models.keys())}")
    
    def _create_and_init_model(self, model_name: str, config: RewardConfig):
        """Create and initialize a reward model from the registry."""
        try:
            model = create_reward_model(
                name=model_name,
                config=config,
                rank=self._global_rank,
                world_size=self._world_size,
            )
            model.init_model()
            
            self.reward_models[model_name] = model
            self.model_weights[model_name] = config.weight
            self.model_configs[model_name] = config
            
            logger.info(f"[JointRM] Initialized '{model_name}' with weight {config.weight}")
            
        except Exception as e:
            import traceback
            logger.error(f"[JointRM] Failed to initialize '{model_name}': {e}")
            logger.error(f"[JointRM] Traceback:\n{traceback.format_exc()}")
            raise
    
    @register(dispatch_mode=Dispatch.ALL_TO_ALL)
    @WorkerProfiler.annotate(color="green")
    def compute_rm_score(self, data: DataProto) -> DataProto:
        """
        Compute reward scores using all joint models with per-model DP.
        
        Architecture:
        - ALL_TO_ALL dispatch: each worker receives the FULL batch
        - Each model has its own dp_fraction/rank_offset configuration
        - Active workers split data and compute; inactive return zeros
        - AllGather collects results from all workers per model
        - Final aggregation combines all model rewards
        
        This enables:
        - Different models running on different GPU subsets
        - MPS for GPU resource sharing within a subset
        - Correct batch_size output matching input
        """
        import torch.distributed as dist
        
        start_time = time.time()
        data = data.to(get_device_id())
        
        full_batch_size = data.batch.batch_size[0]
        all_rewards = {}
        
        for model_name, model in self.reward_models.items():
            try:
                # Each model uses its own DP configuration
                # compute_batch_score_for_joint returns LOCAL results (for this worker's portion)
                result = model.compute_batch_score_for_joint(data)
                reward_key = model.REWARD_KEY
                local_rewards = result.batch[reward_key]
                
                # AllGather to collect rewards from all workers for this model
                if dist.is_initialized() and model.dp_size > 1:
                    # Gather from all active workers in this model's DP group
                    gathered_rewards = self._allgather_rewards(
                        local_rewards, 
                        model.dp_size,
                        model.config.rank_offset,
                        model.is_active
                    )
                else:
                    gathered_rewards = local_rewards
                
                # Verify we got the right batch size
                if gathered_rewards.shape[0] != full_batch_size:
                    logger.warning(
                        f"[JointRM] {model_name}: expected {full_batch_size} rewards, "
                        f"got {gathered_rewards.shape[0]}"
                    )
                
                all_rewards[model_name] = gathered_rewards
                    
            except Exception as e:
                import traceback
                logger.error(f"[JointRM] Error computing {model_name}: {e}")
                logger.error(f"[JointRM] Traceback:\n{traceback.format_exc()}")
                continue
        
        if not all_rewards:
            raise RuntimeError("No rewards computed from any model")
        
        # Aggregate rewards
        joint_cfg = self.config.get("joint", {})
        aggregation = joint_cfg.get("aggregation", "weighted_sum")
        final_rewards = self._aggregate_rewards(all_rewards, aggregation)
        
        # Optionally normalize final rewards
        if joint_cfg.get("normalize_final", False):
            final_rewards = zscore_normalize(final_rewards)
        
        # Build result with full batch size
        batch = TensorDict(
            {"rewards": final_rewards},
            batch_size=full_batch_size
        )
        
        # Also include individual rewards for logging
        for model_name, rewards in all_rewards.items():
            batch[f"{model_name}_rewards"] = rewards
        
        elapsed = time.time() - start_time
        logger.info(f"[JointRM] Compute time: {elapsed:.2f}s, batch_size={full_batch_size}")
        
        return DataProto(batch=batch, non_tensor_batch=data.non_tensor_batch)
    
    def _allgather_rewards(
        self, 
        local_rewards: torch.Tensor,
        dp_size: int,
        rank_offset: int,
        is_active: bool
    ) -> torch.Tensor:
        """
        AllGather rewards from all active workers for a model.
        
        Args:
            local_rewards: This worker's portion of rewards
            dp_size: Number of workers in this model's DP group
            rank_offset: Starting rank for active workers
            is_active: Whether this worker is active for this model
            
        Returns:
            Full batch rewards (concatenated from all active workers)
        """
        import torch.distributed as dist
        
        local_size = local_rewards.shape[0]
        device = local_rewards.device
        
        # Create list to gather into
        gathered_list = [torch.zeros_like(local_rewards) for _ in range(dp_size)]
        
        # Only active workers in the DP group participate
        if is_active:
            # AllGather within the DP group
            # Note: We use all workers but only active ones have real data
            dist.all_gather(gathered_list, local_rewards)
        else:
            # Inactive workers also need to participate in collective
            dist.all_gather(gathered_list, local_rewards)
        
        # Concatenate gathered results
        full_rewards = torch.cat(gathered_list, dim=0)
        
        return full_rewards
    
    def _aggregate_rewards(
        self, 
        all_rewards: Dict[str, torch.Tensor], 
        aggregation: str
    ) -> torch.Tensor:
        """Aggregate rewards from multiple models."""
        if aggregation == "weighted_sum":
            weighted = []
            for model_name, rewards in all_rewards.items():
                weight = self.model_weights.get(model_name, 1.0)
                weighted.append(rewards * weight)
            return sum(weighted)
        
        elif aggregation == "mean":
            stacked = torch.stack(list(all_rewards.values()))
            return stacked.mean(dim=0)
        
        elif aggregation == "max":
            stacked = torch.stack(list(all_rewards.values()))
            return stacked.max(dim=0)[0]
        
        elif aggregation == "min":
            stacked = torch.stack(list(all_rewards.values()))
            return stacked.min(dim=0)[0]
        
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

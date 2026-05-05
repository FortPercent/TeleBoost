# Copyright 2024 Dance-GRPO Team
"""
Dance-GRPO Main Entry Point

This module provides the main entry point for Dance-GRPO training,
which applies Group Relative Policy Optimization to diffusion models
for video generation with multiple reward signals.
"""

import logging
import os
from typing import Dict, Optional, Tuple, Any

# Apply TeleBoost patches over upstream verl (cp grad fix, etc.) BEFORE any
# verl import below: subsequent `from verl.X import Y` then resolves to the
# patched symbols.
import teleboost  # noqa: F401

import hydra
import ray
from omegaconf import DictConfig, OmegaConf
from pprint import pprint

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local

from .teleboost_ray_trainer import RayDanceGRPOTrainer

logger = logging.getLogger(__name__)

# Ray environment variables for distributed training
RAY_ENV_VARS = {
    "TOKENIZERS_PARALLELISM": "true",
    "NCCL_DEBUG": "WARN",
    "VLLM_LOGGING_LEVEL": "WARN",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}


def _init_ray(config: DictConfig) -> None:
    """Initialize Ray runtime if not already initialized.

    verl 0.4.0 (which we pin in ``requirements-verl.txt``) places
    ``num_cpus`` at ``config.ray_init.num_cpus``.  Note that newer
    verl (≥0.6) moves this to ``ray_kwargs.ray_init.num_cpus``; do
    not "fix" this access if a future bump changes the path —
    update both here and the recipe yaml together.
    """
    if ray.is_initialized():
        logger.info("Ray already initialized, skipping")
        return

    num_cpus = config.ray_init.num_cpus
    ray.init(
        runtime_env={"env_vars": RAY_ENV_VARS},
        num_cpus=num_cpus,
    )
    logger.info(f"Ray initialized with num_cpus={num_cpus}")


def _build_tokenizer_and_processor(
    local_path: str,
    actor_rollout_ref_config: DictConfig,
) -> Tuple[Any, Any]:
    """Build tokenizer and processor for the Wan model.

    Reads ``actor_rollout_ref.tokenizer_subpath`` (default
    ``google/umt5-xxl``) and joins it with ``local_path`` (the downloaded
    Wan model directory).  This is **not** verl's
    ``actor_rollout_ref.model.tokenizer_path`` — that key is a full HF
    tokenizer path and defaults to null in ``hf_model.yaml``.  We keep
    the two keys separate so verl's default cannot collide with the
    recipe-level subpath.

    Raises:
        RuntimeError: If tokenizer loading fails.
    """
    from verl.utils import hf_processor, hf_tokenizer

    tokenizer_subpath = actor_rollout_ref_config.get("tokenizer_subpath", "google/umt5-xxl")
    tokenizer_path = os.path.join(local_path, tokenizer_subpath)

    try:
        tokenizer = hf_tokenizer(tokenizer_path)
        processor = hf_processor(local_path, use_fast=True)
        logger.info(f"Loaded tokenizer from {tokenizer_path}")
        return tokenizer, processor
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer from {tokenizer_path}: {e}")


def _resolve_actor_worker_and_group(config: DictConfig) -> Tuple[type, type]:
    """
    Resolve the worker class and group class based on strategy.
    
    Args:
        config: Configuration with actor_rollout_ref settings
        
    Returns:
        Tuple of (RayWorkerGroupClass, ActorWorkerClass)
        
    Raises:
        NotImplementedError: If strategy is not supported
    """
    strategy = config.actor_rollout_ref.actor.strategy
    
    # Only check critic strategy if critic is enabled
    use_critic = config.algorithm.get("adv_estimator", "grpo") == "gae"
    if use_critic and hasattr(config, 'critic'):
        assert strategy == config.critic.strategy, (
            f"Actor strategy ({strategy}) must match critic strategy ({config.critic.strategy})"
        )

    if strategy == "fsdp":
        from verl.single_controller.ray import RayWorkerGroup
        from .teleboost_fsdp_worker import DiffusionActorRolloutRefWorker
        return RayWorkerGroup, DiffusionActorRolloutRefWorker

    if strategy == "megatron":
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        from verl.workers.megatron_workers import ActorRolloutRefWorker
        return NVMegatronRayWorkerGroup, ActorRolloutRefWorker

    raise NotImplementedError(f"Unknown actor strategy: {strategy}")


def _build_reward_fns(config: DictConfig, tokenizer):
    """
    Build reward functions for training and validation.
    
    Args:
        config: Configuration with reward_model settings
        tokenizer: Tokenizer for the model
        
    Returns:
        Tuple of (reward_fn, val_reward_fn)
    """
    from verl.workers.reward_manager import get_reward_manager_cls

    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    reward_manager_cls = get_reward_manager_cls(reward_manager_name)
    compute_score = get_custom_reward_fn(config)

    # Training reward function
    reward_fn = reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=0,
        compute_score=compute_score,
    )

    # Validation reward function (with sample examination)
    val_reward_fn = reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=1,
        compute_score=compute_score,
    )
    
    return reward_fn, val_reward_fn


def _validate_config(config: DictConfig) -> None:
    """
    Validate Dance-GRPO specific configuration.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate reward model configuration
    if config.reward_model.enable:
        strategy = config.reward_model.strategy
        valid_strategies = ["fsdp", "megatron", "diffusion"]
        if strategy not in valid_strategies:
            raise ValueError(
                f"Invalid reward_model.strategy: {strategy}. "
                f"Must be one of {valid_strategies}"
            )
        
        if strategy == "diffusion":
            rm_type = config.reward_model.type
            valid_types = ["qwen", "single", "joint"]
            if rm_type not in valid_types:
                raise ValueError(
                    f"Invalid reward_model.type: {rm_type}. "
                    f"Must be one of {valid_types}"
                )
    
    logger.info("Configuration validation passed")


def _register_reward_workers(
    config: DictConfig,
    role_worker_mapping: Dict,
    mapping: Dict,
    global_pool_id: str
) -> None:
    """
    Register reward model workers based on configuration.
    
    For diffusion strategy, all reward model types (single, joint, etc.) 
    are handled by the UnifiedRewardModelWorker, which dynamically loads
    models from the RewardRegistry based on model_name configuration.
    
    Args:
        config: Configuration with reward_model settings
        role_worker_mapping: Dict to store role -> worker class mappings
        mapping: Dict to store role -> pool_id mappings
        global_pool_id: The global resource pool ID
    """
    from verl.trainer.ppo.ray_trainer import Role
    
    if not config.reward_model.enable:
        return
    
    strategy = config.reward_model.strategy
    rm_type = config.reward_model.type
    
    def register_role(role, worker_cls):
        role_worker_mapping[role] = ray.remote(worker_cls)
        mapping[role] = global_pool_id
        logger.info(f"Registered {role} with worker {worker_cls.__name__}")
    
    if strategy == "fsdp":
        from .teleboost_fsdp_worker import RewardModelWorker
        register_role(Role.RewardModel, RewardModelWorker)
        
    elif strategy == "megatron":
        from verl.workers.megatron_workers import RewardModelWorker
        register_role(Role.RewardModel, RewardModelWorker)
        
    elif strategy == "diffusion":
        # Special case: Qwen needs vLLM-based worker for distributed inference
        model_name = config.reward_model.get("model_name", rm_type)
        
        if model_name == "qwen" or rm_type == "qwen":
            from .teleboost_fsdp_worker import QwenRewardModelWorker
            register_role(Role.RewardModel, QwenRewardModelWorker)
            logger.info("Using QwenRewardModelWorker for vLLM-based inference")
            
        elif rm_type == "joint":
            # Joint mode: uses ALL_TO_ALL dispatch, models handle their own DP splitting
            from .unified_reward_worker import JointRewardModelWorker
            register_role(Role.RewardModel, JointRewardModelWorker)
            logger.info("Using JointRewardModelWorker for joint mode")
            
        else:
            # Single mode: uses DP_COMPUTE_PROTO, data pre-split by framework
            from .unified_reward_worker import UnifiedRewardModelWorker
            register_role(Role.RewardModel, UnifiedRewardModelWorker)
            logger.info(f"Using UnifiedRewardModelWorker for single mode (type='{rm_type}')")
            
    else:
        raise NotImplementedError(f"Unknown reward model strategy: {strategy}")


def run_ppo(config: DictConfig) -> None:
    """
    Initialize Ray and start the PPO training task.
    
    Args:
        config: Hydra configuration
    """
    _init_ray(config)
    
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    """
    Remote task runner for Dance-GRPO training.
    
    This class runs on a Ray worker (not the head node) to avoid
    resource contention with the driver process.
    """
    
    def run(self, config: DictConfig) -> None:
        """
        Execute the training pipeline.
        
        Args:
            config: Training configuration
        """
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.single_controller.ray import RayWorkerGroup
        
        # Print and resolve configuration
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        
        # Validate configuration
        _validate_config(config)

        # Download model checkpoint
        try:
            local_path = copy_to_local(config.actor_rollout_ref.model.path)
            logger.info(f"Model downloaded to {local_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download model from {config.actor_rollout_ref.model.path}: {e}"
            )

        # Build tokenizer and processor (uses ``tokenizer_subpath`` from
        # the recipe-level ``actor_rollout_ref`` config — see docstring).
        tokenizer, processor = _build_tokenizer_and_processor(
            local_path,
            config.actor_rollout_ref,
        )

        # Resolve worker classes
        ray_worker_group_cls, actor_rollout_worker_cls = _resolve_actor_worker_and_group(config)

        # Setup role-worker mapping
        role_worker_mapping = {}
        global_pool_id = "global_pool"
        
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {}

        def register_role(role, worker_cls):
            role_worker_mapping[role] = ray.remote(worker_cls)
            mapping[role] = global_pool_id

        # Register actor/rollout worker
        register_role(Role.ActorRollout, actor_rollout_worker_cls)

        # Register reward workers
        _register_reward_workers(config, role_worker_mapping, mapping, global_pool_id)

        # Build reward functions
        reward_fn, val_reward_fn = _build_reward_fns(config, tokenizer)

        # Create resource pool manager
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, 
            mapping=mapping
        )

        # Get collate function
        from verl.utils.dataset.rl_dataset import wan_preprocessed_collate_function

        # Create and run trainer
        trainer = RayDanceGRPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            collate_fn=wan_preprocessed_collate_function,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="config", config_name="teleboost_trainer", version_base=None)
def main(config: DictConfig) -> None:
    """
    Main entry point for Dance-GRPO training.
    
    Uses Hydra for configuration management.
    
    Args:
        config: Hydra-loaded configuration
    """
    run_ppo(config)


if __name__ == "__main__":
    main()

# Copyright 2024 Dance-GRPO Team
"""
Dance-GRPO Ray Trainer

This module provides the distributed trainer for Dance-GRPO,
which orchestrates video generation and reward computation across
multiple GPU workers.
"""

import logging
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.metric_utils import compute_timing_metrics, reduce_metrics
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer
from verl.utils.debug import marked_timer

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_VAE_STRIDE = [4, 8, 8]
DEFAULT_LATENT_CHANNELS = 16


def compute_advantage(
    data: DataProto,
    adv_estimator,
    gamma=1.0,
    lam=1.0,
    num_repeat=1,
    multi_turn=False,
    norm_adv_by_std_in_grpo=True,
    config=None,
):
    datas = data.pop(
        batch_keys=["rewards"],
    )
    advantages = torch.zeros_like(datas.batch["rewards"])
    # TODO: when batchsize not equal to 1
    group_mean = datas.batch["rewards"].mean()
    group_std = datas.batch["rewards"].std() + 1e-8
    advantages = (datas.batch["rewards"] - group_mean) / group_std
    data.batch["advantages"] = advantages
    return data


def merge_worker_results(
    data_list: List[DataProto],
    skip_all_zero: bool = True,
    use_validity_flag: bool = True
) -> DataProto:
    """
    Merge results from multiple DataProto instances.

    This function handles the merging of results from different data-parallel
    workers. It provides two strategies for identifying valid data:

    1. Validity flag (preferred): Check for '_valid' meta info
    2. Non-zero check (fallback): Skip tensors that are all zeros

    IMPORTANT: The non-zero check can incorrectly skip valid data where
    all values happen to be zero. Prefer using validity flags when possible.

    Args:
        data_list: List of DataProto from different workers
        skip_all_zero: If True, skip tensors/arrays that are all zeros
        use_validity_flag: If True, prefer validity flags over zero-checking

    Returns:
        Merged DataProto combining valid data from all workers
    """
    if data_list is None:
        return DataProto()
    if isinstance(data_list, DataProto):
        return data_list
    if not data_list:
        return DataProto()
    if len(data_list) == 1:
        return data_list[0]

    # Collect all unique keys
    all_batch_keys = set()
    all_non_tensor_keys = set()

    for dp in data_list:
        if dp.batch is not None:
            all_batch_keys.update(dp.batch.keys())
        if dp.non_tensor_batch is not None:
            all_non_tensor_keys.update(dp.non_tensor_batch.keys())

    def _is_valid_tensor(dp: DataProto, key: str, tensor: torch.Tensor) -> bool:
        """Check if a tensor is valid using multiple strategies."""
        # Strategy 1: Check validity flag in meta_info
        if use_validity_flag and hasattr(dp, 'meta_info') and dp.meta_info:
            validity_key = f'{key}_valid'
            if validity_key in dp.meta_info:
                return dp.meta_info[validity_key]

        # Strategy 2: Check for non-zero values (fallback)
        if skip_all_zero:
            return torch.any(tensor != 0).item()

        return True

    def _is_valid_array(dp: DataProto, key: str, arr: np.ndarray) -> bool:
        """Check if a numpy array is valid using multiple strategies."""
        # Strategy 1: Check validity flag in meta_info
        if use_validity_flag and hasattr(dp, 'meta_info') and dp.meta_info:
            validity_key = f'{key}_valid'
            if validity_key in dp.meta_info:
                return dp.meta_info[validity_key]

        # Strategy 2: Check for non-zero values (fallback)
        # Note: Only apply to numeric arrays
        if skip_all_zero:
            if np.issubdtype(arr.dtype, np.number):
                return np.any(arr != 0)

        return True

    # Process batch tensors
    batch_dict = {}
    for key in all_batch_keys:
        tensors_to_concat = []
        for dp in data_list:
            if dp.batch is not None and key in dp.batch:
                tensor = dp.batch[key]
                if _is_valid_tensor(dp, key, tensor):
                    tensors_to_concat.append(tensor)

        if tensors_to_concat:
            batch_dict[key] = torch.cat(tensors_to_concat, dim=0)

    # Process non_tensor_batch arrays
    non_tensor_dict = {}
    for key in all_non_tensor_keys:
        arrays_to_concat = []
        for dp in data_list:
            if dp.non_tensor_batch is not None and key in dp.non_tensor_batch:
                arr = dp.non_tensor_batch[key]
                if _is_valid_array(dp, key, arr):
                    arrays_to_concat.append(arr)

        if arrays_to_concat:
            non_tensor_dict[key] = np.concatenate(arrays_to_concat, axis=0)

    return DataProto.from_dict(tensors=batch_dict, non_tensors=non_tensor_dict)


class _JointRewardRunner:
    def __init__(self, workers: Dict[str, object]):
        import threading

        self._workers = workers
        self._reward_results = {name: None for name in workers}
        self._thread_inputs = {name: None for name in workers}
        self._ready_events = {name: threading.Event() for name in workers}
        self._done_events = {name: threading.Event() for name in workers}

        for name, worker in workers.items():
            t = threading.Thread(target=self._thread_loop, args=(name, worker), daemon=True)
            t.start()

    def _thread_loop(self, name, worker):
        while True:
            self._ready_events[name].wait()  # 等待主线程喂数据
            self._ready_events[name].clear()
            # 调用现有 worker 的 compute_rm_score
            self._reward_results[name] = worker.compute_rm_score(self._thread_inputs[name])
            self._done_events[name].set()  # 通知主线程完成

    def compute(self, batch: DataProto) -> Dict[str, DataProto]:
        for name in self._workers:
            self._thread_inputs[name] = batch
            self._done_events[name].clear()
            self._ready_events[name].set()

        for name in self._workers:
            self._done_events[name].wait()

        return {name: merge_worker_results(self._reward_results[name]) for name in self._workers}


class RayDanceGRPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf
        from pprint import pprint
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        joint_reward_runner = self._maybe_create_joint_reward_runner()

        for epoch in range(self.config.trainer.total_epochs):
            # ======== 1. 数据 ========
            for batch_dict in self.train_dataloader:  # self.train_dataloader处理数据
                metrics = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                gen_batch = self._build_gen_batch(new_batch)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw):
                        # gen_batch_output的数据类型是DataProto
                        # 具体见DiffusionActorRolloutWorker.generate_sequences方法
                        # 得到的gen_batch_output是聚合所有gpu的结果
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    # 目前用的是 GAE(组间相对优势)，TODO:修改reward计算方法
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("validation", timing_raw):
                            self._save_validation_videos(gen_batch_output)

                    with marked_timer("reward", timing_raw):
                        gen_batch_output = self._compute_rewards(gen_batch_output, metrics, joint_reward_runner)

                    with marked_timer("adv", timing_raw):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        gen_batch_output = compute_advantage(
                            gen_batch_output,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )
                        metrics["train/advantage"] = gen_batch_output.batch["advantages"].mean()

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw):
                            gen_batch_output = self.actor_rollout_wg.update_actor(gen_batch_output)
                        actor_output_metrics = reduce_metrics(gen_batch_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_timing_metrics(batch=new_batch, timing_raw=timing_raw))
                print("metrics", metrics)
                print("=" * 100)

                logger.log(data=metrics, step=self.global_steps)
                timing_raw = defaultdict(float)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1

    def _maybe_create_joint_reward_runner(self):
        """
        Create a joint reward runner if configured.
        
        Note: In the new unified architecture, JointRewardModelWorker handles
        all the reward model management internally. This method returns None
        because the trainer should just call self.rm_wg.compute_rm_score()
        which will handle everything (including aggregation).
        
        For DynamicJointRewardRunner (driver-side computation), we still support
        it but it's deprecated in favor of letting workers handle it.
        
        Returns:
            None - joint mode is now handled by JointRewardModelWorker via rm_wg
        """
        if not self.use_rm or self.config.reward_model.type != "joint":
            return None
        
        # Check if using dynamic joint configuration with driver-side runner
        joint_config = self.config.reward_model.get("joint", None)
        use_driver_side_runner = joint_config.get("driver_side_runner", False) if joint_config else False
        
        if use_driver_side_runner and joint_config and joint_config.get("models"):
            # Use new dynamic joint reward runner (driver-side computation)
            from .reward_models.dynamic_joint import (
                DynamicJointRewardRunner,
                JointRewardConfig
            )
            import torch.distributed as dist
            
            rank = dist.get_rank() if dist.is_initialized() else 0
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            
            config = JointRewardConfig.from_dict(joint_config)
            runner = DynamicJointRewardRunner(config, rank, world_size)
            runner.init_all_models()
            
            logger.info(
                f"Created DynamicJointRewardRunner with {len(runner)} models: "
                f"{runner.list_models()}"
            )
            return runner
        
        # Default: Return None - joint mode is handled by JointRewardModelWorker
        # which is accessed via self.rm_wg.compute_rm_score()
        logger.info(
            "Joint reward mode: using JointRewardModelWorker via rm_wg "
            "(worker-side aggregation)"
        )
        return None

    def _build_gen_batch(self, new_batch: DataProto) -> DataProto:
        # trainer的类型是diffusion
        if self.config.trainer.type == "diffusion":
            gen_batch = new_batch.pop(
                batch_keys=["context", "context_orig_lengths", "null_context"],
                non_tensor_batch_keys=["caption"],
            )
            self._prepare_diffusion_inputs(new_batch, gen_batch)
            return gen_batch.repeat(self.config.actor_rollout_ref.rollout.n)

        if "multi_modal_data" in new_batch.non_tensor_batch.keys():
            return new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )

        return new_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids"],
        )

    def _prepare_diffusion_inputs(self, new_batch: DataProto, gen_batch: DataProto) -> None:
        """
        Prepare latent inputs for diffusion model generation.
        
        This initializes:
        - Random noise in latent space
        - Sigma schedule for the diffusion process
        
        Args:
            new_batch: Source batch with metadata
            gen_batch: Target batch to add diffusion inputs to
        """
        batch_size = new_batch.batch.batch_size[0]
        num_steps = self.config.actor_rollout_ref.sampling_steps
        num_frames = self.config.actor_rollout_ref.num_frames
        size = (self.config.actor_rollout_ref.w, self.config.actor_rollout_ref.h)
        
        # Get VAE stride from config (with default fallback)
        # vae_stride controls the spatial/temporal compression ratio
        vae_stride = self.config.actor_rollout_ref.get(
            "vae_stride", DEFAULT_VAE_STRIDE
        )
        
        # Get latent channels from config (default: 16)
        # This is the number of channels in the VAE latent space
        latent_channels = self.config.actor_rollout_ref.get(
            "latent_channels", DEFAULT_LATENT_CHANNELS
        )
        
        latent_shape = (
            latent_channels,
            (num_frames - 1) // vae_stride[0] + 1,
            size[1] // vae_stride[1],
            size[0] // vae_stride[2],
        )

        # Pre-allocate batch tensors
        input_latents = torch.empty((batch_size, *latent_shape), dtype=torch.float32)
        sigma_schedule_B = torch.empty((batch_size, num_steps + 1), dtype=torch.float32)

        for i in range(batch_size):
            sigma_schedule = torch.linspace(1, 0, num_steps + 1)
            sigma_schedule = self._sd3_time_shift(
                self.config.actor_rollout_ref.shift, sigma_schedule
            )
            sigma_schedule_B[i] = sigma_schedule
            input_latents[i] = torch.randn(latent_shape, dtype=torch.float32)

        gen_batch.batch["input_latents"] = input_latents
        gen_batch.batch["sigma_schedule"] = sigma_schedule_B

    @staticmethod
    def _sd3_time_shift(shift, x):
        return (shift * x) / (1 + (shift - 1) * x)

    def _save_validation_videos(self, gen_batch_output: DataProto) -> None:
        from verl.utils.checkpoint.checkpoint_manager import save_video_and_prompt

        video_frames = gen_batch_output.batch["video_frames"]
        for i in range(video_frames.shape[0]):
            save_video_and_prompt(video_frames[i], 0, i)

    def _compute_rewards(self, gen_batch_output: DataProto, metrics: dict, joint_reward_runner):
        # compute scores. Support both model and function-based.
        # We first compute the scores using reward model. Then, we call reward_fn to combine
        # the results from reward model and rule-based results.
        if self.use_rm:
            print("begin to compute reward")
            with torch.amp.autocast("cuda"):
                # If using driver-side joint reward runner (deprecated)
                if joint_reward_runner is not None:
                    return self._compute_joint_reward(gen_batch_output, metrics, joint_reward_runner)
                
                # Joint mode: parallel computation using multiple worker groups
                if self.config.reward_model.type == "joint":
                    return self._compute_joint_parallel_reward(gen_batch_output, metrics)
                
                # Single/Qwen mode: use single rm_wg
                if self.config.reward_model.type in ("qwen", "single"):
                    return self._compute_single_rm_reward(gen_batch_output, metrics)
                    
                raise ValueError(f"Unsupported reward model type: {self.config.reward_model.type}")

        reward_tensor = self.reward_fn(gen_batch_output, return_dict=True)
        gen_batch_output = gen_batch_output.union(reward_tensor)
        gen_batch_output.pop(batch_keys=["video_frames"])
        return gen_batch_output
    
    def _compute_joint_parallel_reward(self, gen_batch_output: DataProto, metrics: dict) -> DataProto:
        """
        Compute joint rewards using multiple worker groups in parallel.
        
        Each reward model has its own worker group. We use Python threading
        to call them in parallel and aggregate the results.
        """
        import threading
        import time
        from tensordict import TensorDict
        
        start_time = time.time()
        
        # Get model weights from config
        joint_cfg = self.config.reward_model.get("joint", {})
        models_cfg = joint_cfg.get("models", {})
        if isinstance(models_cfg, (list, tuple)):
            weights = {m.get("name"): m.get("weight", 1.0) for m in models_cfg if m.get("name")}
        else:
            weights = {k: v.get("weight", 1.0) for k, v in models_cfg.items()}
        
        # Threading infrastructure
        reward_results = {}
        thread_inputs = {}
        ready_events = {}
        done_events = {}
        
        def thread_loop(name, worker_group):
            while True:
                ready_events[name].wait()
                ready_events[name].clear()
                try:
                    reward_results[name] = worker_group.compute_rm_score(thread_inputs[name])
                except Exception as e:
                    logger.error(f"Error computing {name} reward: {e}")
                    reward_results[name] = None
                done_events[name].set()
        
        # Start threads for each reward model worker group
        threads = []
        for name, wg in self.reward_model_wgs.items():
            reward_results[name] = None
            thread_inputs[name] = None
            ready_events[name] = threading.Event()
            done_events[name] = threading.Event()
            t = threading.Thread(target=thread_loop, args=(name, wg), daemon=True)
            t.start()
            threads.append(t)
        
        # Prepare input for all workers
        reward_input = gen_batch_output.select(
            batch_keys=['video_frames'],
            non_tensor_batch_keys=['caption'],
        )
        
        # Dispatch to all workers
        for name in self.reward_model_wgs:
            thread_inputs[name] = reward_input
            done_events[name].clear()
            ready_events[name].set()
        
        # Wait for all to complete
        for name in self.reward_model_wgs:
            done_events[name].wait()
        
        # Collect and merge results
        batch_with_rewards = gen_batch_output
        combined_reward = None
        
        for name, result in reward_results.items():
            if result is None:
                logger.warning(f"No result from {name} reward model")
                continue
            
            # Find the reward tensor using model-specific key
            reward_key = f"{name}_rewards"
            rewards = None
            
            if reward_key in result.batch.keys():
                rewards = result.batch[reward_key]
            else:
                # Try to find any key containing 'reward'
                for key in result.batch.keys():
                    if 'reward' in key.lower():
                        rewards = result.batch[key]
                        break
            
            if rewards is None:
                logger.warning(f"No reward key found in {name} result, keys: {list(result.batch.keys())}")
                continue
            
            # Add to batch with model-specific name
            batch_with_rewards.batch[reward_key] = rewards
            
            # Add to combined reward
            weight = weights.get(name, 1.0)
            weighted_reward = rewards * weight
            if combined_reward is None:
                combined_reward = weighted_reward
            else:
                combined_reward = combined_reward + weighted_reward
            
            # Log individual metrics
            metrics[f"train/rewards_{name}"] = rewards.mean().item()
        
        if combined_reward is None:
            raise RuntimeError("No valid rewards computed from any model")
        
        # Add aggregated reward directly to batch
        batch_with_rewards.batch["rewards"] = combined_reward
        
        # Log overall metrics
        metrics["train/rewards"] = combined_reward.mean().item()
        if "log_probs" in batch_with_rewards.batch.keys():
            metrics["train/log_probs"] = batch_with_rewards.batch["log_probs"].mean().item()
        
        elapsed = time.time() - start_time
        logger.info(f"Joint parallel reward computation took {elapsed:.2f}s")
        
        return batch_with_rewards

    def _compute_joint_reward(
        self, 
        gen_batch_output: DataProto, 
        metrics: dict, 
        joint_reward_runner
    ) -> DataProto:
        """
        Compute weighted combination of multiple reward signals.
        
        Supports two runner types:
        1. DynamicJointRewardRunner: Uses compute_and_aggregate() for full processing
        2. Legacy _JointRewardRunner: Uses manual aggregation with fixed 4 models
        
        Args:
            gen_batch_output: Generated video batch
            metrics: Dict to store metrics
            joint_reward_runner: Runner managing parallel reward computation
            
        Returns:
            DataProto with combined rewards
        """
        from tensordict import TensorDict
        from .reward_models.dynamic_joint import DynamicJointRewardRunner
        
        start_time = time.time()
        
        # Check if using dynamic runner (has compute_and_aggregate method)
        if isinstance(joint_reward_runner, DynamicJointRewardRunner):
            # Use the new dynamic runner which handles everything internally
            final_batch = joint_reward_runner.compute_and_aggregate(gen_batch_output)
            
            # Add metrics from the dynamic runner
            rewards_dict = joint_reward_runner.compute(gen_batch_output)
            metrics.update(joint_reward_runner.get_metrics(rewards_dict))
            metrics["train/rewards"] = final_batch.batch["rewards"].mean().item()
            
            if "log_probs" in gen_batch_output.batch.keys():
                final_batch = gen_batch_output.union(final_batch)
                metrics["train/log_probs"] = final_batch.batch["log_probs"].mean().item()
            
            elapsed = time.time() - start_time
            logger.info(f"Dynamic joint reward computation took {elapsed:.2f}s")
            return final_batch
        
        # Legacy mode: manual aggregation with fixed 4 models
        rewards = joint_reward_runner.compute(gen_batch_output)
        aes_tensor = rewards.get("aes")
        raft_tensor = rewards.get("raft")
        videoclip_tensor = rewards.get("videoclip")
        videophy_tensor = rewards.get("videophy")

        batch_with_rewards = gen_batch_output
        if aes_tensor is not None:
            batch_with_rewards = batch_with_rewards.union(aes_tensor)
        if raft_tensor is not None:
            batch_with_rewards = batch_with_rewards.union(raft_tensor)
        if videoclip_tensor is not None:
            batch_with_rewards = batch_with_rewards.union(videoclip_tensor)
        if videophy_tensor is not None:
            batch_with_rewards = batch_with_rewards.union(videophy_tensor)

        # Get configurable weights (with sensible defaults)
        weights_config = self.config.reward_model.get("weights", {})
        w_aes = weights_config.get("aes", 1.0)
        w_raft = weights_config.get("raft", 1.0)
        w_videoclip = weights_config.get("videoclip", 1.0)
        w_videophy = weights_config.get("videophy", 1.0)
        
        logger.debug(
            f"Reward weights: aes={w_aes}, raft={w_raft}, "
            f"videoclip={w_videoclip}, videophy={w_videophy}"
        )

        # Compute weighted sum of rewards (handle missing rewards gracefully)
        combined_reward = torch.zeros_like(batch_with_rewards.batch.get(
            "aes_rewards", 
            batch_with_rewards.batch.get("raft_rewards")
        ))
        
        if "aes_rewards" in batch_with_rewards.batch.keys():
            combined_reward = combined_reward + w_aes * batch_with_rewards.batch["aes_rewards"]
            metrics["train/rewards_aes"] = batch_with_rewards.batch["aes_rewards"].mean().item()
        if "raft_rewards" in batch_with_rewards.batch.keys():
            combined_reward = combined_reward + w_raft * batch_with_rewards.batch["raft_rewards"]
            metrics["train/rewards_raft"] = batch_with_rewards.batch["raft_rewards"].mean().item()
        if "videoclip_rewards" in batch_with_rewards.batch.keys():
            combined_reward = combined_reward + w_videoclip * batch_with_rewards.batch["videoclip_rewards"]
            metrics["train/rewards_videoclip"] = batch_with_rewards.batch["videoclip_rewards"].mean().item()
        if "videophy_rewards" in batch_with_rewards.batch.keys():
            combined_reward = combined_reward + w_videophy * batch_with_rewards.batch["videophy_rewards"]
            metrics["train/rewards_videophy"] = batch_with_rewards.batch["videophy_rewards"].mean().item()

        # Build final result with combined reward
        reward_td = TensorDict(
            {"rewards": combined_reward}, 
            batch_size=combined_reward.shape[0]
        )
        
        # Get non_tensor_batch from first available tensor
        non_tensor = {}
        for tensor in [aes_tensor, raft_tensor, videoclip_tensor, videophy_tensor]:
            if tensor is not None and tensor.non_tensor_batch:
                non_tensor = tensor.non_tensor_batch
                break
        
        reward_proto = DataProto(batch=reward_td, non_tensor_batch=non_tensor)
        final_batch = batch_with_rewards.union(reward_proto)

        # Log metrics
        metrics["train/rewards"] = combined_reward.mean().item()
        if "log_probs" in final_batch.batch.keys():
            metrics["train/log_probs"] = final_batch.batch["log_probs"].mean().item()

        elapsed = time.time() - start_time
        logger.info(f"Legacy joint reward computation took {elapsed:.2f}s")
        
        return final_batch

    def _compute_single_rm_reward(self, gen_batch_output: DataProto, metrics: dict):
        if self.config.reward_model.type == "qwen":
            reward_input = gen_batch_output.select(
                batch_keys=["null_context"],
                non_tensor_batch_keys=["caption", "video_ids"],
            )
            reward_tensor = self.rm_wg.compute_rm_score(reward_input)
            reward_tensor.pop(non_tensor_batch_keys=["caption", "video_ids"])
        
        else:  # "single"
            reward_input = gen_batch_output.select(
                batch_keys=["video_frames"],
                non_tensor_batch_keys=["caption"],
            )
            reward_tensor = self.rm_wg.compute_rm_score(reward_input)

        # [smoke-patch] defensively pop only keys that exist (original wxe code assumed all 3 present)
        _keys_to_pop = [k for k in ("caption", "video_ids", "video_frames")
                        if k in gen_batch_output.non_tensor_batch]
        if _keys_to_pop:
            gen_batch_output.pop(non_tensor_batch_keys=_keys_to_pop)  # reward计算完就可以丢掉大tensor了
        # 清理原 batch（移除大 tensor）

        self._debug_proto_batch("gen_batch_output", gen_batch_output)
        self._debug_proto_batch("reward_tensor", reward_tensor)
        gen_batch_output = gen_batch_output.union(reward_tensor)

        # [smoke-patch] reward key 在 single 模式下是 "{model_name}_rewards" (e.g. hps_rewards),
        # 而下游 (compute_advantage / metrics) 硬编码读 "rewards".
        # 把 *_rewards 复制为 "rewards" 别名, 不删除源 key 以兼容 joint 等其他引用.
        if "rewards" not in gen_batch_output.batch:
            _src_key = next((k for k in gen_batch_output.batch.keys() if k.endswith("_rewards")), None)
            if _src_key is None:
                raise KeyError(f"no '*rewards' key in gen_batch_output.batch (keys={list(gen_batch_output.batch.keys())})")
            gen_batch_output.batch["rewards"] = gen_batch_output.batch[_src_key]
        metrics["train/rewards"] = gen_batch_output.batch["rewards"].mean()
        metrics["train/log_probs"] = gen_batch_output.batch["log_probs"].mean()
        return gen_batch_output

    @staticmethod
    def _debug_proto_batch(name, proto):
        if proto is None:
            print(f"[debug] {name} is None")
            return
        batch = getattr(proto, "batch", None)
        if batch is None:
            non_tensor = getattr(proto, "non_tensor_batch", None) or {}
            print(f"[debug] {name}.batch is None; non_tensor_keys={list(non_tensor.keys())}")
            return
        print(f"[debug] {name}.batch_size={batch.batch_size}")

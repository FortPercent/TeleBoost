# Copyright 2024 Dance-GRPO Team
"""
Dance-GRPO Ray Trainer

This module provides the distributed trainer for Dance-GRPO,
which orchestrates video generation and reward computation across
multiple GPU workers.
"""

import logging
import uuid
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.metric_utils import reduce_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.debug import marked_timer

from recipe.dancegrpo.algorithms import (
    BGPOMixin,
    JointRewardMixin,
    VIPOMixin,
)

logger = logging.getLogger(__name__)


def _save_video_and_prompt(video_frames: torch.Tensor, rank: int, index: int) -> None:
    """Write a (C, T, H, W) tensor to ./videos/output/wan_video_batch_<ts>_<index>.mp4.

    Pre-X3 lived as `verl.utils.checkpoint.checkpoint_manager.save_video_and_prompt`
    in the in-tree fork; moved here since it's purely a recipe-level validation
    preview helper and has no place in upstream verl.
    """
    from datetime import datetime
    import cv2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    assert video_frames.dim() == 4
    C, T, H, W = video_frames.shape
    video_np = video_frames.permute(1, 2, 3, 0).cpu().numpy()
    video_np = (video_np * 255).astype(np.uint8)
    video_filename = f"wan_video_batch_{timestamp}_{index}.mp4"
    import os as _os
    video_path = _os.path.join("./videos/output", video_filename)
    _os.makedirs("videos/output", exist_ok=True)
    out = cv2.VideoWriter(
        video_path,
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=video_np.shape[0],
        frameSize=(W, H),
    )
    for t in range(T):
        frame = video_np[t]
        if C == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        out.write(frame_bgr)
    out.release()


def _compute_timing_metrics_diffusion(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    """Diffusion-friendly timing metrics.

    Upstream verl 0.4.0 `compute_timing_metrics` requires `batch["responses"]`
    (LM token shape) to derive per-token throughput. Diffusion batches don't
    carry that — they have latents. Pre-X3's in-tree fork commented out the
    per-token block and emitted only raw `timing_s/{name}` entries; mirror
    that here.
    """
    return {f"timing_s/{name}": value for name, value in timing_raw.items()}


compute_timing_metrics = _compute_timing_metrics_diffusion

# Default configuration values
DEFAULT_VAE_STRIDE = [4, 8, 8]
DEFAULT_LATENT_CHANNELS = 16


def compute_advantage(
    data: DataProto,
    gamma=1.0,
    lam=1.0,
    num_repeat=1,
    multi_turn=False,
    norm_adv_by_std_in_grpo=True,
    config=None,
):
    """Group-relative GRPO advantage = (rewards - group_mean) / group_std.

    The verl-side multi-estimator switch (GRPO / GAE / REMAX / etc.) was
    replaced by this GRPO-only inline body in commit 33f9b00c (wxe X3
    series).  ReMax / GAE are not supported here — adding either back
    needs the verl-side switch reinstated, not just an `if` branch.
    """
    rewards = data.batch["rewards"]
    advantages = torch.zeros_like(rewards)
    # TODO: when batchsize not equal to 1
    group_mean = rewards.mean()
    group_std = rewards.std() + 1e-8
    advantages = (rewards - group_mean) / group_std
    data.batch["advantages"] = advantages
    return data



class RayDanceGRPOTrainer(BGPOMixin, VIPOMixin, JointRewardMixin, RayPPOTrainer):
    """Driver-side Dance-GRPO trainer.

    Algorithm hooks come from mixins in :mod:`recipe.dancegrpo.algorithms`:

    * :class:`BGPOMixin` — reward rerange (CRT) + adaptive advantage scaling (RAS)
    * :class:`VIPOMixin` — pixel-weighted dense advantage broadcast
    * :class:`JointRewardMixin` — multi-head joint reward (worker-parallel,
      driver-side dynamic, legacy fixed-4 runners)

    Mixins are no-ops unless their enable flags are set in the Hydra
    config.  See ``recipe/dancegrpo/algorithms/README.md`` for the
    feature-flag matrix.
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
        # Running accumulator used by the CVPR-std adaptive-weight mode.
        self._std_running_accumulation = 0.0

        for epoch in range(self.config.trainer.total_epochs):
            # ======== 1. Data ========
            for batch_dict in self.train_dataloader:
                metrics = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                gen_batch = self._build_gen_batch(new_batch)

                is_last_step = self.global_steps >= self.total_training_steps
                # When joint precompute path fires we must skip the
                # default reward + advantage blocks below (they would recompute).
                joint_adv_precomputed = False

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw):
                        # gen_batch_output is a DataProto aggregated across all GPUs.
                        # See DiffusionActorRolloutRefWorker.generate_sequences.
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

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

                    # Joint mode: pre-compute per-reward advantages + joint weights
                    # BEFORE the reward timer so the downstream block can skip.
                    if self.use_rm and self.config.reward_model.type == "joint":
                        gen_batch_output, joint_adv_precomputed = self._precompute_joint_advantages(
                            gen_batch_output,
                            gen_batch,
                            metrics,
                            joint_reward_runner,
                        )

                    with marked_timer("reward", timing_raw):
                        if not joint_adv_precomputed:
                            gen_batch_output = self._compute_rewards(
                                gen_batch_output, metrics, joint_reward_runner, gen_batch
                            )

                    with marked_timer("adv", timing_raw):
                        if not joint_adv_precomputed:
                            # compute advantages, executed on the driver process
                            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                            gen_batch_output = compute_advantage(
                                gen_batch_output,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            )
                        # Scale advantages before optional VIPO broadcasting.
                        if self._is_bgpo_enabled():
                            gen_batch_output = self._apply_bgpo_on_advantages(
                                gen_batch_output, gen_batch, metrics
                            )
                        # Broadcast scalar advantages to dense pixel-weighted maps.
                        if self._is_pixel_weight_enabled():
                            gen_batch_output = self._apply_vipo_broadcast(
                                gen_batch_output, metrics
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

    def _build_gen_batch(self, new_batch: DataProto) -> DataProto:
        # trainer.type == 'diffusion'
        if self.config.trainer.type == "diffusion":
            non_tensor_keys = ["caption"]
            for optional_key in ("prior", "index_prompt", "id"):
                if optional_key in new_batch.non_tensor_batch:
                    non_tensor_keys.append(optional_key)
            gen_batch = new_batch.pop(
                batch_keys=["context", "context_orig_lengths", "null_context"],
                non_tensor_batch_keys=non_tensor_keys,
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
        video_frames = gen_batch_output.batch["video_frames"]
        for i in range(video_frames.shape[0]):
            _save_video_and_prompt(video_frames[i], 0, i)

    def _compute_rewards(self, gen_batch_output: DataProto, metrics: dict, joint_reward_runner, source_batch: DataProto = None):
        # compute scores. Support both model and function-based.
        # We first compute the scores using reward model. Then, we call reward_fn to combine
        # the results from reward model and rule-based results.
        #
        bgpo_enabled = self._is_bgpo_enabled() and source_batch is not None

        if self.use_rm:
            logger.debug("Computing reward")
            with torch.amp.autocast("cuda"):
                # If using driver-side joint reward runner (deprecated)
                if joint_reward_runner is not None:
                    reward_output = self._compute_joint_reward(gen_batch_output, metrics, joint_reward_runner)
                    if bgpo_enabled:
                        return self._apply_bgpo_on_rewards(reward_output, source_batch, metrics)
                    return reward_output

                # Joint mode: parallel computation using multiple worker groups
                if self.config.reward_model.type == "joint":
                    reward_output = self._compute_joint_parallel_reward(gen_batch_output, metrics)
                    if bgpo_enabled:
                        return self._apply_bgpo_on_rewards(reward_output, source_batch, metrics)
                    return reward_output

                # Single/Qwen mode: use single rm_wg
                if self.config.reward_model.type in ("qwen", "single"):
                    reward_output = self._compute_single_rm_reward(gen_batch_output, metrics)
                    if bgpo_enabled:
                        return self._apply_bgpo_on_rewards(reward_output, source_batch, metrics)
                    return reward_output

                raise ValueError(f"Unsupported reward model type: {self.config.reward_model.type}")

        reward_tensor = self.reward_fn(gen_batch_output, return_dict=True)
        gen_batch_output = gen_batch_output.union(reward_tensor)
        gen_batch_output.pop(batch_keys=["video_frames"])
        if bgpo_enabled:
            return self._apply_bgpo_on_rewards(gen_batch_output, source_batch, metrics)
        return gen_batch_output
    
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

        _keys_to_pop = [k for k in ("caption", "video_ids", "video_frames")
                        if k in gen_batch_output.non_tensor_batch]
        if _keys_to_pop:
            gen_batch_output.pop(non_tensor_batch_keys=_keys_to_pop)

        self._debug_proto_batch("gen_batch_output", gen_batch_output)
        self._debug_proto_batch("reward_tensor", reward_tensor)
        gen_batch_output = gen_batch_output.union(reward_tensor)

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
            logger.debug("%s is None", name)
            return
        batch = getattr(proto, "batch", None)
        if batch is None:
            non_tensor = getattr(proto, "non_tensor_batch", None) or {}
            logger.debug("%s.batch is None; non_tensor_keys=%s", name, list(non_tensor.keys()))
            return
        logger.debug("%s.batch_size=%s", name, batch.batch_size)


import logging
import uuid
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.metric_utils import compute_timing_metrics, reduce_metrics
from verl.trainer.ppo.ray_trainer import AdvantageEstimator
from verl.utils.debug import marked_timer

from .dancegrpo_ray_trainer import RayDanceGRPOTrainer

logger = logging.getLogger(__name__)


def compute_advantage_pixel(
    data: DataProto,
    adv_estimator,
    gamma=1.0,
    lam=1.0,
    num_repeat=1,
    multi_turn=False,
    norm_adv_by_std_in_grpo=True,
    config=None,
):
    datas = data.pop(batch_keys=["rewards", "pixel_weight_maps"])
    rewards = datas.batch["rewards"].to(torch.float32).reshape(-1)
    pixel_weight_maps = datas.batch["pixel_weight_maps"].to(torch.float32)

    scalar_advantages = torch.zeros_like(rewards)
    use_group = num_repeat is not None and num_repeat > 1 and rewards.numel() % num_repeat == 0
    if use_group:
        for start_idx in range(0, rewards.numel(), num_repeat):
            end_idx = start_idx + num_repeat
            group_rewards = rewards[start_idx:end_idx]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            scalar_advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
    else:
        scalar_advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    dense_advantages = scalar_advantages.view(-1, 1, 1, 1) * pixel_weight_maps
    data.batch["advantages"] = dense_advantages
    return data


class RayDanceGRPOTrainerPixel(RayDanceGRPOTrainer):
    def fit(self):
        from omegaconf import OmegaConf
        from pprint import pprint
        from verl.utils.tracking import Tracking

        tracker = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        joint_reward_runner = self._maybe_create_joint_reward_runner()

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                gen_batch = self._build_gen_batch(new_batch)
                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

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
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))],
                        dtype=object,
                    )

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
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        gen_batch_output = compute_advantage_pixel(
                            gen_batch_output,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config,
                        )
                        metrics["train/advantage"] = gen_batch_output.batch["advantages"].mean()

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw):
                            gen_batch_output = self.actor_rollout_wg.update_actor(gen_batch_output)
                        actor_output_metrics = reduce_metrics(gen_batch_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update(compute_timing_metrics(batch=new_batch, timing_raw=timing_raw))
                tracker.log(data=metrics, step=self.global_steps)
                timing_raw = defaultdict(float)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1

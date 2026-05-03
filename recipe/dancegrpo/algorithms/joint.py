"""Joint reward: weighted combination over multiple reward heads.

Active when ``reward_model.type=joint``.  The mixin orchestrates three
parallel-execution paths:

* ``DynamicJointRewardRunner`` (driver-side, dynamic model list) — used
  when ``reward_model.joint.driver_side_runner=true``.
* ``_JointRewardRunner`` (driver-side, legacy fixed 4 models) —
  aesthetic + raft + videoclip + videophy.
* ``JointRewardModelWorker`` (worker-side, parallel worker groups) —
  the default path; threads dispatch to ``self.reward_model_wgs``.

When BGPO is also enabled the joint path pre-computes per-task
advantages and a convex weight matrix (via
:func:`compute_joint_task_weights`) before applying BGPO's reward
rerange. The advantage path then re-derives a scalar advantage from the
post-rerange rewards.

Pure helpers (``_JointRewardRunner``, ``merge_worker_results``) live at
module level. Trainer hooks live on :class:`JointRewardMixin`.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from verl import DataProto

from recipe.dancegrpo.algorithms.grpo_advantage import per_prompt_zscore_advantage
from recipe.dancegrpo.algorithms.multi_reward_aggregation import compute_joint_task_weights

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def merge_worker_results(
    data_list: List[DataProto],
    skip_all_zero: bool = True,
    use_validity_flag: bool = True,
) -> DataProto:
    """Merge results from multiple data-parallel workers.

    Two strategies for identifying valid data:

    1. Validity flag (preferred): check for ``<key>_valid`` in meta_info.
    2. Non-zero check (fallback): skip tensors/arrays that are all zeros.

    The non-zero check can incorrectly skip data where every value is 0;
    prefer validity flags when possible.
    """
    if data_list is None:
        return DataProto()
    if isinstance(data_list, DataProto):
        return data_list
    if not data_list:
        return DataProto()
    if len(data_list) == 1:
        return data_list[0]

    all_batch_keys = set()
    all_non_tensor_keys = set()

    for dp in data_list:
        if dp.batch is not None:
            all_batch_keys.update(dp.batch.keys())
        if dp.non_tensor_batch is not None:
            all_non_tensor_keys.update(dp.non_tensor_batch.keys())

    def _is_valid_tensor(dp: DataProto, key: str, tensor: torch.Tensor) -> bool:
        if use_validity_flag and hasattr(dp, "meta_info") and dp.meta_info:
            validity_key = f"{key}_valid"
            if validity_key in dp.meta_info:
                return dp.meta_info[validity_key]
        if skip_all_zero:
            return torch.any(tensor != 0).item()
        return True

    def _is_valid_array(dp: DataProto, key: str, arr: np.ndarray) -> bool:
        if use_validity_flag and hasattr(dp, "meta_info") and dp.meta_info:
            validity_key = f"{key}_valid"
            if validity_key in dp.meta_info:
                return dp.meta_info[validity_key]
        if skip_all_zero:
            if np.issubdtype(arr.dtype, np.number):
                return np.any(arr != 0)
        return True

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
    """Legacy fixed-model joint runner (aes / raft / videoclip / videophy).

    Spawns one daemon thread per worker; ``compute()`` fans the batch out
    and waits for all workers to finish.

    Newer rollouts should use ``DynamicJointRewardRunner`` (configurable
    model list via ``reward_model.joint.models``); this class is kept
    only for backwards compatibility with checkpoints whose configs
    still use the legacy 4-model layout.
    """

    def __init__(self, workers: Dict[str, object]):
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
            self._ready_events[name].wait()  # wait for main thread to push data
            self._ready_events[name].clear()
            self._reward_results[name] = worker.compute_rm_score(self._thread_inputs[name])
            self._done_events[name].set()

    def compute(self, batch: DataProto) -> Dict[str, DataProto]:
        for name in self._workers:
            self._thread_inputs[name] = batch
            self._done_events[name].clear()
            self._ready_events[name].set()

        for name in self._workers:
            self._done_events[name].wait()

        return {name: merge_worker_results(self._reward_results[name]) for name in self._workers}


# ---------------------------------------------------------------------------
# Trainer mixin
# ---------------------------------------------------------------------------


class JointRewardMixin:
    """Trainer mixin for the ``reward_model.type=joint`` path.

    Three execution modes, selected at runtime:

    * Driver-side dynamic runner (``DynamicJointRewardRunner``) when
      ``reward_model.joint.driver_side_runner=true``.
    * Worker-side parallel groups (``self.reward_model_wgs``) — default.
    * Legacy fixed 4-model driver-side runner (``_JointRewardRunner``).

    All three paths converge on ``rewards`` + per-task ``<name>_rewards``
    keys in the returned ``DataProto``.

    The mixin assumes :class:`BGPOMixin` is also mixed in so that
    ``_apply_bgpo_on_rewards`` and friends are available when joint mode
    is composed with BGPO.
    """

    # -- runner selection ---------------------------------------------------

    def _maybe_create_joint_reward_runner(self):
        """Create a driver-side joint runner if configured.

        Returns ``None`` when joint mode is handled by ``rm_wg`` directly
        (the modern ``JointRewardModelWorker`` path).
        """
        if not self.use_rm or self.config.reward_model.type != "joint":
            return None

        joint_config = self.config.reward_model.get("joint", None)
        use_driver_side_runner = (
            joint_config.get("driver_side_runner", False) if joint_config else False
        )

        if use_driver_side_runner and joint_config and joint_config.get("models"):
            from recipe.dancegrpo.reward_models.dynamic_joint import (
                DynamicJointRewardRunner,
                JointRewardConfig,
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

        logger.info(
            "Joint reward mode: using JointRewardModelWorker via rm_wg "
            "(worker-side aggregation)"
        )
        return None

    # -- worker-side parallel path -----------------------------------------

    def _compute_joint_parallel_reward(
        self,
        gen_batch_output: DataProto,
        metrics: dict,
    ) -> DataProto:
        """Joint rewards via the unified ``JointRewardModelWorker`` worker group.

        The worker (registered as ``Role.RewardModel`` in ``main_dancegrpo.py``)
        owns all per-task models and emits ``<name>_rewards`` keys plus a
        weighted-aggregate ``rewards`` key in a single ``compute_rm_score``
        call.  Earlier revisions exposed one worker group per task on
        ``self.reward_model_wgs``, but that attribute is no longer
        populated; that code path is deleted to avoid silent
        AttributeError at runtime.
        """
        start_time = time.time()

        reward_input = gen_batch_output.select(
            batch_keys=["video_frames"],
            non_tensor_batch_keys=["caption"],
        )

        # ``JointRewardModelWorker.compute_rm_score`` is registered with
        # ``Dispatch.ALL_TO_ALL``: every worker receives the full batch
        # and returns its own DataProto.  The worker performs an
        # AllGather internally so each rank's returned proto is already
        # the merged view, and the dispatcher hands us a list with one
        # equivalent entry per rank.  Concat-merging would inflate the
        # batch dim, so we just take the first.
        raw = self.rm_wg.compute_rm_score(reward_input)
        if isinstance(raw, list):
            if not raw:
                raise RuntimeError("rm_wg.compute_rm_score returned an empty list")
            result = raw[0]
        else:
            result = raw

        batch_with_rewards = gen_batch_output
        combined_reward = None

        joint_cfg = self.config.reward_model.get("joint", {})
        models_cfg = joint_cfg.get("models", {})
        if isinstance(models_cfg, (list, tuple)):
            weights = {m.get("name"): m.get("weight", 1.0) for m in models_cfg if m.get("name")}
        else:
            weights = {k: v.get("weight", 1.0) for k, v in models_cfg.items()}

        # Pull per-task ``<name>_rewards`` keys back into the batch and
        # build the weighted sum from them.  If the worker already
        # supplied an aggregated ``rewards`` key we still recompute here
        # so the metrics' weighted view matches the BGPO precompute path.
        for key in result.batch.keys():
            if not key.endswith("_rewards"):
                continue
            name = key[:-len("_rewards")]
            if name == "":  # the bare aggregated ``rewards`` key
                continue

            rewards = result.batch[key]
            batch_with_rewards.batch[key] = rewards

            weight = weights.get(name, 1.0)
            weighted_reward = rewards * weight
            if combined_reward is None:
                combined_reward = weighted_reward
            else:
                combined_reward = combined_reward + weighted_reward

            metrics[f"train/rewards_{name}"] = rewards.mean().item()

        # Fall back to whatever aggregate the worker already produced if
        # no per-task keys came through (e.g. single-model joint config).
        if combined_reward is None and "rewards" in result.batch:
            combined_reward = result.batch["rewards"]
            batch_with_rewards.batch["rewards"] = combined_reward

        if combined_reward is None:
            raise RuntimeError("No valid rewards computed from any model")

        batch_with_rewards.batch["rewards"] = combined_reward

        metrics["train/rewards"] = combined_reward.mean().item()
        if "log_probs" in batch_with_rewards.batch.keys():
            metrics["train/log_probs"] = batch_with_rewards.batch["log_probs"].mean().item()

        elapsed = time.time() - start_time
        logger.info(f"Joint parallel reward computation took {elapsed:.2f}s")

        return batch_with_rewards

    # -- driver-side runner path -------------------------------------------

    def _compute_joint_reward(
        self,
        gen_batch_output: DataProto,
        metrics: dict,
        joint_reward_runner,
    ) -> DataProto:
        """Joint reward via a driver-side runner.

        Two runner shapes are supported:

        * :class:`DynamicJointRewardRunner` — exposes
          ``compute_and_aggregate``; we delegate everything to it.
        * Legacy :class:`_JointRewardRunner` — fixed 4 models
          (aes/raft/videoclip/videophy); we manually weighted-sum.
        """
        from tensordict import TensorDict
        from recipe.dancegrpo.reward_models.dynamic_joint import DynamicJointRewardRunner

        start_time = time.time()

        if isinstance(joint_reward_runner, DynamicJointRewardRunner):
            final_batch = joint_reward_runner.compute_and_aggregate(gen_batch_output)

            rewards_dict = joint_reward_runner.compute(gen_batch_output)
            metrics.update(joint_reward_runner.get_metrics(rewards_dict))
            metrics["train/rewards"] = final_batch.batch["rewards"].mean().item()

            if "log_probs" in gen_batch_output.batch.keys():
                final_batch = gen_batch_output.union(final_batch)
                metrics["train/log_probs"] = final_batch.batch["log_probs"].mean().item()

            elapsed = time.time() - start_time
            logger.info(f"Dynamic joint reward computation took {elapsed:.2f}s")
            return final_batch

        # Legacy fixed-4 path
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

        weights_config = self.config.reward_model.get("weights", {})
        w_aes = weights_config.get("aes", 1.0)
        w_raft = weights_config.get("raft", 1.0)
        w_videoclip = weights_config.get("videoclip", 1.0)
        w_videophy = weights_config.get("videophy", 1.0)

        logger.debug(
            f"Reward weights: aes={w_aes}, raft={w_raft}, "
            f"videoclip={w_videoclip}, videophy={w_videophy}"
        )

        combined_reward = torch.zeros_like(
            batch_with_rewards.batch.get(
                "aes_rewards",
                batch_with_rewards.batch.get("raft_rewards"),
            )
        )

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

        reward_td = TensorDict(
            {"rewards": combined_reward},
            batch_size=combined_reward.shape[0],
        )

        non_tensor = {}
        for tensor in [aes_tensor, raft_tensor, videoclip_tensor, videophy_tensor]:
            if tensor is not None and tensor.non_tensor_batch:
                non_tensor = tensor.non_tensor_batch
                break

        reward_proto = DataProto(batch=reward_td, non_tensor_batch=non_tensor)
        final_batch = batch_with_rewards.union(reward_proto)

        metrics["train/rewards"] = combined_reward.mean().item()
        if "log_probs" in final_batch.batch.keys():
            metrics["train/log_probs"] = final_batch.batch["log_probs"].mean().item()

        elapsed = time.time() - start_time
        logger.info(f"Legacy joint reward computation took {elapsed:.2f}s")

        return final_batch

    # -- joint + BGPO precompute -------------------------------------------

    def _precompute_joint_advantages(
        self,
        gen_batch_output: DataProto,
        source_batch: DataProto,
        metrics: dict,
        joint_reward_runner,
    ) -> Tuple[DataProto, bool]:
        """Pre-compute joint-mode advantages + per-task convex weights.

        Runs BEFORE the standard reward+advantage block so the caller
        can skip the duplicated computation.  When BGPO is also enabled
        the rewards are reranged here and the advantage is re-derived
        from the post-rerange rewards.

        Returns
        -------
        (DataProto, bool)
            Updated batch, and a flag indicating whether the multi-head
            precompute fired (False -> caller must run the standard path).
        """
        if joint_reward_runner is not None:
            reward_output = self._compute_joint_reward(
                gen_batch_output, metrics, joint_reward_runner
            )
        else:
            reward_output = self._compute_joint_parallel_reward(gen_batch_output, metrics)

        reward_keys = [k for k in reward_output.batch.keys() if k.endswith("_rewards")]
        if not reward_keys:
            return reward_output, False

        per_task_rewards = [reward_output.batch[k].float() for k in reward_keys]
        per_task_advantages = []

        # Per-prompt grouping for the per-task z-score.  Same paper
        # reasoning as the single-reward path in
        # ``dancegrpo_ray_trainer.compute_advantage`` (GRPO arxiv
        # 2402.03300 §4.1.2 + DanceGRPO Eq. 10).
        num_repeat = int(self.config.actor_rollout_ref.rollout.n)

        for reward_key in reward_keys:
            reward_tensor = reward_output.batch[reward_key].float()
            adv = per_prompt_zscore_advantage(reward_tensor, num_repeat)
            task_name = reward_key[:-8]
            reward_output.batch[f"{task_name}_advantages"] = adv
            metrics[f"train/{task_name}_advantages"] = adv.mean().item()
            per_task_advantages.append(adv)

        adv_matrix = torch.stack(per_task_advantages, dim=-1)
        weight_matrix = compute_joint_task_weights(adv_matrix)

        reward_output.batch["task_weights"] = weight_matrix
        reward_output.batch["rewards"] = (weight_matrix * torch.stack(per_task_rewards, dim=-1)).sum(dim=-1)
        reward_output.batch["advantages"] = (weight_matrix * adv_matrix).sum(dim=-1)

        if self._is_bgpo_enabled():
            reward_output = self._apply_bgpo_on_rewards(reward_output, source_batch, metrics)
            if self._get_bgpo_config().get("use_rerange", False):
                # Re-derive advantage with the same per-prompt grouping
                # as above (post-rerange rewards).
                rewards = reward_output.batch["rewards"].float()
                reward_output.batch["advantages"] = per_prompt_zscore_advantage(rewards, num_repeat)

        for idx, reward_key in enumerate(reward_keys):
            task_name = reward_key[:-8]
            metrics[f"train/task_weight_{task_name}"] = weight_matrix[:, idx].mean().item()

        metrics["train/rewards"] = reward_output.batch["rewards"].mean().item()
        return reward_output, True


__all__ = [
    "JointRewardMixin",
    "_JointRewardRunner",
    "merge_worker_results",
]

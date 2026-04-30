"""Whole-module overrides over upstream verl.

For files where the project diverges from upstream verl@v0.4.0, we mirror
them under teleboost/_overrides/verl/<same-path> and replace the upstream
module via sys.modules at patch-apply time.

NOTE: order is important - apply() walks the OVERRIDES list, so dependencies
should appear before their dependents (or at least be tolerant of being
imported with the upstream version still in sys.modules - usually fine
since the override modules tend to import siblings via fully-qualified
verl.X paths and sys.modules is mutated synchronously between iterations).
"""
from __future__ import annotations

import importlib
import sys

# verl module path -> teleboost override module path
# Auto-derived from teleboost/_overrides/verl/**, then hand-curated for ordering.
OVERRIDES = {
    # Module aliases (already-mv'd-in-stage-1) so monkey_patch's relative
    # `from .wan import ulysses_self_flash_attn_forward` resolves.
    "verl.models.transformers.wan": "teleboost.models.transformers.wan",
    "verl.models.transformers.wan22": "teleboost.models.transformers.wan22",
    # Project-modified modules - whole-module replacements.
    "verl.utils.fs": "teleboost._overrides.verl.utils.fs",
    "verl.utils.py_functional": "teleboost._overrides.verl.utils.py_functional",
    "verl.utils.distributed": "teleboost._overrides.verl.utils.distributed",
    "verl.utils.import_utils": "teleboost._overrides.verl.utils.import_utils",
    "verl.utils.device": "teleboost._overrides.verl.utils.device",
    "verl.utils.tracking": "teleboost._overrides.verl.utils.tracking",
    "verl.utils.torch_functional": "teleboost._overrides.verl.utils.torch_functional",
    "verl.utils.experimental.torch_functional": "teleboost._overrides.verl.utils.experimental.torch_functional",
    "verl.utils.activation_offload": "teleboost._overrides.verl.utils.activation_offload",
    "verl.utils.flops_counter": "teleboost._overrides.verl.utils.flops_counter",
    "verl.utils.seqlen_balancing": "teleboost._overrides.verl.utils.seqlen_balancing",
    "verl.utils.megatron_utils": "teleboost._overrides.verl.utils.megatron_utils",
    "verl.utils.megatron.optimizer": "teleboost._overrides.verl.utils.megatron.optimizer",
    "verl.utils.fsdp_utils": "teleboost._overrides.verl.utils.fsdp_utils",
    "verl.utils.model": "teleboost._overrides.verl.utils.model",
    "verl.utils.vllm_utils": "teleboost._overrides.verl.utils.vllm_utils",
    "verl.utils.ulysses": "teleboost._overrides.verl.utils.ulysses",
    "verl.utils.debug": "teleboost._overrides.verl.utils.debug",
    "verl.utils.debug.performance": "teleboost._overrides.verl.utils.debug.performance",
    "verl.utils.checkpoint.checkpoint_manager": "teleboost._overrides.verl.utils.checkpoint.checkpoint_manager",
    "verl.utils.checkpoint.fsdp_checkpoint_manager": "teleboost._overrides.verl.utils.checkpoint.fsdp_checkpoint_manager",
    "verl.utils.dataset.multiturn_sft_dataset": "teleboost._overrides.verl.utils.dataset.multiturn_sft_dataset",
    "verl.utils.dataset.rl_dataset": "teleboost._overrides.verl.utils.dataset.rl_dataset",
    "verl.utils.dataset.sft_dataset": "teleboost._overrides.verl.utils.dataset.sft_dataset",
    "verl.utils.reward_score": "teleboost._overrides.verl.utils.reward_score",
    "verl.utils.reward_score.prime_code.testing_util": "teleboost._overrides.verl.utils.reward_score.prime_code.testing_util",
    "verl.protocol": "teleboost._overrides.verl.protocol",
    "verl.tools.base_tool": "teleboost._overrides.verl.tools.base_tool",
    "verl.single_controller.base.worker": "teleboost._overrides.verl.single_controller.base.worker",
    "verl.single_controller.ray.base": "teleboost._overrides.verl.single_controller.ray.base",
    "verl.workers.reward_manager": "teleboost._overrides.verl.workers.reward_manager",
    "verl.workers.reward_manager.prime": "teleboost._overrides.verl.workers.reward_manager.prime",
    "verl.workers.actor.dp_actor": "teleboost._overrides.verl.workers.actor.dp_actor",
    "verl.workers.fsdp_workers": "teleboost._overrides.verl.workers.fsdp_workers",
    "verl.workers.rollout": "teleboost._overrides.verl.workers.rollout",
    "verl.workers.rollout.async_server": "teleboost._overrides.verl.workers.rollout.async_server",
    "verl.workers.rollout.hf_rollout": "teleboost._overrides.verl.workers.rollout.hf_rollout",
    "verl.workers.rollout.schemas": "teleboost._overrides.verl.workers.rollout.schemas",
    "verl.workers.rollout.sglang_rollout.sglang_rollout": "teleboost._overrides.verl.workers.rollout.sglang_rollout.sglang_rollout",
    "verl.workers.rollout.vllm_rollout.vllm_async_server": "teleboost._overrides.verl.workers.rollout.vllm_rollout.vllm_async_server",
    "verl.workers.rollout.vllm_rollout.vllm_rollout": "teleboost._overrides.verl.workers.rollout.vllm_rollout.vllm_rollout",
    "verl.workers.sharding_manager.fsdp_sglang": "teleboost._overrides.verl.workers.sharding_manager.fsdp_sglang",
    "verl.workers.sharding_manager.fsdp_ulysses": "teleboost._overrides.verl.workers.sharding_manager.fsdp_ulysses",
    "verl.trainer.fsdp_sft_trainer": "teleboost._overrides.verl.trainer.fsdp_sft_trainer",
    "verl.trainer.ppo.core_algos": "teleboost._overrides.verl.trainer.ppo.core_algos",
    "verl.trainer.ppo.metric_utils": "teleboost._overrides.verl.trainer.ppo.metric_utils",
    "verl.trainer.ppo.ray_trainer": "teleboost._overrides.verl.trainer.ppo.ray_trainer",
    "verl.trainer.ppo.reward": "teleboost._overrides.verl.trainer.ppo.reward",
    "verl.models.transformers.monkey_patch": "teleboost._overrides.verl.models.transformers.monkey_patch",
}


def apply() -> None:
    """Walk OVERRIDES, import each teleboost module, redirect verl namespace."""
    for verl_path, teleboost_path in OVERRIDES.items():
        try:
            mod = importlib.import_module(teleboost_path)
        except Exception as e:  # noqa: BLE001 - keep going so other overrides still install
            print(f"[teleboost.patches] WARN failed to load {teleboost_path}: {e}")
            continue
        sys.modules[verl_path] = mod

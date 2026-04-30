# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import re
import time
import warnings
from functools import partial
from typing import Any, Dict, List, Union

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision.transforms import InterpolationMode
from torchvision import transforms

import torch
import torch.distributed
from diffusers.image_processor import VaeImageProcessor
from omegaconf import DictConfig, OmegaConf, open_dict
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import ProfilerConfig, WorkerProfiler, WorkerProfilerExtension, log_gpu_memory_usage, simple_timer
from verl.utils.device import get_device_id, get_device_name, get_nccl_backend
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_local_path_from_hdfs, copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.py_functional import convert_to_regular_types
from verl.workers.fsdp_workers import ActorRolloutRefWorker, RewardModelWorker, create_device_mesh, get_sharding_strategy
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR
    
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def _zscore_tensor(values: torch.Tensor) -> torch.Tensor:
    mean = values.mean()
    std = values.std(unbiased=False)
    return (values - mean) / (std + 1e-8)


def _split_video_frames(data: DataProto, *, permute_to_tchw: bool) -> List[torch.Tensor]:
    batch_size = data.batch.batch_size[0]
    decoded = data.batch["video_frames"]
    frames = [x.squeeze(0) for x in decoded.chunk(batch_size, dim=0)]
    if permute_to_tchw:
        frames = [x.permute(1, 0, 2, 3) for x in frames]
    return frames


def _split_captions(caption, batch_size: int) -> List[str]:
    batch_caption = np.array_split(caption, batch_size)
    return [str(x.squeeze(0)) for x in batch_caption]


def _make_reward_batch(key: str, rewards: torch.Tensor, batch_size: int) -> DataProto:
    batch = TensorDict({key: rewards}, batch_size=batch_size)
    return DataProto(batch=batch)


            
class DiffusionActorRolloutRefWorker(ActorRolloutRefWorker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, model_deployment=None):
        super().__init__(config, role)

        # Wan-specific: separate Ulysses SP mesh for rollout (rollout SP can differ from actor SP).
        # Upstream verl 0.4.0 ActorRolloutRefWorker.__init__ only sets the actor-side
        # `self.ulysses_sharding_manager`; Wan's diffusion rollout needs its own.
        self.rollout_ulysses_sequence_parallel_size = self.config.rollout.get("ulysses_sequence_parallel_size", 1)
        device_name = get_device_name()
        world_size = torch.distributed.get_world_size()

        self.rollout_ulysses_device_mesh = None
        rollout_dp = world_size // self.rollout_ulysses_sequence_parallel_size
        if self.rollout_ulysses_sequence_parallel_size > 1:
            self.rollout_ulysses_device_mesh = init_device_mesh(
                device_name,
                mesh_shape=(rollout_dp, self.rollout_ulysses_sequence_parallel_size),
                mesh_dim_names=["dp", "sp"],
            )

        self.rollout_ulysses_sharding_manager = FSDPUlyssesShardingManager(self.rollout_ulysses_device_mesh)

    def apply_fsdp_checkpointing(self, model, target_types, p=1.0):
        """Activation checkpointing helper for Wan attention blocks (non-reentrant).

        Pre-X3 lived in the in-tree `verl/workers/fsdp_workers.py` fork. Moved here as
        an override so the `import-verl` path keeps Wan support without re-vendoring.
        """
        import math
        targets = [m for m in model.modules() if isinstance(m, target_types)]
        k = math.ceil(len(targets) * float(p))
        to_wrap = set(targets[:k])

        non_re_wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

        def check_fn(m):
            return m in to_wrap

        def wrapper_fn(m):
            return non_re_wrapper(m)

        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=wrapper_fn, check_fn=check_fn)

    def use_compile(self, model):
        """Wrap each Wan WanAttentionBlock.forward with torch.compile."""
        from wan.modules.model import WanAttentionBlock

        def compile_blocks(target):
            for block in target.blocks:
                if isinstance(block, WanAttentionBlock):
                    block.forward = torch.compile(block.forward, mode="max-autotune-no-cudagraphs")

        if hasattr(model, "low_noise_model") and hasattr(model, "high_noise_model"):
            compile_blocks(model.low_noise_model)
            compile_blocks(model.high_noise_model)
        else:
            compile_blocks(model)
        return model

    def _enable_compile(self, model, compile_export_mode):
        if compile_export_mode == "compile":
            model = self.use_compile(model)
        elif compile_export_mode == "export_aoti":
            pass
        elif compile_export_mode == "disabled":
            pass
        else:
            raise RuntimeError("expected compile_export_mode arg to be one of {compile, export_aoti, disabled}")
        return model

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
    ):
        """Wan-aware model + optimizer builder.

        Diverges from upstream `ActorRolloutRefWorker._build_model_optimizer` in three places:
          - tokenizer is loaded from the `google/umt5-xxl/` subdir of `model_path`
            (Wan ships the T5 tokenizer there);
          - `actor_model_config` is built via `GPT2Config.from_pretrained(...)` to
            bypass HF's `model_type` check (Wan does not register a model_type);
          - the actor module class is `wan.modules.model.WanModel`, with a special
            wan22 dual-model branch that wraps a low/high pair in `Wan22DualModel`.
        """
        from torch import optim
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from transformers import AutoModelForVision2Seq, GPT2Config

        from verl.utils.model import print_model_size
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path

        # Wan ships its T5 tokenizer under <model_path>/google/umt5-xxl/.
        tokenizer_path = os.path.join(local_path, "google/umt5-xxl")
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)
        # Wan transformer is bf16 across the board.
        torch_dtype = torch.bfloat16

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # Bypass AutoConfig — Wan has no `model_type` key.
        actor_model_config = GPT2Config.from_pretrained(
            local_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2"
        )

        init_context = get_init_weight_context_manager(mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                actor_module_class = AutoModelForVision2Seq
            else:
                from wan.modules.model import WanModel
                actor_module_class = WanModel

            compile_export_mode = "compile"
            wan_version = self.config.model.get("wan_version", "wan21")
            use_wan22 = wan_version == "wan22"
            if actor_module_class.__name__ == "WanModel" and use_wan22:
                wan22_high_path = self.config.model.get("high_noise_path", None)
                if not wan22_high_path:
                    wan22_high_path = os.path.join(local_path, "high_noise_model")
                    wan22_low_path = os.path.join(local_path, "low_noise_model")
                    if not os.path.exists(wan22_high_path):
                        raise ValueError(
                            "wan_version=wan22 requires model.high_noise_path/model.low_noise_path or existing "
                            "low_noise_model and high_noise_model near model.path"
                        )

            if actor_module_class.__name__ == "WanModel" and use_wan22:
                from teleboost.models.transformers.wan22 import Wan22DualModel

                def build_wan_model(path):
                    model = actor_module_class.from_pretrained(path, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)
                    if use_liger:
                        from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                        _apply_liger_kernel_to_instance(model=model)
                    # Upstream verl's `apply_monkey_patch` reads `model.config.num_attention_heads`,
                    # which fails for Wan (FrozenDict, no such attr). Pre-X3's in-tree fork instead
                    # checked `model.config.model_type == "t2v"` and applied Wan-specific Ulysses
                    # patches only when sp_size > 1. We mirror that here: skip the upstream call
                    # entirely, and install our own Ulysses patches when SP > 1.
                    if self.ulysses_sequence_parallel_size > 1:
                        from teleboost.models.transformers.wan import apply_wan_ulysses_patches
                        apply_wan_ulysses_patches(model)
                    model = self._enable_compile(model, compile_export_mode)
                    model.to(torch_dtype)
                    if enable_gradient_checkpointing:
                        from wan.modules.model import WanAttentionBlock
                        self.apply_fsdp_checkpointing(model, WanAttentionBlock, 1.0)
                    return model

                low_model = build_wan_model(wan22_low_path)
                high_local_path = copy_to_local(wan22_high_path, use_shm=self.config.model.get("use_shm", False))
                high_model = build_wan_model(high_local_path)
                boundary = self.config.model.get("wan22_boundary", 0.9)
                actor_module = Wan22DualModel(low_model, high_model, boundary=boundary)

                if self._is_lora:
                    print("LoRA is not supported for Wan2.2 dual-model setup; skipping.")
            else:
                actor_module = actor_module_class.from_pretrained(local_path, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)

                if use_liger:
                    from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                    _apply_liger_kernel_to_instance(model=actor_module)

                # See note in build_wan_model branch above. Skip apply_monkey_patch for Wan
                # and install our own Ulysses patches when SP > 1.
                if self.ulysses_sequence_parallel_size > 1:
                    from teleboost.models.transformers.wan import apply_wan_ulysses_patches
                    apply_wan_ulysses_patches(actor_module)

                actor_module = self._enable_compile(actor_module, compile_export_mode)
                actor_module.to(torch_dtype)
                if enable_gradient_checkpointing:
                    from wan.modules.model import WanAttentionBlock
                    self.apply_fsdp_checkpointing(actor_module, WanAttentionBlock, 1.0)

                if self._is_lora:
                    print("Applying LoRA to actor module")
                    actor_module.enable_input_require_grads()
                    lora_config = {
                        "task_type": TaskType.CAUSAL_LM,
                        "r": self.config.model.lora_rank,
                        "lora_alpha": self.config.model.lora_alpha,
                        "target_modules": convert_to_regular_types(self.config.model.target_modules),
                        "bias": "none",
                    }
                    actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))

        torch.distributed.barrier()
        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module, config=fsdp_config.get("wrap_policy", None), is_lora=self.config.model.get("lora_rank", 0) > 0
        )

        if self._is_rollout and self.config.rollout.name == "hf":
            auto_wrap_policy = None

        if self.rank == 0:
            print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                use_orig_params=True,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=fsdp_config.get("forward_prefetch", False),
            )
            from verl.utils.ulysses import register_cp_grad_reduce_hook
            register_cp_grad_reduce_hook(actor_module_fsdp)
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch >= 2.4 required for FSDP2"
            mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True)
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        if enable_activation_offload:
            enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)

        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

            actor_optimizer = optim.AdamW(
                actor_module_fsdp.parameters(),
                lr=optim_config.lr,
                betas=optim_config.get("betas", (0.9, 0.999)),
                weight_decay=optim_config.get("weight_decay", 1e-2),
            )

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            warmup_style = optim_config.get("warmup_style", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            if self.rank == 0:
                print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            if warmup_style == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps)
            elif warmup_style == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps, min_lr_ratio=min_lr_ratio, num_cycles=num_cycles
                )
            else:
                raise NotImplementedError(f"Warmup style {warmup_style} is not supported")

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self, trust_remote_code=False):
        """Wan-aware rollout builder.

        Adds a `config.type == "diffusion"` branch (taken by DanceGRPO) on top of
        upstream's vllm/sglang/hf options. The diffusion path uses TeleBoost's
        `DiffusionRollout` and `DiffusionBaseShardingManager` (the in-tree-verl
        re-exports were dropped by X3).
        """
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        device_name = get_device_name()
        rollout_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])
        rollout_name = self.config.rollout.name

        if self.config.type == "diffusion":
            from teleboost.workers.rollout.diffusion_rollout import DiffusionRollout
            from teleboost.workers.sharding_manager.diffusion import DiffusionBaseShardingManager

            rollout = DiffusionRollout(module=self.actor_module_fsdp, config=self.config)
            rollout_sharding_manager = DiffusionBaseShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=None,
                model_config=self.actor_model_config,
                offload_param=self._is_offload_param,
            )
            return rollout, rollout_sharding_manager

        # Non-diffusion paths: defer to upstream impl. We can't just call super() because
        # upstream's _build_rollout is a single method that includes the rollout_name dispatch,
        # so reimplement the vllm/sglang/hf branches here verbatim from upstream v0.4.0.
        if rollout_name == "hf":
            from verl.workers.rollout import HFRollout
            from verl.workers.sharding_manager.base import BaseShardingManager

            rollout = HFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
            rollout_sharding_manager = BaseShardingManager()
        elif rollout_name == "vllm":
            from verl.workers.rollout.vllm_rollout import vLLMRollout, vllm_mode
            from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))
            lora_kwargs = (
                {"lora_kwargs": {"enable_lora": True, "max_loras": 1, "max_lora_rank": self._lora_rank}} if self._is_lora else {}
            )
            if vllm_mode == "customized":
                rollout = vLLMRollout(
                    actor_module=self.actor_module_fsdp,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.actor_model_config,
                    trust_remote_code=trust_remote_code,
                    **lora_kwargs,
                )
            elif vllm_mode == "spmd":
                from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout

                vllm_rollout_cls = vLLMRollout if self.config.rollout.mode == "sync" else vLLMAsyncRollout
                rollout = vllm_rollout_cls(
                    model_path=local_path,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.actor_model_config,
                    device_mesh=rollout_device_mesh,
                    trust_remote_code=trust_remote_code,
                    **lora_kwargs,
                )
            else:
                raise NotImplementedError("vllm_mode must be 'customized' or 'spmd'")

            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)
            full_params = torch.distributed.get_world_size() == 1
            rollout_sharding_manager = FSDPVLLMShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                full_params=full_params,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                load_format=self.config.rollout.load_format,
                layered_summon=self.config.rollout.get("layered_summon", False),
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)
        elif rollout_name in ["sglang", "sglang_async"]:
            if rollout_name == "sglang_async":
                warnings.warn(
                    "'sglang_async' has been deprecated and merged into 'sglang'. Please use 'sglang' going forward.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            from verl.workers.rollout.sglang_rollout import SGLangRollout
            from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager

            local_path = copy_to_local(self.config.model.path)
            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            rollout = SGLangRollout(
                actor_module=local_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
                trust_remote_code=trust_remote_code,
            )
            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)

            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = "dummy_hf"
            rollout_sharding_manager = FSDPSGLangShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout._engine,
                model_config=self.actor_model_config,
                full_params="hf" in self.config.rollout.load_format,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)
        else:
            raise NotImplementedError(f"Rollout name: {self.config.rollout.name} is not supported")

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from .dp_actor import DiffusionDataParallelPPOActor as DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from omegaconf import OmegaConf

        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))

        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
            )

            # get the original unwrapped module
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
                self.config.actor.use_fused_kernels = use_fused_kernels

            # The diffusion actor reads ``pixel_weight`` (VIPO) and
            # ``flow_grpo`` from ``self.config`` inside dp_actor, but
            # those Hydra blocks live at ``actor_rollout_ref.<flag>``,
            # not ``actor_rollout_ref.actor.<flag>``.  Without this
            # merge, VIPO mode produces dense ``(T,H,W)`` log-probs in
            # the rollout but scalar log-probs in the actor (the
            # actor's ``_pixel_enabled()`` returns False), which fails
            # at ``ratio = exp(new - old)`` with a shape mismatch like
            # ``(16) vs (3072)``.  Same path for flow-grpo's
            # ``shuffle_timesteps`` and ``timestep_indices``.
            #
            # ``self.config.actor`` is a struct-typed OmegaConf node;
            # opening struct mode briefly is required to add a new key.
            _was_struct = OmegaConf.is_struct(self.config.actor)
            OmegaConf.set_struct(self.config.actor, False)
            try:
                for _propagate_key in ("pixel_weight", "flow_grpo"):
                    if _propagate_key in self.config and _propagate_key not in self.config.actor:
                        self.config.actor[_propagate_key] = self.config[_propagate_key]
            finally:
                OmegaConf.set_struct(self.config.actor, _was_struct)

            self.actor = DataParallelPPOActor(config=self.config.actor, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer)

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_rollout and hasattr(self.rollout, "vae_module"):
            self.rollout.vae_module.model.decoder = torch.compile(
                self.rollout.vae_module.model.decoder,
                mode="default",
            )
        
        if self._is_ref:
            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                # Upstream verl 0.4.0 expects a list (`["model", "optimizer", "extra"]`); pre-X3
                # accepted the wrapping `{contents: [...]}` dict. Pass the inner list to satisfy
                # FSDPCheckpointManager's `"model" in checkpoint_contents` assertion.
                checkpoint_contents=self.config.actor.checkpoint.contents,
            )

        if not self._is_actor and self._is_rollout:
            # If ActorRolloutRefWorker is initialized as a standalone rollout,
            # create a checkpoint manager for FSDP model to allow loading FSDP checkpoints for rollout.

            checkpoint_contents = OmegaConf.create({"load_contents": ["model"], "save_contents": []})
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=None,
                lr_scheduler=None,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_contents=checkpoint_contents,
            )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @WorkerProfiler.annotate(color="red")
    def generate_sequences(self, prompts: DataProto):
        prompts = prompts.to(get_device_id())
        timing_generate = {}
        with self.rollout_ulysses_sharding_manager:
            with self.rollout_sharding_manager:
                log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)

                prompts = self.rollout_sharding_manager.preprocess_data(prompts)
                prompts = self.rollout_ulysses_sharding_manager.preprocess_data(prompts)
                with simple_timer("generate_sequences", timing_generate):
                    output = self.rollout.generate_sequences(prompts=prompts)
                    
                prompts = self.rollout_sharding_manager.postprocess_data(prompts)
                log_gpu_memory_usage("After rollout generation", logger=logger)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        """Wan-aware actor update.

        Diverges from upstream `ActorRolloutRefWorker.update_actor` in two ways:
        - data is left on CPU (the actor moves it onto GPU per micro-batch inside
          update_policy; diffusion DataProto is too large to fit a whole batch on
          one device);
        - skip the FLOPs / mfu metrics block: it depends on `meta_info["global_token_num"]`,
          which is set by LM rollouts but not by `DiffusionRollout`. We just step the
          scheduler and return the inner update_policy metrics.
        """
        data = data.to("cpu")

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            metrics = self.actor.update_policy(data=data)
            self.actor_lr_scheduler.step()

            output = DataProto(meta_info={"metrics": metrics})
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        return output


class QwenRewardModelWorker(RewardModelWorker):
    """
    Qwen VLM-based Reward Model Worker.
    
    Uses vLLM for distributed inference with Qwen VL model to evaluate
    video quality through structured prompts.
    
    Configuration options (in reward_model config):
        - rollout.temperature: Sampling temperature (default: 0.8)
        - rollout.top_p: Top-p sampling (default: 0.9)
        - rollout.max_tokens: Maximum output tokens (default: 128)
        - extra_config.max_pixels: Max pixels for video (default: 360*420)
        - extra_config.fps: Frames per second (default: 1.0)
        - extra_config.video_base_path: Base path for video files (optional)
    """
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize the Qwen reward model and sampling parameters."""
        try:
            self.reward_rollout, self.reward_rollout_sharding_manager = self._build_reward_rollout()
            
            # Get sampling params from config with defaults
            from vllm import SamplingParams
            temperature = self.config.rollout.get("temperature", 0.8)
            top_p = self.config.rollout.get("top_p", 0.9)
            max_tokens = self.config.rollout.get("max_tokens", 128)
            
            self.sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )
            
            # Get extra config for video processing
            extra_config = self.config.get("extra_config", {})
            self.max_pixels = extra_config.get("max_pixels", 360 * 420)
            self.fps = extra_config.get("fps", 1.0)
            self.video_base_path = extra_config.get("video_base_path", "")
            
            logger.info(f"Qwen reward model initialized successfully")
            logger.info(f"Sampling params: temp={temperature}, top_p={top_p}, max_tokens={max_tokens}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qwen reward model: {e}")
            raise

    def _build_reward_rollout(self, trust_remote_code=False):
        device_name = get_device_name()

        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp  # world_size is the total number of GPUs in the cluster
        assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        rollout_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])
        rollout_name = self.config.rollout.name  # rollout backend (e.g. vllm)
        
        from verl.workers.rollout.vllm_rollout import vllm_mode, vLLMRollout
        from teleboost.workers.sharding_manager.reward_qwen import RewardVLLMManager
        log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
        local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))  # use_shm: whether to use shared memory
       
        # lora_kwargs = {"lora_kwargs": {"enable_lora": True, "max_loras": 1, "max_lora_rank": self._lora_rank}} if self._is_lora else {}
        lora_kwargs = {}
        if vllm_mode == "customized":
            rollout = vLLMRollout(actor_module=self.actor_module_fsdp, config=self.config.rollout, tokenizer=self.tokenizer, model_hf_config=self.actor_model_config, trust_remote_code=trust_remote_code, **lora_kwargs)
            
        elif vllm_mode == "spmd":
            # from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout
            lora_kwargs={}
            from transformers import AutoConfig
            actor_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2"
            )
            tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
            vllm_rollout_cls = vLLMRollout
            rollout = vllm_rollout_cls(model_path=local_path, config=self.config.rollout, tokenizer=tokenizer, model_hf_config=actor_model_config, device_mesh=rollout_device_mesh, trust_remote_code=trust_remote_code, **lora_kwargs)
        else:
            raise NotImplementedError("vllm_mode must be 'customized' or 'spmd'")
        
        log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)
        full_params = torch.distributed.get_world_size() == 1
        #TODO
        rollout_sharding_manager = RewardVLLMManager(
            # module=self.actor_module_fsdp,
            inference_engine=rollout.inference_engine,
            # model_config=self.actor_model_config,
            full_params=full_params,
            device_mesh=rollout_device_mesh,
            # offload_param=self._is_offload_param,
            load_format=self.config.rollout.load_format,
            layered_summon=self.config.rollout.get("layered_summon", False),
        )
        log_gpu_memory_usage("After building sharding manager", logger=logger)
        
        return rollout, rollout_sharding_manager
        
    
    def _create_simple_prompt(self) -> str:
        """Build the structured video-quality evaluation prompt for Qwen-VL.

        Note: prompt content is intentionally Chinese — Qwen-VL is best
        prompted in Chinese for the structured-scoring template below.
        """
        return """请你作为一个专业视频质量评估助手，参考以下评分标准和格式，对给定的视频进行多维度质量评估。请严格按照输出格式，以客观、公正、结构化的方式打分。

                评估维度（每项满分100分）：
                1. 视觉审美（Aesthetics）：
                - 参考项：构图是否合理、光影运用是否自然、色彩搭配是否和谐、整体画面是否具有美感。
                - 高分标准：画面构图精妙、光影自然、色彩生动，具备艺术性。
                - 扣分项：画面凌乱、光照极端或失衡、颜色搭配不当或灰暗。

                2. 局部变形（Distortion）：
                - 参考项：人物或物体是否出现异常形态、肢体是否扭曲、是否有结构性突变或失真、是否突然消失。
                - 高分标准：视频中不存在明显变形，物体结构自然、稳定。
                - 扣分项：出现严重扭曲、肢体不合理、局部区域断裂或消失。

                3. 视觉伪影与不一致（Artifacts/Inconsistency）：
                - 参考项：是否存在突变区域、马赛克、色块、条纹、边缘断裂、纹理模糊等问题。
                - 高分标准：无明显视觉瑕疵，画面一致性强。
                - 扣分项：出现视觉伪影或明显瑕疵，视觉体验受到影响。

                4. 清晰度（Sharpness）：
                - 参考项：细节呈现的清晰度，边缘锐利程度，物体是否具备较高的辨识度。
                - 高分标准：画面细节丰富、边缘清晰锐利。
                - 扣分项：整体模糊、边缘不清晰、细节缺失。

                5. 视觉一致性（Consistency）：
                - 参考项：视频内容在时间上的连贯性，是否存在跳帧、镜头突变或画面不稳定等问题。
                - 高分标准：过渡自然，时间逻辑连贯，画面稳定。
                - 扣分项：镜头跳跃明显、物体突然改变状态、画面抖动。

                评分规则：
                - 每个维度评分在 0 ~ 100 范围内，越好越高分。
                - 合计为五项得分的算术平均，保留整数。
                - 对于某项严重失真或效果极差（如严重模糊、强伪影等），请大胆给出低分（例如低于30分）。
                - 每个视频的打分应充分拉开差距，避免视频之间出现“同分”或“几乎同分”情况。
                - 请确保不同维度之间的评分不互相矛盾，确保评分具有可比性与区分度。

                输出格式（严格遵守）：
                dim1:XX分,dim2:XX分,dim3:XX分,dim4:XX分,dim5:XX分,合计:XX分

                风格要求：
                - 禁止输出解释性文字或分析过程。
                - 禁止使用“我认为”、“可能”、“大致”等模糊词语。
                - 输出必须严格按照上述格式，一次性返回评估结果。

                请严格按照输出格式要求，输出且只输出输出格式的内容。请依照以上标准、逻辑和格式，对视频进行结构化质量评估。
                """
        
    def _generate_chat_batch_prompts(self, batch_path, max_pixels=None, fps=None) -> List:
        """
        Generate prompts for a batch of video paths.
        
        Args:
            batch_path: List of video file paths
            max_pixels: Max pixels for video processing (uses self.max_pixels if None)
            fps: Frames per second (uses self.fps if None)
            
        Returns:
            List of message dicts for LLM chat
        """
        max_pixels = max_pixels or getattr(self, 'max_pixels', 360 * 420)
        fps = fps or getattr(self, 'fps', 1.0)
        
        messages = []
        prompt = self._create_simple_prompt()
        
        for file_path in batch_path:
            # Use configurable base path instead of hardcoded value
            video_path = str(file_path)
            if hasattr(self, 'video_base_path') and self.video_base_path:
                video_path = video_path.replace("./", self.video_base_path)
            
            message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": f"file://{video_path}"},
                            "max_pixels": max_pixels,
                            "fps": fps,
                        },
                        {
                            "type": "text", 
                            "text": prompt
                        },
                    ],
                }
            ]
            messages.append(message)
        
        return messages
    
    def _generate_batch_prompts(self,batch_id) -> List:
        """
        generate prompts for a batch of paths.
        Args:
            - batch_id: a List[str] item that consists all the paths of videos in the batch.
            - max_pixels: default to 360*420, int
            - fps: default to 1.0, float
        Returns:
            - A List[List[Dict[str,Any]]] item that each element is a List satisfying the conversation format of llm.chat method.
        """
        messages=[]
        simple_prompt = self._create_simple_prompt()
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>{simple_prompt}<|im_end|>\n<|im_start|>assistant\n"
        logger.info(f"Starting generating batch of prompts ...")
        for video_id in batch_id:
                message = [
                    {
                        "prompt": prompt,
                        'multi_modal_data': {'video':[video_id]},
                    }
                ]
                messages.append(message)   
        return messages
                
    def _parse_simple_evaluation(self, output_text: str) -> Dict[str, Any]:
        """
        extract the score in the output_text. Aligning with new prompts format.
        Args:
            - output_text: the output text in "str" form from llm rollout
        Returns:
            - A dict that includes keys:
                - overall_score: the score extract from the output text, float
                - dimensionn_scores: dimension scores, Dict[str,float]
                - summery: output_text[:500], str
                - raw_output: output_text, str
        """
        # Score-extraction patterns. Patterns contain Chinese on purpose — they
        # match the Qwen-VL output format used in `_create_simple_prompt`.
        score_patterns = [
                r'合计[：:]\s*(\d+(?:\.\d+)?)\s*分',          # overall total
                r'综合得分[：:]\s*(\d+(?:\.\d+)?)\s*分',      # composite score
                r'总分[：:]\s*(\d+(?:\.\d+)?)\s*分',          # total score
                r'dim5[：:]\s*(\d+(?:\.\d+)?)\s*分.*?合计[：:]\s*(\d+(?:\.\d+)?)\s*分',  # extract overall after dim5
                r'最终[：:]\s*(\d+(?:\.\d+)?)\s*分',          # final score
                r'评分[：:]\s*(\d+(?:\.\d+)?)\s*分',          # rating
                r'质量评分[：:]\s*(\d+(?:\.\d+)?)',          # quality rating
                r'分数[：:]\s*(\d+(?:\.\d+)?)',              # score
                r'(\d+(?:\.\d+)?)\s*分',                     # bare "N points"
                r'(\d+(?:\.\d+)?)/100',                      # N/100
                r'(\d+(?:\.\d+)?)%',                         # N%
            ]

        score = 50.0  # default score when nothing matches
        for pattern in score_patterns:
            match = re.search(pattern, output_text)
            if match:
                # For multi-group patterns take the last group (the overall total).
                if len(match.groups()) > 1:
                    found_score = float(match.group(2))
                else:
                    found_score = float(match.group(1))

                if found_score > 100:
                    found_score = min(found_score, 100)
                score = found_score
                logger.info(f"Found score: {score} using pattern: {pattern}")
                break

        if score == 50.0:
            logger.warning(f"No score found in output, using default 50.0. Output: {output_text[:200]}...")

        # Try to extract per-dimension scores.
        dimension_scores = {}
        dim_patterns = [
                (r'dim1[：:]\s*(\d+(?:\.\d+)?)\s*分', 'visual_artifacts'),
                (r'dim2[：:]\s*(\d+(?:\.\d+)?)\s*分', 'local_deformation'),
                (r'dim3[：:]\s*(\d+(?:\.\d+)?)\s*分', 'noise_quality'),
                (r'dim4[：:]\s*(\d+(?:\.\d+)?)\s*分', 'clarity_sharpness'),
                (r'dim5[：:]\s*(\d+(?:\.\d+)?)\s*分', 'color_accuracy'),
            ]
            
        for pattern, dim_name in dim_patterns:
            match = re.search(pattern, output_text)
            if match:
                dimension_scores[dim_name] = float(match.group(1))
            
        result = {
                "overall_score": score,
                "summary": output_text[:500] + "..." if len(output_text) > 500 else output_text,
                "raw_output": output_text
            }
            
        # If per-dimension scores were extracted, include them in the result.
        if dimension_scores:
            result["dimension_scores"] = dimension_scores
            logger.info(f"Extracted dimension scores: {dimension_scores}")
            
        return result 
    
    def _get_batch_reward(self, batch_output: List) -> List:
        """
        Extract rewards from batch output using _parse_simple_evaluation.
        
        Args:
            batch_output: List of model outputs to parse
            
        Returns:
            List of overall scores
        """
        results = []
        for single_prompt_output in batch_output:
            for response in single_prompt_output:
                output = response.outputs[0]
                output_text = output.text
                logger.debug(f"Generated text length: {len(output_text)}")
                logger.debug(f"Generated text preview: {output_text[:200]}...")
                
                result = self._parse_simple_evaluation(output_text)
                overall_score = result.get("overall_score", 50.0)
                
                logger.info(f"Quality score: {overall_score}/100")
                results.append(overall_score)
        return results


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @WorkerProfiler.annotate(color="brown")
    def compute_rm_score(self, datas: DataProto):
        """Compute reward scores using Qwen VLM."""
        import time
        start_time = time.time()
        
        datas = datas.to(get_device_id())
        
        with self.reward_rollout_sharding_manager:
            datas = self.reward_rollout_sharding_manager.preprocess_data(datas)
            
            logger.info("Starting Qwen reward computation...")
            video_ids = datas.non_tensor_batch['video_ids']
            
            import numpy as np
            batch_ids = np.array_split(video_ids, datas.batch.batch_size[0])
            
            all_rewards = []
            for batch_id in batch_ids:
                batch_message = self._generate_batch_prompts(batch_id)
                batch_output = []
                
                for message in batch_message:
                    output = self.reward_rollout.inference_engine.generate(
                        message,
                        sampling_params=self.sampling_params
                    )
                    batch_output.append(output)
                    
                batch_reward = self._get_batch_reward(batch_output)
                all_rewards += batch_reward
        
            all_rewards = torch.tensor(all_rewards)
            batch = TensorDict(
                {"rewards": all_rewards},
                batch_size=datas.batch.batch_size[0]
            )
            
            batch_reward = DataProto(batch=batch, non_tensor_batch=datas.non_tensor_batch)
            batch_reward = self.reward_rollout_sharding_manager.postprocess_data(batch_reward)
        
        elapsed = time.time() - start_time
        logger.info(f"Qwen reward computation completed in {elapsed:.2f}s")
        
        return batch_reward
        
    
# TODO(sgm): we may need to extract it to dp_reward_model.py
class DiffusionRewardModelWorker(RewardModelWorker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        self.reward_module, self.preprocess_val,self.tokenizer = self._build_model(config=self.config)
        
        #TODO
        self.image_processor = VaeImageProcessor(16)

    def _build_model(self, config):
        # the following line is necessary

        use_shm = config.model.get("use_shm", False)
        # download the checkpoint from hdfs
        local_path = copy_to_local(config.model.path, use_shm=use_shm)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer, use_shm=use_shm)
            self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=config.model.get("trust_remote_code", False))
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get("trust_remote_code", False))

        from typing import Union

        import huggingface_hub
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        from hpsv2.utils import hps_version_map, root_path

        def initialize_model():
            model_dict = {}
            model, preprocess_train, preprocess_val = create_model_and_transforms(
                'ViT-H-14',
                self.config.model.path,
                precision='amp',
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=False,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                light_augmentation=True,
                aug_cfg={},
                output_dict=True,
                with_score_predictor=False,
                with_region_predictor=False
            )
            model_dict['model'] = model
            model_dict['preprocess_val'] = preprocess_val
            return model_dict
        model_dict = initialize_model()
        reward_module = model_dict['model']
        preprocess_val = model_dict['preprocess_val']

        checkpoint = torch.load(self.config.model.path)
        reward_module.load_state_dict(checkpoint['state_dict'])
        processor = get_tokenizer('ViT-H-14')
        
        return reward_module, preprocess_val,processor
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @WorkerProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        
        # Support all hardwares
        data=data.pop(
            batch_keys=['video_frames'],
            non_tensor_batch_keys=["caption"],
        )
        decoded_image=data.batch['video_frames']
        decoded_images = decoded_image.chunk(data.batch.batch_size[0], dim=0)
        decoded_images = [x.squeeze(0) for x in decoded_images]
        caption=data.non_tensor_batch['caption']
        batch_caption = _split_captions(caption, data.batch.batch_size[0])
        batch_indices = torch.chunk(torch.arange(len(batch_caption)), len(batch_caption))
        all_rewards = []  
        self.reward_module.to(device=get_device_id())
        for index, batch_idx in enumerate(batch_indices):
            with torch.no_grad():
                frame = decoded_images[index][:, 0, :, :]  # (C, H, W)
                # Convert to PIL image
                frame_np = frame.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
                frame_np = (frame_np * 255).astype(np.uint8)
                frame_pil = Image.fromarray(frame_np)
                    
                image = self.preprocess_val(frame_pil).unsqueeze(0).to(device=get_device_id(), non_blocking=True)
                # Process the prompt
                text = self.tokenizer([batch_caption[index]]).to(device=get_device_id(), non_blocking=True)
                # Calculate the HPS
                with torch.amp.autocast('cuda'):
                    outputs = self.reward_module(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T
                    hps_score = torch.diagonal(logits_per_image)
                all_rewards.append(hps_score.unsqueeze(0))

        all_rewards = torch.cat(all_rewards, dim=0)

        # all_rewards=all_rewards.to(torch.device('cpu'))
        batch = TensorDict(
            {
                "rewards": all_rewards,
            },
            batch_size=len(batch_caption)
        )
        self.reward_module.to(torch.device('cpu'))
        batch_reward= DataProto(batch=batch)
        
        return batch_reward


# ================================= Async related workers =================================
class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    def _build_rollout(self, trust_remote_code=False):
        rollout, rollout_sharding_manager = super()._build_rollout(trust_remote_code)

        # NOTE: rollout is not actually initialized here, it's deferred
        # to be initialized by AsyncvLLMServer.

        self.vllm_tp_size = self.config.rollout.tensor_model_parallel_size
        self.vllm_dp_rank = int(os.environ["RANK"]) // self.vllm_tp_size
        self.vllm_tp_rank = int(os.environ["RANK"]) % self.vllm_tp_size

        # used for sleep/wake_up
        rollout.sharding_manager = rollout_sharding_manager

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        raise NotImplementedError("AsyncActorRolloutRefWorker does not support generate_sequences")

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        """Called by ExternalRayDistributedExecutor collective_rpc."""
        if self.vllm_tp_rank == 0 and method != "execute_model":
            print(f"[DP={self.vllm_dp_rank},TP={self.vllm_tp_rank}] execute_method: {method if isinstance(method, str) else 'Callable'}")
        return self.rollout.execute_method(method, *args, **kwargs)

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def chat_completion(self, json_request):
        ret = await self.rollout.chat_completion(json_request)
        return ret

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def wake_up(self):
        await self.rollout.wake_up()
        # return something to block the caller
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def sleep(self):
        await self.rollout.sleep()
        # return something to block the caller
        return True

# Helper function to compute position ids with attention mask
def clip_transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=BICUBIC, antialias=False),
        transforms.CenterCrop(n_px),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])
    
def _split_batch(data: DataProto, dp_size: int, dp_rank: int) -> DataProto:
    """
    Split the batch for data parallelism.
    """
    batch_size = data.batch.batch_size[0]
    
    base = batch_size // dp_size
    rem = batch_size % dp_size
    start = dp_rank * base + min(dp_rank, rem)
    end = start + base + (1 if dp_rank < rem else 0)
    
    local_batch = data.batch[start: end]
    local_non_tensor = {k: v[start: end] for k, v in data.non_tensor_batch.items()}
    
    return DataProto(batch=local_batch, non_tensor_batch=local_non_tensor)


 # ------------------------------------------------------------------------------------------------
# WORLD_SIZE = torch.distributed.get_world_size()
# AestheticRewardModelWorker is a worker that computes aesthetic scores for images.
class AestheticRewardModelWorker(RewardModelWorker):
    """
    RewardModelWorker for aesthetic score evaluation using CLIP + linear regression.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self.is_active = False
        self.aes_dp = torch.distributed.get_world_size() // 4
        # self.rank = dist.get_rank()
        if self.rank < self.aes_dp:
            self.is_active = True
            aesthetic_cfg = self.config.get("aesthetic", {}) or {}
            self.clip_model_path = aesthetic_cfg.get(
                "clip_model_path",
                "/gemini/space/wyb/model/arena_model/ViT-L-14.pt",
            )
            self.aes_model_path = aesthetic_cfg.get(
                "aes_model_path",
                "/gemini/space/wyb/model/arena_model/sa_0_4_vit_l_14_linear.pth",
            )
            self._build_model() 
        else:
            print(f"[RANK {self.rank}] AestheticWorker is inactive")
        
    
    def _load_aesthetic_model(self, cache_folder):
        path_to_model = cache_folder
        m = nn.Linear(768, 1)
        s = torch.load(path_to_model)
        m.load_state_dict(s)
        m.eval()
        return m
    
    def _build_model(self):
        
        from teleboost.models.offline_clip import create_offline_clip_model
        self.clip_model = create_offline_clip_model(self.clip_model_path, "cpu")
        self.aesthetic_model = self._load_aesthetic_model(self.aes_model_path)
        num_params_1 = sum(p.numel() for p in self.clip_model.visual.parameters())
        num_params_3 = sum(p.numel() for p in self.clip_model.original_model.parameters())
        num_params_2 = sum(p.numel() for p in self.aesthetic_model.parameters())
        print(f"aes_model params: {num_params_1 + num_params_2 + num_params_3}")

    @register(dispatch_mode=Dispatch.ALL_TO_ALL)
    @WorkerProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        print(f"  -> Actor 'aes' CUDA_MPS_ACTIVE_THREAD_PERCENTAGE = {self.get_mps_percentage()}")

        start_time = time.time()
        datas=data.pop(
            batch_keys=['video_frames'],
            non_tensor_batch_keys=["caption"],
        )
    
        if self.is_active == False:
            empty_bs = datas.batch.batch_size[0] // self.aes_dp
            print(f"[RANK {self.rank}] AestheticWorker is inactive, returning zeros")
            dummy_rewards = torch.zeros(empty_bs, device=torch.device('cpu'))
            return _make_reward_batch("aes_rewards", dummy_rewards, empty_bs)
        
        datas = _split_batch(datas, self.aes_dp, self.rank % self.aes_dp)
        print(f"[RANK {self.rank}] AestheticWorker processing batch size: {datas.batch.batch_size[0]}")
        decoded_images = _split_video_frames(datas, permute_to_tchw=True)
        caption = datas.non_tensor_batch['caption']       

        batch_caption = _split_captions(caption, datas.batch.batch_size[0])
        batch_indices = torch.chunk(torch.arange(len(batch_caption)), len(batch_caption))
        
        transform = clip_transform(224)
        
        all_rewards = []
        print(f"aes batch size: {len(batch_caption)}")
        for index, batch_idx in enumerate(batch_indices):
            with torch.no_grad():
                transformed = torch.stack([transform(image) for image in decoded_images[index]])
                transformed = transformed.to(device=get_device_id()).to(device=get_device_id())
                self.clip_model.to(device=get_device_id())
                self.aesthetic_model.to(device=get_device_id())
                # Compute the aesthetic score
                features = self.clip_model.encode_image(transformed).float()
                features = F.normalize(features, dim=-1)
                score = self.aesthetic_model(features).squeeze(-1)
                
            mean_score = (score / 10).mean().item()
            print(f"aes_score value: {mean_score}")
            all_rewards.append(torch.tensor(mean_score, device=get_device_id()).unsqueeze(0))
            
        all_rewards = torch.cat(all_rewards, dim=0)
        
        all_rewards = _zscore_tensor(all_rewards)
        
        all_rewards = all_rewards.to(torch.device('cpu'))
        batch = TensorDict(
            {
                "aes_rewards": all_rewards,
            },
            batch_size=len(batch_caption)
        )
        self.clip_model.to(torch.device('cpu'))
        self.aesthetic_model.to(torch.device('cpu'))
        # non_tensor_batch = data.non_tensor_batch
        
        end_time = time.time()
        print(f"aes compute time: {end_time - start_time}s")
        
        # return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)   
        return DataProto(batch=batch)     
   
import argparse
def dict_to_namespace(d):
    return argparse.Namespace(**d)
     
# RAFTRewardModelWorker is a worker that computes RAFT scores for images.    
class RAFTRewardModelWorker(RewardModelWorker):
    """
    RAFTRewardModelWorker is a worker that computes RAFT scores for images.
    It uses a pre-trained model to evaluate the RAFT quality of images.
    """

    @register(dispatch_mode=Dispatch.ALL_TO_ALL)
    def init_model(self, stride: int = 1):
        self.is_active = False
        self.raft_dp = torch.distributed.get_world_size() // 2
        # self.rank = dist.get_rank()
        if self.rank >= self.raft_dp:
            self.is_active = True
            raft_cfg = self.config.get("raft", {}) or {}
            self.raft_model_path = raft_cfg.get(
                "model_path",
                "/gemini/space/wyb/model/arena_model/raft-things.pth",
            )
            self.stride = stride
            self._build_model()
        else:
            print(f"[RANK {self.rank}] RAFTWorker is inactive")
            
    
    def _build_model(self):

        args_dict = {
            "small": False,
            "mixed_precision": False,
            "alternate_corr": False,
        }
        args = dict_to_namespace(args_dict)
        
        from teleboost.models.raft.raft import RAFT
        model = RAFT(args)
        
        from collections import OrderedDict
        # Strip "module." prefix
        state_dict = torch.load(self.raft_model_path, map_location="cpu")
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k  # remove 'module.'
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        
        self.raft_model = model
        self.raft_model.eval()
        self.raft_model.args.mixed_precision = False
        
        num_params = sum(p.numel() for p in self.raft_model.parameters())
        print(f"raft_model params: {num_params}")        
        

    def calculate_flow_score_from_tensor(self, frames):
        if len(frames) < 2:
            print("Not enough frames to compute optical flow.")
            return 0.0

        from teleboost.models.utils.utils import InputPadder

        optical_flows = []
        with torch.no_grad():
            for i in range(len(frames) - 1):
                image1 = frames[i].float().unsqueeze(0)
                image2 = frames[i + 1].float().unsqueeze(0)
                
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = self.raft_model(image1, image2, iters=20, test_mode=True)
                flow_magnitude = torch.norm(flow_up.squeeze(0), dim=0)
                mean_flow = flow_magnitude.mean().item()
                optical_flows.append(mean_flow)
        res = float(np.mean(optical_flows))
        return res        
    
    @register(dispatch_mode=Dispatch.ALL_TO_ALL)
    @WorkerProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        print(f"  -> Actor 'raft' CUDA_MPS_ACTIVE_THREAD_PERCENTAGE = {self.get_mps_percentage()}")

        start_time = time.time()
        # Read data without popping it
        datas=data.pop(
            batch_keys=['video_frames'],
            non_tensor_batch_keys=["caption"],
        )
        
        if self.is_active == False:
            empty_bs = datas.batch.batch_size[0] // self.raft_dp
            print(f"[RANK {self.rank}] RAFTWorker is inactive, returning zeros")
            dummy_rewards = torch.zeros(empty_bs, device=torch.device('cpu'))
            return _make_reward_batch("raft_rewards", dummy_rewards, empty_bs)
        datas = _split_batch(datas, self.raft_dp, self.rank % self.raft_dp)
        print(f"[RANK {self.rank}] RAFTWorker processing batch size: {datas.batch.batch_size[0]}")
        decoded_images = _split_video_frames(datas, permute_to_tchw=True)
        
        caption = datas.non_tensor_batch['caption']       

        batch_caption = _split_captions(caption, datas.batch.batch_size[0])
        batch_indices = torch.chunk(torch.arange(len(batch_caption)), len(batch_caption))

        self.raft_model.to(get_device_id())
        print(f"raft batch size: {len(batch_caption)}")
        all_rewards = []
        for index, batch_idx in enumerate(batch_indices):
            video = decoded_images[index][::self.stride]
            # print(f"video shape: {video.shape[0]}")
            video = video.to(get_device_id())    
            flow_score = self.calculate_flow_score_from_tensor(video)
    
            print(f"flow_score value: {flow_score}")
            all_rewards.append(torch.tensor(flow_score, device=get_device_id()).unsqueeze(0))
            
        all_rewards = torch.cat(all_rewards, dim=0)
        all_rewards = _zscore_tensor(all_rewards)
                
        all_rewards = all_rewards.to(torch.device('cpu'))
        batch = TensorDict(
            {
                "raft_rewards": all_rewards,
            },
            batch_size=len(batch_caption)
        )
        self.raft_model.to(torch.device('cpu'))
        # non_tensor_batch = data.non_tensor_batch
        
        end_time = time.time()
        print(f"raft compute time: {end_time - start_time}s")        
        
        # return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
        return DataProto(batch=batch)

# VideoclipRewardModelWorker is a worker that computes video-clip scores for images. 
class VideoclipRewardModelWorker(RewardModelWorker):
    """
    VideoclipRewardModelWorker is a worker that computes video-clip scores for images.
    It uses a pre-trained model to evaluate the video-clip quality of images.
    """

    @register(dispatch_mode=Dispatch.ALL_TO_ALL)
    def init_model(self):
        self.is_active = False
        self.videoclip_dp = torch.distributed.get_world_size() // 2
        # self.rank = dist.get_rank()
        if self.rank < self.videoclip_dp:
            self.is_active = True
            videoclip_cfg = self.config.get("videoclip", {}) or {}
            # .get() returns the value if present else the default
            # .get(key, default_value)
            self.videoclip_model_path = videoclip_cfg.get(
                "model_path",
                "/gemini/space/wyb/model/arena_model/VideoCLIP-XL/VideoCLIP-XL.bin",
            )
            self.v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
            self.v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
            self._build_model() 
        else:
            print(f"[RANK {self.rank}] VideoclipWorker is inactive")
        
    def _build_model(self):
        from teleboost.models.VideoCLIP_XL.modeling import VideoCLIP_XL
        # Match VideoCLIP_XL vision_model frame configuration
        self.videoclip_model = VideoCLIP_XL()
        state_dict = torch.load(self.videoclip_model_path, map_location="cpu")
        self.videoclip_model.load_state_dict(state_dict)
        
        num_params = sum(p.numel() for p in self.videoclip_model.parameters())
        print(f"videoclip_model params: {num_params}")             
        # self.videoclip_model.to(get_device_id).eval()

    def _video_preprocessing(self, frames: torch.Tensor, fnum=8):
        """
        frames: torch.Tensor, shape [C, T, H, W], e.g. [3, 13, 720, 720]
        Output: torch.Tensor, shape [1, T, C, 224, 224]
        """
        frames = frames.permute(1, 2, 3, 0).cpu().numpy()  # [C, T, H, W] -> [T, H, W, C]

        total_frames = frames.shape[0]
        step = max(1, total_frames // fnum)
        sampled_frames = frames[::step][:fnum]  # [fnum, H, W, C]
        
        import cv2
        vid_tube = []
        
        for fr in sampled_frames:
            fr = fr[:, :, ::-1]  # BGR to RGB
            fr = cv2.resize(fr, (224, 224))
            fr = self._normalize(fr)
            fr = np.expand_dims(fr, axis=(0, 1)) # (1,1,H,W,C)
            vid_tube.append(fr)
        
        vid_tube = np.concatenate(vid_tube, axis=1)  # (1,T,H,W,C)
        vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))  # (1,T,C,H,W)
        # print(f"vid_tube.shape: {vid_tube.shape}")
        return torch.from_numpy(vid_tube).float()
    
    def _normalize(self, data):
        return (data / 255.0 - self.v_mean) / self.v_std    


    @register(dispatch_mode=Dispatch.ALL_TO_ALL)
    @WorkerProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        print(f"  -> Actor 'videoclip' CUDA_MPS_ACTIVE_THREAD_PERCENTAGE = {self.get_mps_percentage()}")
        start_time = time.time()
        # Read data without popping it
        datas=data.pop(
            batch_keys=['video_frames'],
            non_tensor_batch_keys=["caption"],
        )
        
        if self.is_active == False:
            empty_bs = datas.batch.batch_size[0] // self.videoclip_dp
            print(f"[RANK {self.rank}] VideoclipWorker is inactive, returning zeros")
            dummy_rewards = torch.zeros(empty_bs, device=torch.device('cpu'))
            return _make_reward_batch("videoclip_rewards", dummy_rewards, empty_bs)
        
        datas = _split_batch(datas, self.videoclip_dp, self.rank % self.videoclip_dp)
        print(f"[RANK {self.rank}] VideoclipWorker processing batch size: {datas.batch.batch_size[0]}")
        # (B, C, Frame, H, W)
        if datas.batch.batch_size[0]==0:
            print(torch.get.dis)
        decoded_images = _split_video_frames(datas, permute_to_tchw=False)

        caption = datas.non_tensor_batch['caption']       

        batch_caption = _split_captions(caption, datas.batch.batch_size[0])
        batch_indices = torch.chunk(torch.arange(len(batch_caption)), len(batch_caption))

        from teleboost.models.VideoCLIP_XL.utils.text_encoder import text_encoder
        all_rewards = []
        self.videoclip_model.to(get_device_id()).eval()
        print(f"videoclip batch size: {len(batch_caption)}")
        for index, batch_idx in enumerate(batch_indices):
            with torch.no_grad():
                video_inputs = self._video_preprocessing(decoded_images[index]).to(get_device_id())
                # print(f"video_inputs.shape: {video_inputs.shape}")
                video_features = self.videoclip_model.vision_model.get_vid_features(video_inputs).float()
                video_features = F.normalize(video_features, dim=-1)
                
                text_inputs = text_encoder.tokenize([batch_caption[index]], truncate=True).to(get_device_id())
                text_features = self.videoclip_model.text_model.encode_text(text_inputs).float()
                text_features = F.normalize(text_features, dim=-1)

                similarity = (video_features @ text_features.T) * 100
                similarity = similarity.view(-1)  # yields torch.Size([1])
                print(f"similarity_score value: {similarity}")
            all_rewards.append(similarity)
        
        all_rewards = torch.cat(all_rewards, dim=0)
        
        all_rewards = _zscore_tensor(all_rewards)
        
        all_rewards = all_rewards.to(torch.device('cpu'))
        batch = TensorDict(
            {
                "videoclip_rewards": all_rewards,
            },
            batch_size=len(batch_caption)
        )
        self.videoclip_model.to(torch.device('cpu'))
        # non_tensor_batch = data.non_tensor_batch
       
        end_time = time.time()
        print(f"videoclip compute time: {end_time - start_time}s") 
                
        # return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
        return DataProto(batch=batch)

# Videophy CAPTION
CAPTION = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <|video|>\nHuman: Does this video follow the physical laws?\nAI: "

# VideophyRewardModelWorker is a worker that computes video-phy scores for images.
class VideophyRewardModelWorker(RewardModelWorker):
    """
    VideophyRewardModelWorker is a worker that computes video-phy scores for images.
    It uses a pre-trained model to evaluate the video-phy quality of images.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self, media_tokens = ["<image>", "<|video|>"]):
        self.is_active = False
        self.videophy_dp = torch.distributed.get_world_size()
        # self.rank = dist.get_rank()
        if self.rank < self.videophy_dp:
            self.is_active = True
            # self.checkpoint = "/nvfile-heatstorage/tele_data_share/wyb/model/arena_models/videocon_physics"
            videophy_cfg = self.config.get("videophy", {}) or {}
            self.checkpoint = videophy_cfg.get(
                "model_path",
                "/gemini/space/wyb/model/arena_model/videocon_physics",
            )
            self.max_length = 256
            self._build_model()
            self.media_tokens = {k: -int(i + 1) for i, k in enumerate(media_tokens)}
            self.media_lengths = {"<image>": 1 + 64, "<|video|>": 1 + 64}   
        else:
            print(f"[RANK {self.rank}] VideophyWorker is inactive")  
           
        
    def _build_model(self):

        from transformers.models.llama.tokenization_llama import LlamaTokenizer
        from teleboost.models.Videophy.mplug_owl_video import MplugOwlForConditionalGeneration
        from teleboost.models.Videophy.mplug_owl_video import (
            MplugOwlImageProcessor,
            MplugOwlProcessor,
        )
        from teleboost.models.Videophy.mplug_owl_video import MplugOwlConfig        
        self.tokenizer = LlamaTokenizer.from_pretrained(self.checkpoint)
        print("Model Loading")
        self.videophy_model = MplugOwlForConditionalGeneration.from_pretrained(
            self.checkpoint,
            torch_dtype=torch.bfloat16,
            config=MplugOwlConfig.from_pretrained(self.checkpoint)
        )
        print("Model Loaded")
        self.videophy_model.eval()
        image_processor = MplugOwlImageProcessor.from_pretrained(self.checkpoint)
        self.processor = MplugOwlProcessor(image_processor, self.tokenizer)
        
        num_params_1 = sum(p.numel() for p in self.videophy_model.parameters())
        # num_params_2 = sum(p.numel() for p in image_processor.parameters())
        print(f"videophy_model params: {num_params_1}")             

    def _extract_text_token_from_conversation(self, max_length, index):  # index
        # output enc_chunk
        enc_chunk = []

        if self.tokenizer.bos_token_id > 0:
            prompt_chunk = [self.tokenizer.bos_token_id]
        else:
            prompt_chunk = []

        # conversation = data["completion"]
        conversation = CAPTION

        # For Text only data
        if all(
            [
                media_token not in conversation
                for media_token in self.media_tokens.keys()
            ]
        ):
            pattern = "|".join(map(re.escape, ["AI: ", "\nHuman: "]))
            chunk_strs = re.split(f"({pattern})", conversation)
            prompt_length = -1
            stop_flag = False
            for idx, chunk_str in enumerate(chunk_strs):
                if idx == 0:
                    enc_chunk = (
                        prompt_chunk
                        + self.tokenizer(chunk_str, add_special_tokens=False)[
                            "input_ids"
                        ]
                    )
                    enc_length = len(enc_chunk)
                    label_chunk = [0] * enc_length
                else:
                    if chunk_strs[idx - 1] == "AI: ":
                        curr_chunk = self.tokenizer(
                            chunk_str, add_special_tokens=False
                        )["input_ids"]
                        if enc_length + len(curr_chunk) >= max_length:
                            curr_chunk = curr_chunk[: max_length - enc_length]
                            stop_flag = True
                        curr_chunk += [self.tokenizer.eos_token_id]
                        enc_length += len(curr_chunk)
                        enc_chunk += curr_chunk
                        label_chunk += [1] * len(curr_chunk)
                    else:
                        curr_chunk = self.tokenizer(
                            chunk_str, add_special_tokens=False
                        )["input_ids"]
                        if enc_length + len(curr_chunk) >= max_length + 1:
                            curr_chunk = curr_chunk[: max_length + 1 - enc_length]
                            stop_flag = True
                        enc_length += len(curr_chunk)
                        enc_chunk += curr_chunk
                        label_chunk += [0] * len(curr_chunk)
                    if stop_flag:
                        break

        # For Image-Text Data
        else:
            enc_length = 0
            prompt_length = -2
            pattern = "|".join(
                map(re.escape, list(self.media_tokens.keys()) + ["AI: ", "\nHuman: "])
            )
            chunk_strs = re.split(f"({pattern})", conversation)
            chunk_strs = [x for x in chunk_strs if len(x) > 0]
            for idx, chunk_str in enumerate(chunk_strs):
                if enc_length >= max_length + 1:
                    break

                if idx == 0:
                    enc_chunk = (
                        prompt_chunk
                        + self.tokenizer(chunk_str, add_special_tokens=False)[
                            "input_ids"
                        ]
                    )
                    enc_length = len(enc_chunk)
                    label_chunk = [0] * enc_length
                else:
                    if chunk_str in self.media_tokens:
                        # [CLS] + 256 + [EOS]
                        if enc_length + self.media_lengths[chunk_str] > max_length + 1:
                            break
                        else:
                            enc_chunk += [
                                self.media_tokens[chunk_str]
                            ] * self.media_lengths[chunk_str]
                            enc_length += self.media_lengths[chunk_str]
                            label_chunk += [0] * self.media_lengths[chunk_str]
                    else:
                        if chunk_strs[idx - 1] == "AI: ":
                            curr_chunk = self.tokenizer(
                                chunk_str, add_special_tokens=False
                            )["input_ids"]
                            if enc_length + len(curr_chunk) >= max_length:
                                curr_chunk = curr_chunk[: max_length - enc_length]
                            curr_chunk += [self.tokenizer.eos_token_id]
                            enc_length += len(curr_chunk)
                            enc_chunk += curr_chunk
                            label_chunk += [1] * len(curr_chunk)
                        else:
                            curr_chunk = self.tokenizer(
                                chunk_str, add_special_tokens=False
                            )["input_ids"]
                            if enc_length + len(curr_chunk) >= max_length + 1:
                                curr_chunk = curr_chunk[: max_length + 1 - enc_length]
                            enc_length += len(curr_chunk)
                            enc_chunk += curr_chunk
                            label_chunk += [0] * len(curr_chunk)

        if enc_length < max_length + 1:
            padding_chunk = [self.tokenizer.pad_token_id] * (
                max_length + 1 - enc_length
            )
            padding_length = len(padding_chunk)
            label_chunk += [0] * (max_length + 1 - enc_length)
            enc_chunk = enc_chunk + padding_chunk
        else:
            padding_length = 0

        assert enc_length + padding_length == max_length + 1, (
            index,
            prompt_length,
            enc_length,
            padding_length,
            max_length + 1,
        )
        assert len(label_chunk) == max_length + 1, (len(label_chunk), max_length + 1)
        non_padding_mask = [1 if i < enc_length - 1 else 0 for i in range(max_length)]

        enc_chunk = torch.tensor(enc_chunk).long()
        non_padding_mask = torch.tensor(non_padding_mask).long()
        prompt_mask = torch.tensor(label_chunk)[1:].long()
        prompt_length = torch.tensor([prompt_length]).long()

        # Create loss mask
        if all(
            [
                media_token not in conversation
                for media_token in self.media_tokens.keys()
            ]
        ):
            non_media_mask = torch.ones_like(non_padding_mask).long()
        else:
            tmp_enc_chunk = enc_chunk.clone()
            tmp_enc_chunk[tmp_enc_chunk >= 0] = 1
            tmp_enc_chunk[tmp_enc_chunk < 0] = 0
            non_media_mask = torch.tensor(tmp_enc_chunk).long()
            non_media_mask = non_media_mask[1:].long()
        return {
            "input_ids": enc_chunk,
            "prompt_length": prompt_length,
            "seq_length": enc_length,
            "non_padding_mask": non_padding_mask,
            "non_media_mask": non_media_mask,
            "prompt_mask": prompt_mask,
        }

    def _get_input(self, batch):
        # TODO: batch_size > 1

        video = [data["video"] if data["video"] is not None else None for data in batch]
        if all([img is None for img in video]):
            video = None
        else:
            video = torch.cat([img for img in video if img is not None], dim=0)
        num_videos_per_sample = torch.LongTensor(
            [
                data["video"].size(0) if data["video"] is not None else 0
                for data in batch
            ]
        )
        num_images_per_sample = torch.LongTensor([0 for data in batch])

        text = torch.stack(
            [torch.LongTensor(data["text"]["input_ids"]) for data in batch], dim=0
        )
        non_padding_mask = torch.stack(
            [torch.LongTensor(data["text"]["non_padding_mask"]) for data in batch],
            dim=0,
        )
        non_media_mask = torch.stack(
            [torch.LongTensor(data["text"]["non_media_mask"]) for data in batch], dim=0
        )
        prompt_mask = torch.stack(
            [torch.LongTensor(data["text"]["prompt_mask"]) for data in batch], dim=0
        )
        # videopaths = [data["videopath"] for data in batch]
        captions = [data["caption"] for data in batch]
        output_batch = {
            "pixel_values": None,
            "video_pixel_values": video,
            "input_ids": text.long(),
            "labels": text.long().clone(),
            "num_images": num_images_per_sample.long(),
            "num_videos": num_videos_per_sample.long(),
            "non_padding_mask": non_padding_mask.long(),
            "non_media_mask": non_media_mask.long(),
            "prompt_mask": prompt_mask.long(),
            # "videopaths": videopaths,
            "captions": captions,
        }

        return output_batch

    def get_entail(self, logits, input_ids):
        softmax = nn.Softmax(dim=2)
        logits = softmax(logits)
        token_id_yes = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        token_id_no = self.tokenizer.encode("No", add_special_tokens=False)[0]
        entailment = []
        for j in range(len(logits)):
            for i in range(len(input_ids[j])):
                if (
                    input_ids[j][i] == self.tokenizer.pad_token_id
                ):  # pad token if the answer is not present
                    i = i - 1
                    break
                elif i == len(input_ids[j]) - 1:
                    break
            score = logits[j][i][token_id_yes] / (
                logits[j][i][token_id_yes] + logits[j][i][token_id_no]
            )
            entailment.append(score)
        entailment = torch.stack(entailment)
        return entailment

    def get_scores(self, inputs):
        with torch.no_grad():
            # for index, inputs in tqdm(enumerate(dataloader)):
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    if v.dtype == torch.float:
                        inputs[k] = v.bfloat16()
                    inputs[k] = inputs[k].to(get_device_id())
                    # print(f'{k}: {v.shape}')
            # inputs["videophy_score"] = []
            # print("compute videophy")
            outputs = self.videophy_model(
                pixel_values=inputs["pixel_values"],
                video_pixel_values=inputs["video_pixel_values"],
                labels=None,
                num_images=inputs["num_images"],
                num_videos=inputs["num_videos"],
                input_ids=inputs["input_ids"],
                non_padding_mask=inputs["non_padding_mask"],
                non_media_mask=inputs["non_media_mask"],
                prompt_mask=inputs["prompt_mask"],
            )
            # print("compute videophy done")
            logits = outputs["logits"]
            entail_scores = self.get_entail(logits, inputs["input_ids"])
            # print(len(entail_scores))
            # for m in range(len(entail_scores)):
            #     inputs["videophy_score"].append(entail_scores[m].item())
            # print(f"Batch {index} Done")
            assert len(entail_scores) == 1
        return entail_scores[0].item()

    def resize_video_frames(self, video_tensor, target_size=(224, 224)):
        """
        video_tensor: torch.Tensor, shape [B, C, T, H, W]
        return: torch.Tensor, shape [B, C, T, 224, 224]
        """
        B, C, T, H, W = video_tensor.shape
        target_H, target_W = target_size
        # Flatten the time dim into the batch dim
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        video_tensor = video_tensor.reshape(B * T, C, H, W)  # [B*T, C, H, W]
        
        # -------------------- Step 1: Resize (preserve aspect, align shortest side) --------------------
        # Compute scale factor (preserves the original ratio)
        scale = max(target_H / H, target_W / W)
        new_H = int(round(H * scale))
        new_W = int(round(W * scale))

        resized = F.interpolate(video_tensor, size=(new_H, new_W), mode='bilinear', align_corners=False)

        # -------------------- Step 2: Center Crop --------------------
        top = (new_H - target_H) // 2
        left = (new_W - target_W) // 2
        cropped = resized[:, :, top:top+target_H, left:left+target_W]  # [B*T, C, target_H, target_W]

        # -------------------- Step 3: Restore video shape --------------------
        cropped = cropped.view(B, T, C, target_H, target_W).permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

        return cropped    
  
    @register(dispatch_mode=Dispatch.ALL_TO_ALL)
    @WorkerProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        print(f"  -> Actor 'videophy' CUDA_MPS_ACTIVE_THREAD_PERCENTAGE = {self.get_mps_percentage()}")
        start_time = time.time()
        # Read data without popping it
        datas=data.pop(
            batch_keys=['video_frames'],
            non_tensor_batch_keys=["caption"],
        )
        
        if self.is_active == False:
            empty_bs = datas.batch.batch_size[0] // self.videophy_dp
            print(f"[RANK {self.rank}] VideophyWorker is inactive, returning zeros")
            dummy_rewards = torch.zeros(empty_bs, device=torch.device('cpu'))
            batch = TensorDict(
                {
                    "videophy_rewards": dummy_rewards,
                },
                batch_size = empty_bs
            )
            return DataProto(batch=batch)
        
        datas = _split_batch(datas, self.videophy_dp, self.rank)
        print(f"[RANK {self.rank}] VideophyWorker processing batch size: {datas.batch.batch_size[0]}")
        # (B, C, Frame, H, W)
        decoded_image = datas.batch['video_frames']
        # [B, C, 224, 224]
        # Plan A
        # print("start resize video frame")
        decoded_image = self.resize_video_frames(decoded_image, target_size=(224, 224)) 
        # print("resize video frame done")
        # List of [(1, C, Frame, 224, 224),...,...]
        decoded_images = decoded_image.chunk(datas.batch.batch_size[0], dim=0)
        # print("decoded_image.chunk done")
        # List of [(C, Frame, 224, 224),...,...]
        # decoded_images = [x.squeeze(0) for x in decoded_images]

        caption = datas.non_tensor_batch['caption']       

        batch_caption = np.array_split(caption, datas.batch.batch_size[0])
        batch_caption = [str(x.squeeze(0)) for x in batch_caption]
        # print("str(x.squeeze(0) done")
        batch_indices = torch.chunk(torch.arange(len(batch_caption)), len(batch_caption))
        # print("torch.chunk(torch.arange(len(batch_caption)) done")

        start_time = time.time()
        self.videophy_model.to(get_device_id())
        load_time = time.time() - start_time
        print(f"load model time: {load_time}s")
        
        all_rewards = []
        print(f"videophy batch size: {len(batch_caption)}")
        for index, batch_idx in enumerate(batch_indices):
            with torch.no_grad():
                # print("start _extract_text_token_from_conversation")
                text_input = self._extract_text_token_from_conversation(
                    self.max_length, index
                )
                # print("done _extract_text_token_from_conversation")
                inputs = {
                    "video": decoded_images[index],
                    "text": text_input,
                    "caption": CAPTION,
                    # "video_path": "/nvfile-heatstorage/liangyzh/interns/zhangxin/videophy/arena-example/A_wooden_spoon_stirs_the_hot_soup_in_the_pot._1.mp4"
                }
                
                inputs = self._get_input([inputs])
                # print(f"inputs: {inputs}")
                score = self.get_scores(inputs)
                print(f"videophy_value: {score}")
            all_rewards.append(torch.tensor([score]))
        
        all_rewards = torch.cat(all_rewards, dim=0)
        
        # Z-score normalization
        mean = all_rewards.mean()
        std = all_rewards.std(unbiased=False)  # unbiased=False -> population variance
        all_rewards = (all_rewards - mean) / (std + 1e-8)  # epsilon avoids div-by-zero
                
        all_rewards = all_rewards.to(torch.device('cpu'))
        self.videophy_model.to("cpu")
        batch = TensorDict(
            {
                "videophy_rewards": all_rewards,
            },
            batch_size=len(batch_caption)
        )
        
        # non_tensor_batch = data.non_tensor_batch
        
        end_time = time.time()
        print(f"videophy compute time: {end_time - start_time}s")         
        
        # return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
        return DataProto(batch=batch)
 
class MultiRewardModelWorker(RewardModelWorker):
    """
    Aggregates the four RewardModelWorkers: Aesthetic, RAFT, Videoclip, Videophy.
    """
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # Init each reward worker
        self.aesthetic_worker = AestheticRewardModelWorker()
        self.raft_worker = RAFTRewardModelWorker()
        self.videoclip_worker = VideoclipRewardModelWorker()
        self.videophy_worker = VideophyRewardModelWorker()

        self.aesthetic_worker.init_model()
        self.raft_worker.init_model()
        self.videoclip_worker.init_model()
        self.videophy_worker.init_model()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @WorkerProfiler.annotate(color="purple")
    def compute_rm_score(self, data: DataProto):
        datas=data.pop(
            batch_keys=['video_frames'],
            non_tensor_batch_keys=["caption"],
        )        
        print(f"datas.batch['video_frames'].shape: {datas.batch['video_frames'].shape}")
        streams = [torch.cuda.Stream() for _ in range(4)]
        
        # Call compute_rm_score on each reward worker
        with torch.cuda.stream(streams[0]):
            aes_result = self.aesthetic_worker.compute_rm_score(datas)
        with torch.cuda.stream(streams[1]):
            raft_result = self.raft_worker.compute_rm_score(datas)
        with torch.cuda.stream(streams[2]):
            videoclip_result = self.videoclip_worker.compute_rm_score(datas)
        with torch.cuda.stream(streams[3]):
            videophy_result = self.videophy_worker.compute_rm_score(datas)
        torch.cuda.synchronize()

        aes_result = self.aesthetic_worker.compute_rm_score(datas)
        raft_result = self.raft_worker.compute_rm_score(datas)
        videoclip_result = self.videoclip_worker.compute_rm_score(datas)
        videophy_result = self.videophy_worker.compute_rm_score(datas)
        torch.cuda.synchronize()
        

        # Merge results
        batch = TensorDict(
            {
                "aes_rewards": aes_result.batch["aes_rewards"],
                "raft_rewards": raft_result.batch["raft_rewards"],
                "videoclip_rewards": videoclip_result.batch["videoclip_rewards"],
                "videophy_rewards": videophy_result.batch["videophy_rewards"],
            },
            batch_size=aes_result.batch.batch_size
        )
        non_tensor_batch = data.non_tensor_batch
        print(f"cuda worker batch['aes_rewards'].shape: {batch['aes_rewards'].shape}")
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

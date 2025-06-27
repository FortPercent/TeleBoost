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
import warnings

import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh

from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_tokenizer
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name, get_nccl_backend
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.import_utils import import_external_libs
from verl.workers.fsdp_workers import create_device_mesh, get_sharding_strategy, ActorRolloutRefWorker
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.utils.debug import ProfilerConfig, WorkerProfiler, WorkerProfilerExtension, log_gpu_memory_usage, simple_timer
from tqdm.auto import tqdm
import math
from diffusers.image_processor import VaeImageProcessor
from tensordict import TensorDict

from omegaconf import DictConfig, open_dict

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class PRIMERewardModelWorker(Worker):
    def __init__(self, config):
        super().__init__()
        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=get_nccl_backend())
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(get_device_name(), mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.mini_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size
            assert self.config.mini_batch_size % self.config.micro_batch_size_per_gpu == 0

    def _build_reward_ref_model_optimizer(self, config):
        # the following line is necessary
        from torch import optim
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision

        from verl.utils.model import print_model_size
        from verl.utils.torch_dtypes import PrecisionType

        local_path = copy_local_path_from_hdfs(config.model.path)

        tokenizer_path = copy_local_path_from_hdfs(config.model.tokenizer_path)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))

        from omegaconf import OmegaConf

        override_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f"Reward model overriding config {override_config_kwargs}")

        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig, AutoModelForCausalLM

        trust_remote_code = False
        reward_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        reward_model_config.num_labels = 1

        init_context = get_init_weight_context_manager(use_meta_tensor=not reward_model_config.tie_word_embeddings)
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reward_model_config.classifier_dropout = 0.0
            reward_model_config.hidden_dropout = "0"
            reward_module = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=reward_model_config,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            fused_kernel_options = config.model.get("fused_kernel_options", None)
            fused_kernels_backend = fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None

            apply_monkey_patch(
                model=reward_module,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_remove_padding=config.model.get("use_remove_padding", False),
                use_fused_kernels=config.model.get("use_fused_kernels", False),
                fused_kernels_backend=fused_kernels_backend,
            )

            # some parameters may not in torch_dtype
            reward_module.to(torch_dtype)

            if config.model.get("enable_gradient_checkpointing", False):
                reward_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if self.rank == 0:
            print_model_size(reward_module)

        self.reward_model_config = reward_model_config

        fsdp_config = self.config.model.fsdp_config
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

        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config.wrap_policy)

        log_gpu_memory_usage("Before reward model FSDP", logger=None)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reward_model_config.classifier_dropout = 0.0
            reward_model_config.hidden_dropout = "0"
            ref_module = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=copy_local_path_from_hdfs(config.model.ref_path),
                torch_dtype=torch_dtype,
                config=reward_model_config,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            # some parameters may not in torch_dtype
            ref_module.to(torch_dtype)

        reward_module = FSDP(
            reward_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=get_device_id(),
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            sync_module_states=True,
            forward_prefetch=False,
            device_mesh=self.device_mesh,
            cpu_offload=None,
        )

        log_gpu_memory_usage("After reward FSDP", logger=None)

        ref_module = FSDP(
            ref_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=get_device_id(),
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            sync_module_states=True,
            forward_prefetch=False,
            device_mesh=self.device_mesh,
            cpu_offload=None,
        )

        reward_optimizer = optim.AdamW(
            reward_module.parameters(),
            lr=config.model.optim.lr,
            betas=config.model.optim.get("betas", (0.9, 0.999)),
            weight_decay=config.model.optim.get("weight_decay", 1e-2),
        )

        total_steps = config.model.optim.get("total_training_steps", 0)
        num_warmup_steps = int(config.model.optim.get("lr_warmup_steps", -1))
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = config.model.optim.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        from verl.utils.torch_functional import get_constant_schedule_with_warmup

        reward_lr_scheduler = get_constant_schedule_with_warmup(optimizer=reward_optimizer, num_warmup_steps=num_warmup_steps)

        return reward_module, ref_module, reward_optimizer, reward_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from .prime_dp_rm import DataParallelPRIMERewardModel

        self.reward_module, self.ref_module, self.reward_optimizer, self.reward_lr_scheduler = self._build_reward_ref_model_optimizer(config=self.config)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.reward_module)
            offload_fsdp_model_to_cpu(self.ref_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.reward_optimizer)

        self.rm = DataParallelPRIMERewardModel(
            config=self.config,
            reward_module=self.reward_module,
            ref_module=self.ref_module,
            reward_optimizer=self.reward_optimizer,
        )

        self.flops_counter = FlopsCounter(self.reward_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.reward_module,
            optimizer=self.reward_optimizer,
            lr_scheduler=self.reward_lr_scheduler,
            tokenizer=self.tokenizer,
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.reward_module)

        self.checkpoint_manager.save_checkpoint(local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.reward_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, del_local_after_load=True):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.reward_module)

        self.checkpoint_manager.load_checkpoint(local_path=local_path, del_local_after_load=del_local_after_load)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.reward_module)
            
class DiffusionActorRolloutRefWorker(ActorRolloutRefWorker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, model_deployment=None):
        super().__init__(config,role)
        
        
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @WorkerProfiler.annotate(color="red")
    def generate_samples(self, prompts: DataProto):
        prompts = prompts.to(get_device_id())
        num_frames = self.config.num_frames
        size = (self.config.w, self.config.h)
        sample_steps = self.config.sampling_steps
        sigma_schedule = torch.linspace(1, 0, self.config.sampling_steps + 1)
        def sd3_time_shift(shift, num_frames):
            return (shift * num_frames) / (1 + (shift - 1) * num_frames)
    
        sigma_schedule = sd3_time_shift(self.config.shift, sigma_schedule)
        
        def assert_eq(x, y, msg=None):
            assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"

        context=prompts.batch['context']
        caption=prompts.non_tensor_batch['caption']
        B = len(caption)
        
        # B = caption.shape[0]
        assert_eq(
            len(sigma_schedule),
            sample_steps + 1,
            "sigma_schedule must have length sample_steps + 1",
        )

        all_latents = []
        all_log_probs = []
        all_rewards = []

        # VAE参数
        vae_stride = [4, 8, 8]
        patch_size = [1, 2, 2]
        # 根据是否使用FP16选择数据类型

        batch_size=1
        batch_indices = torch.chunk(torch.arange(B), B // batch_size)
        latent_dtype = torch.float16 #TODO
        if self.config.init_same_noise:
            latent_shape = (
                16,
                (num_frames - 1) // vae_stride[0] + 1,
                size[1] // vae_stride[1],
                size[0] // vae_stride[2]
            )
            input_latents = torch.randn(latent_shape, device=get_device_id(), dtype=latent_dtype)

        for index, batch_idx in enumerate(batch_indices):
            batch_captions = [caption[i] for i in batch_idx]
            batch_contexts = [context[i].to(get_device_id()) for i in batch_idx]
            
            if not self.config.init_same_noise:
                latent_shape = (
                    16,
                    (num_frames - 1) // vae_stride[0] + 1,
                    size[1] // vae_stride[1],
                    size[0] // vae_stride[2]
                )
                input_latents = torch.randn(latent_shape, device=get_device_id(), dtype=latent_dtype)

            seq_len = math.ceil(
                (latent_shape[2] * latent_shape[3]) / (patch_size[1] * patch_size[2]) * latent_shape[1]
            )

            grpo_sample = True
            progress_bar = tqdm(range(0, self.config.sampling_steps), desc="WAN Sampling Progress")
            
           
            with torch.no_grad():      
                _, final_latents, batch_latents, batch_log_probs = self.run_wan_sample_step(
                    [input_latents],
                    progress_bar,
                    sigma_schedule,
                    self.actor_module_fsdp,
                    batch_contexts,
                    seq_len,
                    grpo_sample,
                )

            batch_latents = batch_latents.unsqueeze(0)
            batch_log_probs = batch_log_probs.unsqueeze(0)

            all_latents.append(batch_latents)
            all_log_probs.append(batch_log_probs)

        if len(all_latents) > 1:
            all_latents = torch.cat(all_latents, dim=0)
            all_log_probs = torch.cat(all_log_probs, dim=0)
        else:
            all_latents = all_latents[0]
            all_log_probs = all_log_probs[0]      

        batch = TensorDict(
            {
                "latents": all_latents,
                "rollout_log_probs": all_log_probs,  # we will recompute old log prob with actor
            },
            batch_size=B
        )

        non_tensor_batch = prompts.non_tensor_batch
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


    def run_wan_sample_step(
        self,
        latents,  # [(16, 7, 64, 64)]
        progress_bar, 
        sigma_schedule,  # 添加sigma_schedule
        transformer,
        context,
        seq_len,
        grpo_sample,
    ):
        """WAN采样步骤，支持(C,T,H,W)格式输入"""
        if grpo_sample:
            all_latents = [latents[0]]  # 存储初始latent (16, 7, 64, 64)
            all_log_probs = []
            
            for i in progress_bar:
                B = len(context) if isinstance(context, list) else context.shape[0]
                
                # 确保设备一致
                device = latents[0].device
                
                # 使用sigma值计算timestep
                sigma = sigma_schedule[i]
                timestep_value = int(sigma * 1000)
                timestep = torch.full([B], timestep_value, device=device, dtype=torch.long)
                
                transformer.eval()
                with torch.autocast("cuda", torch.float32):
                    # WAN模型输入：x是(C,T,H,W)格式的列表

                    #TODO!!!!
                    #管理!!!
                    transformer.to(device)

                    pred = transformer(
                        x=latents,  # [tensor(16, 7, 64, 64)]
                        t=timestep,
                        context=context,
                        seq_len=seq_len
                    )
                    
                    # 处理模型输出
                    if isinstance(pred, dict) and 'rgb' in pred:
                        model_output = pred['rgb'][0]
                    elif isinstance(pred, list):
                        model_output = pred[0]
                    else:
                        model_output = pred

                # WAN的SDE采样步骤
                next_latents, pred_original, log_prob = self.wan_step(
                    model_output, 
                    latents[0].to(torch.float32),  # (16, 7, 64, 64)
                    self.config.eta, 
                    sigma_schedule,  # 传入sigma_schedule
                    i, 
                    prev_sample=None, 
                    grpo=True, 
                    sde_solver=True  # 启用SDE求解器
                )
                
                latents = [next_latents.to(torch.float32)]  # [(16, 7, 64, 64)]
                all_latents.append(latents[0])  # 存储 (16, 7, 64, 64)
                all_log_probs.append(log_prob)  # 存储 log概率
            
            final_latents = pred_original
                    # all the tp ranks should contain the same data here. data in all ranks are valid

            # 修正：WAN的all_latents维度是 (num_steps+1, 16, 7, 64, 64)
            all_latents = torch.stack(all_latents, dim=0)  # (9, 16, 7, 64, 64)
            all_log_probs = torch.stack(all_log_probs, dim=0)  # (8, B) -> (8,)
            
            # print(f"WAN after stack: all_latents={all_latents.shape}, all_log_probs={all_log_probs.shape}")
            # (9, 16, 7, 64, 64), (8,)
            
            return latents, final_latents, all_latents, all_log_probs

    def wan_step(
        self,
        model_output: torch.Tensor,  # 模型预测的flow
        latents: torch.Tensor,       # 当前时间步的潜在表示 (16, 7, 64, 64)
        eta: float,                  # 控制随机性强度
        sigmas: torch.Tensor,        # sigma调度序列 (类似FLUX)
        index: int,                  # 当前时间步索引  
        prev_sample: torch.Tensor,   # 前一步的样本（用于GRPO重计算）
        grpo: bool,                  # True时会得到logprob
        sde_solver: bool,            # 使用SDE求解器
    ):
        """WAN的Flow Matching采样步骤，转换为SDE求解器支持GRPO"""
        
        sigma = sigmas[index]
        dsigma = sigmas[index + 1] - sigma  # sigma差分
        
        # 确定性更新部分
        prev_sample_mean = latents + dsigma * model_output
        
        # 预测的原始样本
        pred_original_sample = latents - sigma * model_output
        
        delta_t = sigma - sigmas[index + 1]  # 时间差分
        std_dev_t = eta * math.sqrt(abs(delta_t))  # 随机噪声的std
        
        if sde_solver:  # 使用SDE求解器（和FLUX相同）
            score_estimate = -(latents - pred_original_sample * (1 - sigma)) / (sigma**2)  # 估计的得分
            log_term = -0.5 * eta**2 * score_estimate  # 对数项修正
            prev_sample_mean = prev_sample_mean + log_term * dsigma  # 修正的均值
        
        if grpo and prev_sample is None:
            prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

        if grpo:
            # 计算log概率
            log_prob = (
                -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2))
            )
            - math.log(std_dev_t + 1e-8) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))

            # 在除batch维度外的所有维度上求平均
            log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
            return prev_sample, pred_original_sample, log_prob
        else:
            return prev_sample_mean, pred_original_sample
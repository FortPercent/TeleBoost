import itertools
import math
import logging
from collections import defaultdict
import os
import numpy as np
import torch

from recipe.spin.core_algos import compute_online_dpo_loss, get_batch_logps
from verl import DataProto
from verl.utils.device import get_device_name
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.workers.actor import DataParallelPPOActor
from verl.utils.ulysses import gather_outpus_and_unpad

from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id, get_device_name, get_nccl_backend
# from tensorwatch import TensorWatch,watch_module_forward_backward
__all__ = ["DiffusionDataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# hook 函数：保存输入数据和模块名
def save_input_hook(name):
    def hook(module, input, output):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        file_name = f"output/{rank}_{name}_input.pt"
        for param_name, param in module.named_parameters(recurse=False):
            print(f"  Param: {name}, Shape: {param.shape},{param.float().norm().item()}")

        # print(f"[HOOK] Saving {name} input to {file_name}")
        torch.save({
            "name": name,
            "input": input,
            "output": output
        }, file_name)
    return hook

def register_all_hooks(model):
    for name, module in model.named_modules():
        print(f"[REGISTER] Hooking module: {name}")
        module.register_forward_hook(save_input_hook(name))

class DiffusionDataParallelPPOActor(DataParallelPPOActor):

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        #这里显存就开始增加了
        print("in update policy",self.actor_module.dtype)
        data = data.to(get_device_id())
        # make sure we are in training mode
        self.actor_module.train()
 
        device = torch.device(f"cuda:{get_device_id()}")
        perms = torch.stack([
            torch.randperm(len(data.batch["timesteps"][0])) 
            for _ in range(data.batch.batch_size[0])
        ]).to(device)
        for key in ["timesteps", "latents", "next_latents", "log_probs"]:
            data.batch[key] = data.batch[key][
                torch.arange(data.batch.batch_size[0]).to(device)[:, None],
                perms,
            ]
        
        # print(data.batch["timesteps"].shape,type(data.batch["timesteps"]),data.batch["timesteps"][0].shape,type(data.batch["timesteps"][0]),len(data.batch["timesteps"][0].shape[1]))
        train_timesteps = int(len(data.batch["timesteps"][0]) * self.config.timestep_fraction)
        grad_norm = None
        
        # num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size

        select_keys=["timesteps", "latents", "next_latents", "log_probs","contexts","sigma_schedule","advantages","context_orig_lengths","null_context"]
        non_tensor_select_keys = ["caption"]

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        
        # from tensordict import TensorDict

        # batch = TensorDict()
        # for k, v in data.batch.items():
        #     if isinstance(v, torch.Tensor):
        #         batch[k] = v.unsqueeze(1)  # 添加一维，保持 [B, 1, ...] 结构
        # batch.batch_size=data.batch.batch_size


        # 合并到 data 中（假设 DataProto 接收一个包含 batch 的 dict）
        # data.batch=batch
        self.gradient_accumulation = (
                self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
            )
        dataloader = data.select(select_keys, non_tensor_select_keys).chunk(data.batch.batch_size[0])
        for batch_idx, data in enumerate(dataloader):
            # mini_batch = data
            # # split batch into micro_batches
            # micro_batches = mini_batch.chunk(self.config.ppo_micro_batch_size_per_gpu)

            self.actor_optimizer.zero_grad()
            batch_contexts = [data.batch["contexts"][i] for i in range(len(data))]
            batch_null_contexts = [data.batch["null_context"][i] for i in range(len(data))]

            for i in range(len(data)):
                orig_lengths =int(data.batch['context_orig_lengths'][i])
                assert batch_contexts[i].shape[0] >= orig_lengths, \
                    f"Context length mismatch: expected at least {orig_lengths}, but got {data.batch['contexts'][i].shape[0]}. Caption: {data.non_tensor_batch['caption']}"
                batch_contexts[i] = batch_contexts[i][:orig_lengths]
                
            for step_idx in range(train_timesteps):
                clip_range = self.config.clip_range
                adv_clip_max = self.config.adv_clip_max
                
                latent_shape = data.batch["latents"][:, step_idx].shape
                seq_len = math.ceil(
                    (latent_shape[3] * latent_shape[4]) / (2 * 2) * latent_shape[2]
                )
    

                new_log_probs = self.grpo_wan_one_step(
                    data.batch["latents"][:, step_idx],
                    data.batch["next_latents"][:, step_idx],
                    batch_contexts,  # List[Tensor]格式
                    batch_null_contexts,
                    seq_len,
                    self.actor_module,
                    data.batch["timesteps"][:, step_idx],
                    perms[batch_idx][step_idx],
                    data.batch["sigma_schedule"][0], 
                )

                # print(new_log_probs)
                # exit(0)
                # 其余训练逻辑保持不变...
                advantages = torch.clamp(
                    data.batch["advantages"],
                    -adv_clip_max,
                    adv_clip_max,
                )
                # print("adv_clip_max",adv_clip_max)
                # file_name = f"/nvfile-heatstorage/teleai-infra/wxe/dancegrpo_aigc/debug_current_log_probs.pt"
                # # print(file_name)
                # batch = torch.load(file_name)
                # current_log_probs=batch["current_log_probs"]
                # advantages=batch["advantages"]
                ratio = torch.exp(new_log_probs - data.batch["log_probs"][:, step_idx])
                # ratio = torch.exp(new_log_probs - current_log_probs)
                
                unclipped_loss = -advantages * ratio
                clipped_loss = -advantages * torch.clamp(
                    ratio,
                    1.0 - clip_range,
                    1.0 + clip_range,
                )
                # print("current_log_probs",current_log_probs)
                print("new_log_probs",new_log_probs)
                print("ratio",ratio)
                print("advantages",advantages)
                print("Uncipped Loss:", unclipped_loss)
                print("Clipped Loss:", clipped_loss)
                print("Maximum Loss:", torch.maximum(unclipped_loss, clipped_loss))
                # print("args.gradient_accumulation_steps",self.gradient_accumulation)
                # print("train_timesteps",train_timesteps)
                loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (self.gradient_accumulation * train_timesteps)
                # loss = torch.mean(unclipped_loss)/(self.gradient_accumulation * train_timesteps)
                # print("-"*100)
                # print("loss",loss)
                loss.backward()
                # with FSDP.summon_full_params(self.actor_module,writeback=False,   # 不回写
                #     with_grads=False,   # 关键：还原 grad
                #     offload_to_cpu=False,  # 需要的话可 True 省显存
                #     rank0_only=True):
                #     for name, param in self.actor_module.named_parameters():
                #         print("for post backward check", name, param.float().norm().item())

                # # torch.save(param, f"saved_params_for_check/{name}.pt")
                # param=float(param.float().norm())
                # exit(0)
                # TensorWatch.step()
  
                avg_loss = loss.detach().clone()

                torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.AVG)
                # total_loss += avg_loss.item()
                
            if (batch_idx + 1) % self.gradient_accumulation == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()
                self.actor_optimizer.zero_grad()
                #self.actor_lr_scheduler.step() constant

    def grpo_wan_one_step(
        self,
        latents,
        pre_latents,
        context,
        context_null,
        seq_len,
        transformer,
        timesteps,
        i,
        sigma_schedule,
        guide_scale=5.0,
    ):
        """GRPO的单步训练，支持FP16优化"""
        B = len(context) if isinstance(context, list) else context.shape[0]
        transformer.train()
        
        # 确保latents维度正确：(16, 7, 64, 64)
        if latents.dim() == 5:
            latents = latents.squeeze(0)
        
        if pre_latents.dim() == 5:
            pre_latents = pre_latents.squeeze(0)
        
        if latents.shape[0] != 16:
            raise ValueError(f"Expected 16 channels, got {latents.shape[0]} channels")
        

        autocast_dtype = torch.bfloat16
        with torch.autocast("cuda", dtype=autocast_dtype):
            # 加载保存的 tensor 和数据
            # file_name = f"/nvfile-heatstorage/teleai-infra/wxe/dancegrpo_aigc/0_latent_timestep_data.pt"
            # print(file_name)
            # data = torch.load(file_name)

            # latents = data["latents"]      # torch.Tensor
            # timesteps = data["timestep"]     # torch.Tensor
            # seq_len = data["seq_len"]          # int
            # context = data["context"]       # list of tensor 或 tensor
            # pre_latents = data["pre_latents"]
            # context_null = data["context_null"]
            # sigma_schedule = data["sigma_schedule"]

            # print("latents:", latents.shape)
            # print("timestep:", timesteps)
            # print("seq_len:", seq_len)
            # print(len(context))
            latents.to(pre_latents.device)

            # torch.manual_seed(42)

            # watch_module_forward_backward(transformer, use_megatron=False, use_deepspeed=False,use_fsdp=True)

            # print("come here!!!!")
            # register_all_hooks(transformer)
            
            pred_cond = transformer(
                x=[latents],
                t=timesteps,
                context=context,
                seq_len=seq_len
            )
            # TensorWatch.step()
            # file_name = f"{torch.distributed.get_rank()}_pred_cond_debug_tensors.pt"
            # print(file_name)
            # torch.save(pred, file_name)
            # exit(0)
            
            # 处理条件预测输出
            if isinstance(pred_cond, dict) and 'rgb' in pred_cond:
                model_output_cond = pred_cond['rgb'][0]
            elif isinstance(pred_cond, list):
                model_output_cond = pred_cond[0]
            else:
                model_output_cond = pred_cond
                
            # 立即清理
            del pred_cond
            torch.cuda.empty_cache()
                
            # 再计算无条件预测
            pred_uncond = transformer(
                x=[latents],  # 保持List[Tensor]格式
                t=timesteps,
                context=context_null,  # List[Tensor]
                seq_len=seq_len
            )

                
            # 处理无条件预测输出
            if isinstance(pred_uncond, dict) and 'rgb' in pred_uncond:
                model_output_uncond = pred_uncond['rgb'][0]
            elif isinstance(pred_uncond, list):
                model_output_uncond = pred_uncond[0]
            else:
                model_output_uncond = pred_uncond
                
            del pred_uncond
            torch.cuda.empty_cache()
            
            # CFG组合
            model_output = model_output_uncond + guide_scale * (model_output_cond - model_output_uncond)
            del model_output_cond, model_output_uncond


        # 确保数据类型一致性

        computation_dtype = torch.float32
        _, _, log_prob = self.wan_step(
            model_output.to(computation_dtype), 
            latents.to(computation_dtype), 
            self.config.eta, 
            sigma_schedule,
            i, 
            prev_sample=pre_latents.to(computation_dtype), 
            grpo=True, 
            sde_solver=True
        )
        
        return log_prob

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
        print("model_output",model_output.float().norm())
        print("latents",latents.float().norm())
        print("eta",eta)
        print("sigmas",sigmas)
        index=0
        print("index",index)
        sigma = sigmas[index]
        dsigma = sigmas[index + 1] - sigma  # sigma差分
        
        # 确定性更新部分
        prev_sample_mean = latents + dsigma * model_output
        
        # 预测的原始样本
        pred_original_sample = latents - sigma * model_output
        
        delta_t = sigma - sigmas[index + 1]  # 时间差分
        # std_dev_t = eta * math.sqrt(abs(delta_t))  # 随机噪声的std
        std_dev_t = eta * torch.sqrt(delta_t)  # 根据hunyuan改的
        
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


import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def print_full_grads_per_param(model: FSDP):
    """
    遍历 FSDP 包装的模型，并打印每个原始参数的完整梯度范数。
    警告：这个函数会进行通信和显存分配，只用于调试。
    """
    if dist.get_rank() != 0:
        return

    world_size = dist.get_world_size()

    print("\n--- Rank 0: Printing full gradients per original parameter ---")

    # 遍历 FSDP 模型中的所有 FSDP 包装的模块
    for fsdp_module in model.modules():
        print(isinstance(fsdp_module, FSDP),fsdp_module)
        if isinstance(fsdp_module, FSDP):
            # 获取 FSDP 模块内部的原始参数列表
            # FSDP 在内部维护了一个 `flat_param`，但我们仍然可以访问原始参数的元数据
            
            # 这一步是关键：我们需要获取每个参数的梯度分片
            # 我们直接从 FSDP 模块内部的参数列表来获取
            # `fsdp_module.params` 包含了所有原始参数的引用
            for param_name, param in fsdp_module.named_parameters(recurse=False):
                # param.grad 此时就是分片梯度
                if param.grad is not None:
                    # 创建一个列表来收集所有分片
                    grad_shard = param.grad.data
                    grad_list = [torch.zeros_like(grad_shard) for _ in range(world_size)]
                    
                    # 执行 all_gather
                    dist.all_gather(grad_list, grad_shard)
                    
                    # 将分片拼接成完整的梯度
                    # FSDP 默认沿第一个维度分片，因此我们用 torch.cat(..., dim=0)
                    full_grad = torch.cat(grad_list, dim=0)
                    
                    # 打印结果
                    print(f"  > FSDP Module: {fsdp_module.name}, Param: {param_name}, Full Grad Norm: {full_grad.norm().item()}")

    print("--- Rank 0: Finished printing full gradients ---\n")

# 在你的训练循环中调用它
# print_full_grads_per_param(fsdp_model)
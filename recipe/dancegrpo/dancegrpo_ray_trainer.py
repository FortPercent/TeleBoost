import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
import time

# from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from verl import DataProto
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer, apply_kl_penalty, compute_response_mask
from verl.utils.debug import marked_timer
from verl.utils.device import get_device_id, get_device_name, get_nccl_backend


def fprint(*args, **kwargs):
    text = " ".join(str(a) for a in args)
    with open("output.log", "a", encoding="utf-8") as f:
        f.write(text + "\n")

# 这里计算优势函数时，直接使用组内标准化的奖励作为优势
def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, config=None):
    datas=data.pop(
        batch_keys=['rewards'],
    )
    advantages=torch.zeros_like(datas.batch['rewards'])
    #TODO when batchsize not equal to 1
    group_mean = datas.batch['rewards'].mean()
    group_std = datas.batch['rewards'].std() + 1e-8
    advantages = (datas.batch['rewards'] - group_mean) / group_std
    data.batch["advantages"] = advantages
    return data

from typing import List
import torch
import numpy as np

@staticmethod
def merge_worker_results(data_list: List[DataProto], dummy_key="dummy_rewards") -> DataProto:
    """
    拼接多个 DataProto。
    对于每个数据键(key)，只拼接那些自身包含非零元素的 tensor/array。
    完全由零组成的 tensor/array 会被忽略。
    
    Args:
        data_list (List[DataProto]): 多个 DataProto
        dummy_key (str): (此参数在当前逻辑中未使用)
    
    Returns:
        DataProto: 拼接后的 DataProto
    """
    if not data_list:
        return DataProto()
    
    # 1. 收集所有唯一的 key
    all_batch_keys = set() # 初始化空的集合 set会自动去重 但是List不会
    all_non_tensor_keys = set() # 返回的dataproto包当中的key里面可能会有重复的
    for dp in data_list:
        if dp.batch is not None:
            all_batch_keys.update(dp.batch.keys())
        if dp.non_tensor_batch is not None:
            all_non_tensor_keys.update(dp.non_tensor_batch.keys())

    # 2. 处理 batch (PyTorch tensors)
    batch_dict = {}
    for key in all_batch_keys:
        # 准备一个列表，只存放“有意义的”（非全零）tensors
        tensors_to_concat = []
        for dp in data_list:
            if dp.batch is not None and key in dp.batch:
                tensor = dp.batch[key]
                # 关键改动：检查这一个 tensor 是否包含非零元素
                if torch.any(tensor != 0):# any任意一个元素不为0返回true
                    # 如果包含，则将其加入待拼接列表
                    tensors_to_concat.append(tensor)
        
        # 如果列表不为空（即至少有一个 tensor 是有意义的），则执行拼接
        if tensors_to_concat:
            batch_dict[key] = torch.cat(tensors_to_concat, dim=0)
            
    # 3. 处理 non_tensor_batch (NumPy arrays)
    non_tensor_dict = {}
    for key in all_non_tensor_keys:
        # 准备一个列表，只存放“有意义的”（非全零）arrays
        arrays_to_concat = []
        for dp in data_list:
            if dp.non_tensor_batch is not None and key in dp.non_tensor_batch:
                arr = dp.non_tensor_batch[key]
                # 关键改动：检查这一个 array 是否包含非零元素
                if np.any(arr != 0):
                    # 如果包含，则将其加入待拼接列表
                    arrays_to_concat.append(arr)
        
        # 如果列表不为空，则执行拼接
        if arrays_to_concat:
            non_tensor_dict[key] = np.concatenate(arrays_to_concat, axis=0)
            
    return DataProto.from_dict(tensors=batch_dict, non_tensors=non_tensor_dict) # 转换成标准的DataProto格式

class RayDanceGRPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """
    #TODO dataset!!!! DataProto
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

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

        # perform validation before training
        # currently, we only support validation using the reward_function.
        # if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
        #     val_metrics = self._validate()
        #     assert val_metrics, f"{val_metrics=}"
        #     pprint(f"Initial validation metrics: {val_metrics}")
        #     logger.log(data=val_metrics, step=self.global_steps)
        #     if self.config.trainer.get("val_only", False):
        #         return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        
        if self.use_rm and self.config.reward_model.type=="joint":
            import threading
            reward_results = {}
            thread_inputs = {}
            ready_events = {}
            done_events = {}

            def thread_loop(name, worker):
                while True:
                    ready_events[name].wait()      # 等待主线程喂数据
                    ready_events[name].clear()
                    # 调用现有 worker 的 compute_rm_score
                    reward_results[name] = worker.compute_rm_score(thread_inputs[name])
                    done_events[name].set()        # 通知主线程完成

            # 四个线程
            workers = {
                "aes": self.aes_rm_wg,
                "raft": self.raft_rm_wg,
                "videoclip": self.videoclip_rm_wg,
                "videophy": self.videophy_rm_wg,
            }

            threads = []
            for name, worker in workers.items():
                reward_results[name] = None
                thread_inputs[name] = None
                ready_events[name] = threading.Event()
                done_events[name] = threading.Event()
                t = threading.Thread(target=thread_loop, args=(name, worker), daemon=True)
                t.start()
                threads.append(t)       
        
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
            
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                
                # pop those keys for generation TODO!!!
                # trainer的类型是diffusion
                if self.config.trainer.type=="diffusion":
                    # print("non-tensor keys:", new_batch.non_tensor_batch_keys.keys())
                    gen_batch = new_batch.pop(
                        batch_keys=["context","context_orig_lengths","null_context"],
                        non_tensor_batch_keys=["caption"],
                    )
                    
                    # 形状与配置
                    B = new_batch.batch.batch_size[0]
                    S = self.config.actor_rollout_ref.sampling_steps
                    num_frames = self.config.actor_rollout_ref.num_frames
                    size = (self.config.actor_rollout_ref.w, self.config.actor_rollout_ref.h)
                    vae_stride = [4, 8, 8] # VAE 
                    latent_dtype = torch.float32
                    latent_shape = (
                        16, # 这里的16指的是什么
                        (num_frames - 1) // vae_stride[0] + 1,
                        size[1] // vae_stride[1],
                        size[0] // vae_stride[2],
                    )

                    # 预分配批量张量
                    input_latents = torch.empty((B, *latent_shape), dtype=latent_dtype)
                    sigma_schedule_B = torch.empty((B, S+1), dtype=torch.float32)

                    def sd3_time_shift(shift, x):
                        return (shift * x) / (1 + (shift - 1) * x)

                    for i in range(new_batch.batch.batch_size[0]):  # 注意：这里用 gen_batch，不是 new_batch
                        sigma_schedule = torch.linspace(1, 0, S + 1)
                        
                        sigma_schedule = sd3_time_shift(self.config.actor_rollout_ref.shift, sigma_schedule)   # [S+1]
                        
                        sigma_schedule_B[i] = sigma_schedule

                        input_latents[i] = torch.randn(latent_shape, dtype=latent_dtype)

                    gen_batch.batch["input_latents"]  = input_latents
                    gen_batch.batch["sigma_schedule"] = sigma_schedule_B
                    
                    gen_batch = gen_batch.repeat(self.config.actor_rollout_ref.rollout.n) # 采样的次数 重复n

                elif "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw):
                        # gen_batch_output的数据类型是DataProto
                        # 具体见DiffusionActorRolloutWorker.generate_sequences方法
                        # 得到的gen_batch_output是聚合所有gpu的结果
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch) # 这里生成视频

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

                    new_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                    
                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with marked_timer("validation", timing_raw):
                            from verl.utils.checkpoint.checkpoint_manager import save_video_and_prompt
                            video_frames = gen_batch_output.batch["video_frames"] 
                            for i in range(video_frames.shape[0]):
                                save_video_and_prompt(video_frames[i] , 0, i)
                    # repeat to align with repeated responses in rollout
                    # new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            print("begin to compute reward")
                            with torch.amp.autocast('cuda'):

                                if self.config.reward_model.type == "joint":
                                    # ======================
                                    # Joint Reward Models
                                    # ======================
                                    import time
                                    start_time = time.time()

                                    # 启动所有 RM 线程
                                    for name in workers:
                                        thread_inputs[name] = gen_batch_output
                                        done_events[name].clear()
                                        ready_events[name].set()

                                    # 等待完成
                                    for name in workers:
                                        done_events[name].wait()

                                    # 收集并 merge 结果
                                    aes_tensor = merge_worker_results(reward_results["aes"])
                                    raft_tensor = merge_worker_results(reward_results["raft"])
                                    videoclip_tensor = merge_worker_results(reward_results["videoclip"])
                                    videophy_tensor = merge_worker_results(reward_results["videophy"])

                                    # 调试输出（可选）
                                    # for idx, data in enumerate(videophy_tensor):
                                    #     print(f"idx {idx}: {data.print_data_proto(f'videophy_data_{idx}')}")

                                    # 合并所有 reward 到 batch
                                    batch_with_rewards = (
                                        gen_batch_output
                                        .union(aes_tensor)
                                        .union(raft_tensor)
                                        .union(videoclip_tensor)
                                        .union(videophy_tensor)
                                    )

                                    # 计算平均 reward（可配置权重）
                                    avg_reward = (
                                        batch_with_rewards.batch["aes_rewards"]
                                        + batch_with_rewards.batch["raft_rewards"]
                                        + batch_with_rewards.batch["videoclip_rewards"]
                                        + batch_with_rewards.batch["videophy_rewards"]
                                    )

                                    # 构造最终 reward DataProto（带平均 reward）
                                    from tensordict import TensorDict
                                    avg_reward_td = TensorDict({"rewards": avg_reward}, batch_size=avg_reward.shape[0])
                                    avg_reward_proto = DataProto(
                                        batch=avg_reward_td,
                                        non_tensor_batch=aes_tensor.non_tensor_batch  # 假设一致
                                    )
                                    final_batch = batch_with_rewards.union(avg_reward_proto)

                                    # 更新指标
                                    metrics["train/rewards"] = avg_reward.mean()
                                    metrics["train/log_probs"] = final_batch.batch["log_probs"].mean()

                                    # 替换 gen_batch_output（用于后续训练）
                                    gen_batch_output = final_batch

                                    print(f"reward time: {time.time() - start_time:.2f}s")

                                elif self.config.reward_model.type in ("qwen", "single"):
                                    # ======================
                                    # Single / Qwen Reward Model
                                    # ======================
                                    reward_input = gen_batch_output.select( # 选取之后 原对象的这些属性还在gen_batch_output里面
                                        batch_keys=['null_context'],
                                        non_tensor_batch_keys=["caption", "video_ids"]
                                    )
                                    if self.config.reward_model.type == "qwen":
                                     # 调用统一的 rm_wg（仅 single/qwen 有）
                                        reward_tensor = self.rm_wg.compute_rm_score(reward_input)
                                        reward_tensor.pop(non_tensor_batch_keys=["caption", "video_ids"])
                                        gen_batch_output.pop(
                                            non_tensor_batch_keys=["caption", "video_ids"]
                                        )
                                        # 清理原 batch（移除大 tensor 如 video_frames）
                                    else:  # "single"
                                        reward_tensor = self.rm_wg.compute_rm_score(reward_input)
                                        reward_input = gen_batch_output
                                    
                                    _debug_proto_batch("gen_batch_output", gen_batch_output)
                                    _debug_proto_batch("reward_tensor", reward_tensor)
                                    gen_batch_output = gen_batch_output.union(reward_tensor)
                                    # gen_batch_output = gen_batch_output.pop(
                                    #     non_tensor_batch_keys=["caption", "video_ids"]
                                    # )
                                    # 更新指标
                                    metrics["train/rewards"] = gen_batch_output.batch['rewards'].mean()
                                    metrics["train/log_probs"] = gen_batch_output.batch["log_probs"].mean()

                                else:
                                    raise ValueError(f"Unsupported reward model type: {self.config.reward_model.type}")
                
                        else:
                            reward_tensor = self.reward_fn(gen_batch_output, return_dict=True)
                            gen_batch_output = gen_batch_output.union(reward_tensor)
                            gen_batch_output.pop(batch_keys=['video_frames'])
                    
                    # === Updating ===
                    # batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.

                    
                    # if self.config.trainer.balance_batch:
                    #     self._balance_batch(new_batch, metrics=metrics)

                    # compute global_valid tokens
                    # batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # if self.use_reference_policy:
                    #     # compute reference log_prob
                    #     with marked_timer("ref", timing_raw):
                    #         ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    #         batch = batch.union(ref_log_prob)

                    # compute values
                    # if self.use_critic:
                    #     with marked_timer("values", timing_raw):
                    #         values = self.critic_wg.compute_values(batch)
                    #         batch = batch.union(values)

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
                        metrics["train/advantage"] = gen_batch_output.batch['advantages'].mean()

                    # # update critic
                    # if self.use_critic:
                    #     with marked_timer("update_critic", timing_raw):
                    #         critic_output = self.critic_wg.update_critic(batch)
                    #     critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    #     metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw):
                            gen_batch_output = self.actor_rollout_wg.update_actor(gen_batch_output)
                        actor_output_metrics = reduce_metrics(gen_batch_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    # if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    #     with marked_timer("testing", timing_raw):
                    #         val_metrics: dict = self._validate()
                    #         if is_last_step:
                    #             last_val_metrics = val_metrics
                    #     metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with marked_timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                # metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=new_batch, timing_raw=timing_raw))
                print("metrics",metrics)
                print("="*100)
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                # metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # timing_raw = defaultdict(float)  # clear timing

                # metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                logger.log(data=metrics, step=self.global_steps)
                timing_raw = defaultdict(float)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
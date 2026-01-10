import argparse
import math
import os
from pathlib import Path
import time
from torch.utils.data import DataLoader
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data.distributed import DistributedSampler
# 在文件开头添加这些导入
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
import glob 
# import wandb
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
import torch.distributed as dist
from collections import deque
import numpy as np
from einops import rearrange
import torch.distributed as dist
from torch.nn import functional as F
from typing import List
from PIL import Image
import cv2
import logging
import json
import random
import os
# 导入WAN相关模块  这个是OOM错误的版本
import wan
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel  
from wan.modules.vae import WanVAE
from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler

from fastvideo.reward_tracker import RewardTracker,create_reward_summary_plot
from fastvideo.utils.checkpoint import save_checkpoint


#from detection_reward import DetectionReward
# from Qwen_Reward import  QwenVideoRewardModel
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

check_min_version("0.31.0")

# 保存checkpoint功能，只保存，不加载也不清理
def save_checkpoint_simple(model, optimizer, lr_scheduler, step, epoch, rank, output_dir, args):
    """
    简单的checkpoint保存功能 - 只保存，不加载，不清理
    
    Args:
        model: FSDP包装的模型
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        step: 当前训练步数
        epoch: 当前epoch
        rank: 当前进程rank
        output_dir: 输出目录
        args: 训练参数
    """
    if not output_dir:
        print("Warning: output_dir is None, skipping checkpoint save")
        return
    
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-step-{step}")
    
    # 只有rank 0创建目录和保存文件
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"💾 Saving checkpoint at step {step} to {checkpoint_dir}")
    
    # 同步所有进程
    if dist.is_initialized():
        dist.barrier()
    
    try:
        # 使用FSDP的state_dict保存
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=False, rank0_only=True)):
            model_state_dict = model.state_dict()
        
        if rank == 0:
            # 【修改】导入safetensors并使用safetensor格式保存模型权重
            from safetensors.torch import save_file
            
            # 保存模型权重为safetensor格式
            model_file = os.path.join(checkpoint_dir, 'model.safetensors')
            save_file(model_state_dict, model_file)
            
            # 保存其他训练状态（优化器、调度器等）为标准格式
            training_state = {
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'step': step,
                'epoch': epoch,
                'args': vars(args),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            torch.save(training_state, os.path.join(checkpoint_dir, 'training_state.bin'))
            
            # 保存简单的配置文件
            config_path = os.path.join(checkpoint_dir, 'config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                import json
                config_dict = {
                    'step': step,
                    'epoch': epoch,
                    'learning_rate': args.learning_rate,
                    'train_batch_size': args.train_batch_size,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                }
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            # 检查保存是否成功
            if os.path.exists(model_file):
                file_size = os.path.getsize(model_file) / 1024**3  # GB
                print(f"✅ Checkpoint saved successfully:")
                print(f"   - Model: {model_file}")
                print(f"   - Size: {file_size:.2f} GB")
                print(f"   - Step: {step}, Epoch: {epoch}")
                print(f"   - Format: SafeTensor")
            else:
                print(f"❌ Failed to save checkpoint file")
    
    except Exception as e:
        if rank == 0:
            print(f"❌ Error saving checkpoint at step {step}: {e}")
    
    # 最终同步
    if dist.is_initialized():
        dist.barrier()


def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)

# 保存完整视频文件到指定目录供Reward计算使用
def save_video_for_reward(video_frames, caption, batch_idx, step, 
                          reward_dir="/gemini/space/ljm/Wan2.1-main_72btest_for_more/Reward"):
    """
    保存完整视频文件到指定目录供Reward计算使用
    
    Args:
        video_frames: torch.Tensor, shape (C, T, H, W), 范围 [0, 1]
        caption: str, 对应的文本prompt
        batch_idx: int, 当前batch的索引
        step: int, 当前训练步数
        reward_dir: str, 奖励计算目录路径
    
    Returns:
        video_path: str, 保存的视频文件路径
        meta_path: str, 保存的元数据文件路径
    """
    import time
    import json
    from datetime import datetime
    import cv2
    
    # 创建奖励计算目录结构
    os.makedirs(reward_dir, exist_ok=True)
    os.makedirs(os.path.join(reward_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(reward_dir, "metadata"), exist_ok=True)
    os.makedirs(os.path.join(reward_dir, "results"), exist_ok=True)
    
    # 生成唯一的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    sample_id = f"step{step}_batch{batch_idx}_{timestamp}"
    
    # 视频文件路径
    video_filename = f"video_{sample_id}.mp4"
    video_path = os.path.join(reward_dir, "videos", video_filename)
    
    # 元数据文件路径
    meta_filename = f"meta_{sample_id}.json"
    meta_path = os.path.join(reward_dir, "metadata", meta_filename)
    
    assert video_frames.dim() == 4, f"Expected 4D tensor, got {video_frames.shape}"
    
    C, T, H, W = video_frames.shape
    
    # 转换为CPU并确保数值范围正确
    video_frames = video_frames.cpu().clone()
    video_frames = torch.clamp(video_frames, 0, 1)
    
    # 转换为numpy格式 (T, H, W, C)
    video_np = video_frames.permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
    video_np = (video_np * 255).astype(np.uint8)
    
    # 如果是单通道，扩展为3通道
    if C == 1:
        video_np = np.repeat(video_np, 3, axis=-1)
    elif C > 3:
        video_np = video_np[:, :, :, :3]  # 只取前3个通道
    
    # 保存视频文件
    fps = 15  # 视频帧率，与你的设置一致
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (W, H))
    
    for t in range(T):
        frame = video_np[t]  # (H, W, C)
        # OpenCV使用BGR格式
        if C >= 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        out.write(frame_bgr)
    
    out.release()
    
    # 保存元数据
    metadata = {
        "sample_id": sample_id,
        "caption": caption,
        "batch_idx": batch_idx,
        "step": step,
        "timestamp": timestamp,
        "video_shape": [C, T, H, W],
        "video_path": video_path,
        "video_filename": video_filename,
        "fps": fps,
        "total_frames": T,
        "status": "pending"  # pending, processing, completed, failed
    }
    
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Video saved for reward computation: {video_path}")
    print(f"  - {T} frames at {fps} FPS")
    print(f"  - Resolution: {W}x{H}")
    print(f"  - Caption: {caption[:50]}...")
    
    return video_path, meta_path

# 在指定设备上运行的Qwen奖励计算函数，监听新视频并计算奖励
# inputs：reward_dir reward存储路径
# qwen_model_path: qwen模型路径
# device: 设备
# 

def compute_qwen_vllm_reward_on_cuda7(
    llm,
    sampling_params,
    reward_dir="/gemini/space/ljm/Wan2.1-main_72btest_for_more/Reward",
    # qwen_model_path="/gemini/platform/public/aigc/zqni/wan2.1/Lirui/Wan2.1-main/Qwen72b",
    qwen_model_path="/gemini/space/Qwen/Qwen2___5-VL-72B-Instruct",
    device="auto",  # 这里接收"auto"参数
    check_interval=1.0,
    # max_wait_time=30000.0,
    new_files = None,
):
    """
    在指定设备上运行的Qwen奖励计算函数，监听新视频并计算奖励
    """
    # from Qwen_Reward import QwenVideoRewardModel
    import glob
    # 确保CUDA可用
    assert torch.cuda.is_available(), "CUDA not available"
    
    # 【修复】处理device参数
    if device == "auto":
        # 当传入auto时，使用自动分片模式
        actual_device = "auto"
        print(f"Qwen reward computation using auto device mapping across available GPUs")
    else:
        # 当传入具体设备时，设置该设备并使用单GPU模式
        torch.cuda.set_device(device)
        actual_device = device
        print(f"Qwen reward computation running on {device}")
    
    # 创建目录
    metadata_dir = os.path.join(reward_dir, "metadata")
    videos_dir = os.path.join(reward_dir, "videos")
    results_dir = os.path.join(reward_dir, "results")
    
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    processed_files = set()  # 记录已处理的文件
    
    print(f"Starting Qwen reward computation service...")
    print(f"Monitoring directory: {metadata_dir}")
    print(f"Check interval: {check_interval}s")
    # print(f"Max wait time: {max_wait_time}s")
    
    if new_files != None:
        from qwen_reward import generate_batch_prompts_for_train,get_batch_results
        
        print("开始生成prompts")
        messages = generate_batch_prompts_for_train(new_files)
        print(f"成功加载{len(messages)}个prompts")
        
        print("="*40)
        print("开始用vllm计算得分")
        outputs=[]
        
        i=1
        print(len(messages))
        
        for message in messages:
            output = llm.chat(message, sampling_params= sampling_params)
            print(f"成功处理第{i}个message")
            i+=1
            outputs.append(output)
        results = get_batch_results(outputs)
        if len(results)!=len(new_files):
            print(f"len(results)={len(results)}不等于len(new_files)={len(new_files)}")
            exit(1)

        for i in range(len(new_files)):
            # 处理新文件
            meta_file = new_files[i]   
            result = results[i]
            print(f"Processing: {os.path.basename(meta_file)}")
            
            # 读取元数据
            with open(meta_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 跳过已失败或已完成的任务
            if metadata.get("status") in ["failed", "completed"]:
                processed_files.add(meta_file)
                continue
            
            sample_id = metadata["sample_id"]
            caption = metadata["caption"]
            video_path = metadata["video_path"]
            
            # 检查视频文件是否存在
            assert os.path.exists(video_path), f"Video file not found: {video_path}"
            
            # 更新状态为processing
            metadata["status"] = "processing"
            metadata["processing_start_time"] = time.time()
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            print(f"  - Video path: {video_path}")
            print(f"  - Caption: {caption[:100]}...")
                
            # 获取奖励分数
            raw_score = result.get("overall_score", 50.0)
            qwen_reward = float(raw_score)
            
            print(f"  - Raw score: {raw_score:.4f}")
            print(f"  - Adaptive reward: {qwen_reward:.4f}")
            
            # 保存奖励结果
            result_filename = f"reward_{sample_id}.txt"
            result_path = os.path.join(results_dir, result_filename)
            
            reward_data = {
                "sample_id": sample_id,
                "raw_score": float(raw_score),
                "adaptive_reward": float(qwen_reward),
                "caption": caption,
                "video_path": video_path,
                "computation_time": time.time() - metadata["processing_start_time"],
                "timestamp": time.time()
            }
            
            # 保存为JSON格式
            result_json_path = os.path.join(results_dir, f"reward_{sample_id}.json")
            with open(result_json_path, 'w', encoding='utf-8') as f:
                json.dump(reward_data, f, ensure_ascii=False, indent=2)
            
            # 保存简单的txt格式（兼容性）
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write(f"{qwen_reward:.6f}\n")
            
            # 更新元数据状态为completed
            metadata["status"] = "completed"
            metadata["raw_score"] = float(raw_score)
            metadata["adaptive_reward"] = float(qwen_reward)
            metadata["result_path"] = result_json_path
            metadata["completion_time"] = time.time()
            
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ Completed: {sample_id} -> Raw: {raw_score:.4f}, Reward: {qwen_reward:.4f}")
            
            # 清理显存
            torch.cuda.empty_cache()
            
            processed_files.add(meta_file)
    
    print(f"Qwen成功计算{len(new_files)}个实例的奖励分数")    

    

# 从文件中加载奖励分数供GRPO训练使用
def load_reward_from_file(sample_ids, reward_dir="/gemini/space/ljm/Wan2.1-main_72btest_for_more/Reward", max_wait_time=30000000.0, check_interval=0.5):
    """
    从文件中加载奖励分数供GRPO训练使用
    
    Args:
        sample_ids: List[str], 样本ID列表
        reward_dir: str, 奖励计算目录路径
        max_wait_time: float, 最大等待时间（秒）
        check_interval: float, 检查间隔时间（秒）
    
    Returns:
        rewards: torch.Tensor, shape (B,), 奖励分数张量
        success_flags: List[bool], 每个样本是否成功加载奖励
    """
    import time
    import json
    import os
    
    results_dir = os.path.join(reward_dir, "results")
    metadata_dir = os.path.join(reward_dir, "metadata")
    
    rewards = []
    success_flags = []
    start_time = time.time()
    
    # 新增
    torch.cuda.memory._record_memory_history(max_entries=100000)
    
    print(f"Loading rewards for {len(sample_ids)} samples...")
    
    
    for i, sample_id in enumerate(sample_ids):
        reward_loaded = False
        sample_start_time = time.time()
        
        # 定义文件路径
        reward_txt_path = os.path.join(results_dir, f"reward_{sample_id}.txt")
        reward_json_path = os.path.join(results_dir, f"reward_{sample_id}.json")
        meta_path = os.path.join(metadata_dir, f"meta_{sample_id}.json")
        
        # 等待奖励文件生成
        while not reward_loaded and (time.time() - sample_start_time) < max_wait_time:
            # 检查元数据状态
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                status = metadata.get("status", "pending")
                
                if status == "completed":
                    # 尝试加载JSON格式的奖励（优先）
                    if os.path.exists(reward_json_path):
                        with open(reward_json_path, 'r', encoding='utf-8') as f:
                            reward_data = json.load(f)
                        
                        # 获取自适应奖励分数
                        adaptive_reward = reward_data.get("adaptive_reward")
                        if adaptive_reward is not None:
                            rewards.append(float(adaptive_reward))
                            success_flags.append(True)
                            reward_loaded = True
                            print(f"  ✓ Sample {i+1}/{len(sample_ids)}: {sample_id} -> {adaptive_reward:.4f}")
                            break
                    
                    # 如果JSON不存在，尝试加载TXT格式
                    elif os.path.exists(reward_txt_path):
                        with open(reward_txt_path, 'r', encoding='utf-8') as f:
                            reward_score = float(f.read().strip())
                        
                        rewards.append(reward_score)
                        success_flags.append(True)
                        reward_loaded = True
                        print(f"  ✓ Sample {i+1}/{len(sample_ids)}: {sample_id} -> {reward_score:.4f}")
                        break
                    
                elif status == "failed":
                    print(f"  ✗ Sample {i+1}/{len(sample_ids)}: {sample_id} -> Failed to compute reward")
                    # 使用默认奖励分数
                    default_reward = 50.0
                    rewards.append(default_reward)
                    success_flags.append(False)
                    reward_loaded = True
                    break
                
                elif status == "processing":
                    print(f"  ⏳ Sample {i+1}/{len(sample_ids)}: {sample_id} -> Still processing...")
                
            # 等待一段时间后重试
            time.sleep(check_interval)
        
        # 如果超时仍未加载到奖励
        if not reward_loaded:
            print(f"  ⏰ Sample {i+1}/{len(sample_ids)}: {sample_id} -> Timeout, using default reward")
            default_reward = 50.0
            rewards.append(default_reward)
            success_flags.append(False)
    
    # 转换为张量
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    
    # 统计信息
    success_count = sum(success_flags)
    total_time = time.time() - start_time
    
    print(f"Reward loading completed:")
    print(f"  - Successfully loaded: {success_count}/{len(sample_ids)} ({success_count/len(sample_ids)*100:.1f}%)")
    print(f"  - Failed/Timeout: {len(sample_ids)-success_count}/{len(sample_ids)}")
    print(f"  - Total time: {total_time:.2f}s")
    print(f"  - Average reward: {rewards_tensor.mean().item():.4f}")
    print(f"  - Reward range: [{rewards_tensor.min().item():.4f}, {rewards_tensor.max().item():.4f}]")
    
    # 新增
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    file_name = f"visual_mem_{timestamp}.pickle"
    # save record:
    torch.cuda.memory._dump_snapshot(file_name)
    # Stop recording memory snapshot history:
    torch.cuda.memory._record_memory_history(enabled=None)
    
    return rewards_tensor, success_flags

# 将Qwen奖励计算集成到GRPO训练循环中
def integrate_qwen_reward_to_grpo(
    llm,
    sampling_params,
    videos_list,           # List[torch.Tensor], 生成的视频列表 (每个形状: C, T, H, W)  
    prompts_list,          # List[str], 对应的prompt列表
    step,                  # int, 当前训练步数
    reward_dir="/gemini/space/ljm/Wan2.1-main_72btest_for_more/Reward",
    meta_dir="/gemini/space/ljm/Wan2.1-main_72btest_for_more/Reward",
    max_wait_time=30000000.0,    # 最大等待奖励计算时间
    device="cuda",         # 当前训练设备
    num_generations=2      # 每个prompt的生成数量
):
    """
    将Qwen奖励计算集成到GRPO训练循环中
    
    Args:
        videos_list: List[torch.Tensor], 生成的视频列表，长度为 batch_size * num_generations
        prompts_list: List[str], 对应的prompt列表，长度为 batch_size
        step: int, 当前训练步数
        reward_dir: str, 奖励计算目录
        max_wait_time: float, 等待奖励计算的最大时间
        device: str, 当前训练设备
        num_generations: int, 每个prompt的生成数量
    
    Returns:
        rewards: torch.Tensor, shape (batch_size, num_generations), 奖励分数
        rewards_flat: torch.Tensor, shape (batch_size * num_generations), 展平的奖励分数
        success_rate: float, 成功计算奖励的比例
    """
    import time
    
    batch_size = len(prompts_list)
    total_videos = len(videos_list)
    
    assert total_videos == batch_size * num_generations, \
        f"Expected {batch_size * num_generations} videos, got {total_videos}"
    
    print(f"🎬 GRPO Reward Integration - Step {step}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Generations per prompt: {num_generations}")
    print(f"  - Total videos: {total_videos}")
    
    # 1. 批量保存视频供奖励计算
    print(f"📹 Saving videos for reward computation...")
    sample_ids = []
    video_paths = []
    
    for video_idx, video_frames in enumerate(videos_list):
        # 计算对应的prompt索引和生成索引
        prompt_idx = video_idx // num_generations
        gen_idx = video_idx % num_generations
        
        prompt = prompts_list[prompt_idx]
        
        # 为每个视频创建唯一的batch_idx（包含生成索引信息）
        unique_batch_idx = video_idx  # 使用视频索引作为唯一标识
        
        video_path, meta_path = save_video_for_reward(
            video_frames=video_frames,
            caption=prompt,
            batch_idx=unique_batch_idx,
            step=step,
            reward_dir=reward_dir
        )
        new_files=[]
        if video_path is not None and meta_path is not None:
            # 从元数据文件名中提取sample_id
            meta_filename = os.path.basename(meta_path)
            new_files.append(meta_path)
            sample_id = meta_filename.replace("meta_", "").replace(".json", "")
            sample_ids.append(sample_id)
            video_paths.append(video_path)
            print(f"  ✓ Video {video_idx+1}/{total_videos}: prompt_{prompt_idx}_gen_{gen_idx} -> {sample_id}")
        else:
            print(f"  ✗ Failed to save video {video_idx+1}/{total_videos}")
            sample_ids.append(None)
            video_paths.append(None)
    
    successful_saves = sum(1 for sid in sample_ids if sid is not None)
    print(f"📹 Video saving completed: {successful_saves}/{total_videos} videos saved")
    
    if successful_saves == 0:
        print("❌ No videos saved successfully, using default rewards")
        # 返回默认奖励
        default_reward = torch.tensor(50.0)
        rewards_flat = default_reward.repeat(total_videos).to(device)
        rewards = rewards_flat.view(batch_size, num_generations)
        return rewards, rewards_flat, 0.0

    try:
        compute_qwen_vllm_reward_on_cuda7(
            llm=llm,
            sampling_params=sampling_params,
            new_files=new_files,  #将待评估的文件路径传入
            reward_dir="/gemini/space/ljm/Wan2.1-main_72btest_for_more/Reward",
            qwen_model_path="/gemini/space/Qwen/Qwen2___5-VL-72B-Instruct",
            device="auto",
            check_interval=1.0,
            # max_wait_time=360000000000000.0
        )
    except Exception as e:
        print(f"❌ Error in Qwen service: {e}")
    finally:
        #cleanup_distributed()
        pass
    
    # 2. 等待并加载奖励分数
    print(f"⏳ Waiting for reward computation (max {max_wait_time}s)...")
    
    # 过滤出有效的sample_ids
    valid_sample_ids = [sid for sid in sample_ids if sid is not None]
    
    rewards_tensor, success_flags = load_reward_from_file(
        sample_ids=valid_sample_ids,
        reward_dir=reward_dir,
        max_wait_time=max_wait_time,
        check_interval=0.5
    )
    
    # 3. 重构奖励张量为正确的形状
    print(f"🔄 Reconstructing reward tensor...")
    
    # 创建完整的奖励列表（包含失败的条目）
    full_rewards = []
    valid_idx = 0
    
    for i, sample_id in enumerate(sample_ids):
        if sample_id is not None:
            # 使用计算得到的奖励
            if valid_idx < len(rewards_tensor):
                reward_value = rewards_tensor[valid_idx].item()
                full_rewards.append(reward_value)
                valid_idx += 1
            else:
                full_rewards.append(50.0)  # 默认奖励
        else:
            full_rewards.append(50.0)  # 默认奖励
    
    # 转换为张量并移动到正确设备
    rewards_flat = torch.tensor(full_rewards, dtype=torch.float32, device=device)
    
    # 重塑为 (batch_size, num_generations)
    rewards = rewards_flat.view(batch_size, num_generations)
    
    # 4. 计算统计信息
    success_count = sum(success_flags) if success_flags else 0
    success_rate = success_count / total_videos if total_videos > 0 else 0.0
    
    print(f"🎯 Reward computation completed:")
    print(f"  - Success rate: {success_rate:.1%} ({success_count}/{total_videos})")
    print(f"  - Reward tensor shape: {rewards.shape}")
    print(f"  - Reward range: [{rewards.min().item():.3f}, {rewards.max().item():.3f}]")
    print(f"  - Average reward: {rewards.mean().item():.3f}")
    
    # 5. 按prompt分组显示奖励（用于调试）
    print(f"📊 Rewards by prompt:")
    for prompt_idx in range(batch_size):
        prompt_rewards = rewards[prompt_idx]  # shape: (num_generations,)
        prompt_text = prompts_list[prompt_idx][:50] + "..." if len(prompts_list[prompt_idx]) > 50 else prompts_list[prompt_idx]
        
        rewards_str = ", ".join([f"{r:.3f}" for r in prompt_rewards])
        print(f"  Prompt {prompt_idx}: [{rewards_str}] | '{prompt_text}'")
    
    return rewards, rewards_flat, success_rate

# WAN的Flow Matching采样步骤，转换为SDE求解器支持GRPO
def wan_step(
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
    # 根据模型预测的flow更新潜在变量
    
    sigma = sigmas[index]           # 当前噪声水平（如0.9）
    dsigma = sigmas[index + 1] - sigma  # 噪声变化量（如-0.02）
    # 数学形式：z_{t-1} = z_t + Δσ * model_output
    prev_sample_mean = latents + dsigma * model_output
    
    # --- 预测原始样本（去噪结果）---
    # 数学形式：x_0 ≈ (z_t - σ·model_output)
    pred_original_sample = latents - sigma * model_output
    # --- 随机性控制 ---
    delta_t = sigma - sigmas[index + 1]  # 时间间隔（正值）
    std_dev_t = eta * torch.sqrt(delta_t)  # 噪声标准差（Hunyuan风格实现）
    
    if sde_solver:  # 使用SDE求解器（和FLUX相同）
        # 计算得分函数估计：score = -(z_t - x_0*(1-σ))/σ²
        score_estimate = -(latents - pred_original_sample * (1 - sigma)) / (sigma**2)  # 估计的得分
        # 对数修正项：-0.5*η²*score*Δσ
        log_term = -0.5 * eta**2 * score_estimate  # 对数项修正
        # 修正均值：z_{t-1} += log_term*Δσ
        prev_sample_mean = prev_sample_mean + log_term * dsigma  # 修正的均值
    # --- GRPO模式初始化 ---
    if grpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t # 添加随机噪声：z_{t-1} = mean + N(0, std_dev_t)

    if grpo:
        # 计算对数概率密度（高斯分布）：
        # log p(z_{t-1}|z_t) = -||z_{t-1}-mean||²/(2σ²) - log(σ√2π)
        log_prob = (-((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2)))- math.log(std_dev_t + 1e-8) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))

        # 沿所有非batch维度求平均（保留batch维度）
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean, pred_original_sample

class WanPreprocessedDataset(torch.utils.data.Dataset):
    """加载预处理context的WAN数据集"""
    
    def __init__(self, processed_json_path):
        # 加载处理后的数据
        with open(processed_json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} preprocessed items")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 加载numpy文件并转换为tensor
        context_numpy = np.load(item['context_path'])
        context = torch.from_numpy(context_numpy)  # shape: (L, C)
        
        return {
            'context': context,           # 单个tensor，shape (L, C)
            'caption': item['caption']    # 原始文本（用于调试）
        }

def wan_preprocessed_collate_function(batch):
    """修复的collate函数 - 返回List[Tensor]格式"""
    # 将每个context tensor放入列表中，这样就是List[Tensor]格式
    contexts = [item['context'] for item in batch]  # List[Tensor]，每个tensor shape (L, C)
    captions = [item['caption'] for item in batch]
    
    return {
        'contexts': contexts,  # List[Tensor]，符合WAN模型期望
        'captions': captions   # list of strings
    }

class WanDataset(torch.utils.data.Dataset):
    """WAN训练数据集"""
    
    def __init__(self, data_json_path, text_len=512):
        self.text_len = text_len
        
        # 加载数据
        with open(data_json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 如果数据是字典格式，转换为列表
        if isinstance(self.data, dict):
            self.data = list(self.data.values())
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 提取caption
        if isinstance(item, dict):
            caption = item.get('caption', item.get('text', 'A video'))
        else:
            caption = str(item)
        
        return {
            'caption': caption
        }


def save_video_and_prompt(video_frames, caption, rank, index, args):
    """
    保存视频文件和对应的prompt文本
    Args:
        video_frames: torch.Tensor, shape (C, T, H, W), 范围 [0, 1]
        caption: str, 对应的文本prompt
        rank: int, 当前进程的rank
        index: int, 当前batch的索引
        args: 配置参数
    """
    import time
    from datetime import datetime
    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 确保video_frames是正确的格式 (C, T, H, W)
    if video_frames.dim() == 4:
        C, T, H, W = video_frames.shape
        
        # 转换为numpy格式 (T, H, W, C)
        video_np = video_frames.permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, C)
        video_np = (video_np * 255).astype(np.uint8)
        
        # 如果是单通道，扩展为3通道
        if C == 1:
            video_np = np.repeat(video_np, 3, axis=-1)
        
        # 1. 保存第一帧图像
        first_frame = video_np[0]  # (H, W, C)
        
        # 保存第一帧为PNG图像
        if C >= 3:
            first_frame_pil = Image.fromarray(first_frame)
        else:
            first_frame_pil = Image.fromarray(first_frame[:,:,0], mode='L')
        
        image_filename = f"wan_frame_rank{rank}_batch{index}.png"
        image_path = os.path.join("./images", image_filename)
        
        try:
            first_frame_pil.save(image_path)
            # print(f"First frame saved: {image_path}")
        except Exception as e:
            print(f"Error saving first frame {image_path}: {e}")

        # 保存视频
        video_filename = f"wan_video_rank{rank}_batch{index}.mp4"
        video_path = os.path.join("./videos", video_filename)
        
        try:
            # 使用opencv保存视频
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = args.video_fps if hasattr(args, 'video_fps') else 8  # 默认8fps
            
            out = cv2.VideoWriter(video_path, fourcc, fps, (W, H))
            
            for t in range(T):
                frame = video_np[t]  # (H, W, C)
                # OpenCV使用BGR格式
                if C == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                out.write(frame_bgr)
            
            out.release()
            print(f"Video saved: {video_path}")
            
        except Exception as e:
            print(f"Error saving vid/eo {video_path}: {e}")
            # 如果视频保存失败，至少保存第一帧作为图像
            first_frame = video_np[0]  # (H, W, C)
            if C == 3:
                first_frame_pil = Image.fromarray(first_frame)
            else:
                first_frame_pil = Image.fromarray(first_frame[:,:,0], mode='L')
            
            image_filename = f"wan_frame_rank{rank}_batch{index}_{timestamp}.png"
            image_path = os.path.join("./images", image_filename)
            first_frame_pil.save(image_path)
            # print(f"First frame saved as image: {image_path}")
    else:
        print(f"Unexpected video_frames shape: {video_frames.shape}")


def run_wan_sample_step(
    args,
    latents,  # [(16, 7, 64, 64)]
    progress_bar, 
    sigma_schedule,  # 添加sigma_schedule sigma_schedule = torch.linspace(1, 0, sample_steps + 1)如[1.0, 0.875, 0.75, ..., 0.0]
    transformer,
    context,
    context_null,
    seq_len,
    grpo_sample,
    guide_scale=5.0,
):
    """WAN采样步骤，修正CFG实现避免通道数错误"""
    if grpo_sample:
        all_latents = [latents[0]]  # 存储初始latent (16, 7, 64, 64)
        all_log_probs = []
        
        for i in progress_bar:# 遍历每个扩散步
            B = len(context) if isinstance(context, list) else context.shape[0]
            # 确保设备一致
            device = latents[0].device
            
            # 使用sigma值计算timestep
            sigma = sigma_schedule[i]
            timestep_value = int(sigma * 1000)
            timestep = torch.full([B], timestep_value, device=device, dtype=torch.long)
            
            transformer.eval()
            
            # --- 分类器自由引导(CFG)实现 ---
            with torch.autocast("cuda", torch.bfloat16 if args.use_bf16 else torch.float32):
                with torch.no_grad():
                    # 方法1：完全序列化CFG，分别计算条件和无条件
                    if args.use_sequential_cfg:
                        # 先计算条件预测
                        pred_cond = transformer(
                            x=latents,  # 保持原始格式 [(16, 7, 64, 64)]
                            t=timestep,
                            context=context,  # List[Tensor]
                            seq_len=seq_len
                        )
                        
                        if isinstance(pred_cond, dict) and 'rgb' in pred_cond:
                            model_output_cond = pred_cond['rgb'][0]
                        elif isinstance(pred_cond, list):
                            model_output_cond = pred_cond[0]
                        else:
                            model_output_cond = pred_cond
                        
                        # 立即清理显存
                        del pred_cond
                        torch.cuda.empty_cache()
                        
                        # 再计算无条件预测
                        pred_uncond = transformer(
                            x=latents,  # 保持原始格式 [(16, 7, 64, 64)]
                            t=timestep,
                            context=context_null,  # List[Tensor]
                            seq_len=seq_len
                        )
                        
                        if isinstance(pred_uncond, dict) and 'rgb' in pred_uncond:
                            model_output_uncond = pred_uncond['rgb'][0]
                        elif isinstance(pred_uncond, list):
                            model_output_uncond = pred_uncond[0]
                        else:
                            model_output_uncond = pred_uncond
                        
                        del pred_uncond
                        torch.cuda.empty_cache()
                    
                    else:
                        # 方法2：如果显存足够，使用批次CFG（但不在latents维度合并）
                        # 创建两倍batch的timestep和context，但latents保持独立
                        
                        # 为条件和无条件分别创建timestep
                        timestep_cond = timestep
                        timestep_uncond = timestep
                        
                        # 为条件预测准备输入
                        pred_cond = transformer(
                            x=latents,  # [(16, 7, 64, 64)]
                            t=timestep_cond,
                            context=context,
                            seq_len=seq_len
                        )
                        
                        if isinstance(pred_cond, dict) and 'rgb' in pred_cond:
                            model_output_cond = pred_cond['rgb'][0]
                        elif isinstance(pred_cond, list):
                            model_output_cond = pred_cond[0]
                        else:
                            model_output_cond = pred_cond
                        
                        # 为无条件预测准备输入
                        pred_uncond = transformer(
                            x=latents,  # [(16, 7, 64, 64)]
                            t=timestep_uncond,
                            context=context_null,
                            seq_len=seq_len
                        )
                        
                        if isinstance(pred_uncond, dict) and 'rgb' in pred_uncond:
                            model_output_uncond = pred_uncond['rgb'][0]
                        elif isinstance(pred_uncond, list):
                            model_output_uncond = pred_uncond[0]
                        else:
                            model_output_uncond = pred_uncond
                        
                        del pred_cond, pred_uncond
                
                # CFG组合
                model_output = model_output_uncond + guide_scale * (model_output_cond - model_output_uncond)
                del model_output_cond, model_output_uncond
                torch.cuda.empty_cache()

            # WAN的SDE采样步骤
            next_latents, pred_original, log_prob = wan_step(
                model_output, 
                latents[0].to(torch.float32),  # (16, 7, 64, 64)
                args.eta, 
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
        all_latents = torch.stack(all_latents, dim=0)  # (timestep, 16, 7, 64, 64)
        all_log_probs = torch.stack(all_log_probs, dim=0)  # (timestep, B) -> (timestep,)
        
        return latents, final_latents, all_latents, all_log_probs

def sample_wan_reference_model(
    args,
    llm,
    sampling_params,
    device, 
    transformer,
    vae,
    contexts,
    captions,
    context_null_single,
    reward_model=None,
    tokenizer=None,
    preprocess_val=None,
    qwen_reward_model=None,  # 这个参数现在不使用，保持兼容性
    use_file_based_reward=False  # 新增参数
):
    """WAN参考模型采样，支持文件通信的奖励计算"""
    # 视频参数
    frame_num = args.t
    size = (args.w, args.h)
    
    # 创建sigma调度
    sample_steps = args.sampling_steps
    sigma_schedule = torch.linspace(1, 0, sample_steps + 1).to(device)
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)
    
    B = len(captions)
    batch_size = 1
    batch_indices = torch.chunk(torch.arange(B), B // batch_size)

    all_latents = []
    all_log_probs = []
    all_rewards = []

    # VAE参数
    vae_stride = [4, 8, 8]
    patch_size = [1, 2, 2]
    
    latent_dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    
    # 收集所有生成的视频用于批量奖励计算
    all_video_frames = []
    all_batch_captions = []
    
    if args.init_same_noise:
        latent_shape = (
            16,
            (frame_num - 1) // vae_stride[0] + 1,
            size[1] // vae_stride[1],
            size[0] // vae_stride[2]
        )
        input_latents = torch.randn(latent_shape, device=device, dtype=latent_dtype)

    # 逐批次生成视频
    for index, batch_idx in enumerate(batch_indices):
        batch_captions = [captions[i] for i in batch_idx]
        batch_contexts = [contexts[i].to(device) for i in batch_idx]
        batch_context_null = [context_null_single[0] for _ in batch_idx]
        
        if not args.init_same_noise:
            latent_shape = (
                16,
                (frame_num - 1) // vae_stride[0] + 1,
                size[1] // vae_stride[1],
                size[0] // vae_stride[2]
            )
            input_latents = torch.randn(latent_shape, device=device, dtype=latent_dtype)

        seq_len = math.ceil(
            (latent_shape[2] * latent_shape[3]) / (patch_size[1] * patch_size[2]) * latent_shape[1]
        )

        grpo_sample = True
        progress_bar = tqdm(range(0, args.sampling_steps), desc="WAN Sampling Progress")
        guide_scale = getattr(args, 'guide_scale', 5.0)

        with torch.no_grad():      
            _, final_latents, batch_latents, batch_log_probs = run_wan_sample_step(
                args,
                [input_latents],
                progress_bar,
                sigma_schedule,
                transformer,
                batch_contexts,
                batch_context_null,
                seq_len,
                grpo_sample,
                guide_scale,
            )

        batch_latents = batch_latents.unsqueeze(0)
        batch_log_probs = batch_log_probs.unsqueeze(0)
        
        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)

        # VAE解码
        rank = int(os.environ.get("RANK", 0))
        
        with torch.inference_mode():
            autocast_dtype = torch.bfloat16 if args.use_bf16 else torch.float32
            with torch.autocast("cuda", dtype=autocast_dtype):
                final_latents_vae = final_latents.to(dtype=autocast_dtype)
                decoded_videos = vae.decode([final_latents_vae])#对final_latents进行解码
                video_frames = decoded_videos[0]
                
                # 后处理
                video_frames = (video_frames + 1.0) / 2.0
                video_frames = torch.clamp(video_frames, 0, 1)
                
                # 创建输出目录
                os.makedirs("./videos", exist_ok=True)
                os.makedirs("./images", exist_ok=True)
                
                # 保存视频
                save_video_and_prompt(
                    video_frames,
                    batch_captions[0],
                    rank,
                    index,
                    args
                )

        # 收集视频和标题用于批量奖励计算
        all_video_frames.append(video_frames)
        all_batch_captions.extend(batch_captions)

        del final_latents, video_frames
        torch.cuda.empty_cache()

    # 使用文件通信计算奖励
    if use_file_based_reward:
        try:
            # 使用新的集成函数计算奖励
            rewards, rewards_flat, success_rate = integrate_qwen_reward_to_grpo(
                llm=llm,
                sampling_params=sampling_params,
                videos_list=all_video_frames,
                prompts_list=all_batch_captions,
                step=getattr(args, 'current_step', 0),
                reward_dir="/gemini/space/ljm/Wan2.1-main_72btest_for_more/Reward",
                max_wait_time=600000.0,  # 等不到就一直等
                device=device,
                num_generations=1  # 每个prompt只有1个视频
            )
            
            # 将奖励转换为正确格式
            for reward in rewards_flat:
                reward_tensor = torch.tensor([reward.item()], device=device, dtype=torch.float32)
                all_rewards.append(reward_tensor)
                
            rank = int(os.environ.get("RANK", 0))
            if rank == 0:
                print(f"📊 File-based reward computed successfully:")
                print(f"  - Success rate: {success_rate:.1%}")
                print(f"  - Average reward: {rewards_flat.mean().item():.3f}")
                
        except Exception as e:
            logger.error(f"File-based reward computation failed: {e}")
            # 使用默认奖励作为fallback
            for _ in all_video_frames:
                default_reward = torch.tensor([50.0], device=device, dtype=torch.float32)
                all_rewards.append(default_reward)
    
    # 原有的其他奖励计算方式（保持完整逻辑）
    elif args.use_hpsv2 and reward_model is not None and tokenizer is not None and preprocess_val is not None:
        # HPSv2奖励计算（保持原有逻辑）
        for video_frames in all_video_frames:
            try:
                # 这里保持你原有的HPSv2计算逻辑
                hps_reward = torch.tensor([50.0], device=device, dtype=torch.float32)  # 占位符
                all_rewards.append(hps_reward)
            except Exception as e:
                logger.error(f"Error computing HPSv2 reward: {e}")
                default_reward = torch.tensor([50.0], device=device, dtype=torch.float32)
                all_rewards.append(default_reward)
    
    elif qwen_reward_model is not None:
        # 直接使用Qwen模型计算奖励（保持原有逻辑）
        for video_frames in all_video_frames:
            try:
                qwen_reward = qwen_reward_model.evaluate_video_for_reward(video_frames)
                qwen_reward_tensor = torch.tensor([qwen_reward], device=device, dtype=torch.float32)
                all_rewards.append(qwen_reward_tensor)
                
                rank = int(os.environ.get("RANK", 0))
                if rank == 0:
                    print(f"Qwen video quality reward: {qwen_reward:.3f}")
                    with open('./wan_qwen_reward.txt', 'a') as f:
                        f.write(f"Video {len(all_rewards)-1}: Qwen_Quality={qwen_reward:.3f}\n")
                        
            except Exception as e:
                logger.error(f"Error computing Qwen reward: {e}")
                default_reward = torch.tensor([50.0], device=device, dtype=torch.float32)
                all_rewards.append(default_reward)
                rank = int(os.environ.get("RANK", 0))
                if rank == 0:
                    print(f"Warning: Qwen reward computation failed, using default reward 50.0")
    else:
        # 使用默认奖励
        for _ in all_video_frames:
            default_reward = torch.tensor([50.0], device=device, dtype=torch.float32)
            all_rewards.append(default_reward)
            rank = int(os.environ.get("RANK", 0))
            if rank == 0:
                print("Warning: No reward model provided, using default reward 50.0")

    # 拼接结果
    if len(all_latents) > 1:
        all_latents = torch.cat(all_latents, dim=0) #B, t, 16, 7, 64, 64
        all_log_probs = torch.cat(all_log_probs, dim=0) #B, t
        all_rewards = torch.cat(all_rewards, dim=0) #B
    else:
        all_latents = all_latents[0]
        all_log_probs = all_log_probs[0]
        all_rewards = all_rewards[0]
    
    return all_rewards, all_latents, all_log_probs, sigma_schedule

def grpo_wan_one_step(
    args,
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
    """GRPO的单步训练，修正CFG实现"""
    B = len(context) if isinstance(context, list) else context.shape[0]
    transformer.train()
    
    # 确保latents维度正确：(16, 7, 64, 64)
    if latents.dim() == 5:
        latents = latents.squeeze(0)
    if pre_latents.dim() == 5:
        pre_latents = pre_latents.squeeze(0)
    
    if latents.shape[0] != 16:
        raise ValueError(f"Expected 16 channels, got {latents.shape[0]} channels")
    
    # 使用适当的数据类型进行autocast
    autocast_dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    with torch.autocast("cuda", dtype=autocast_dtype):
        # 【关键修改】先计算无条件预测，并冻结梯度
        with torch.no_grad():  # 冻结无条件分支
            pred_uncond = transformer(
                x=[latents],
                t=timesteps,
                context=context_null,  # 无条件
                seq_len=seq_len
            )
            
            # 处理无条件预测输出
            if isinstance(pred_uncond, dict) and 'rgb' in pred_uncond:
                model_output_uncond = pred_uncond['rgb'][0].detach()  # 确保detach
            elif isinstance(pred_uncond, list):
                model_output_uncond = pred_uncond[0].detach()
            else:
                model_output_uncond = pred_uncond.detach()
                
            del pred_uncond
            torch.cuda.empty_cache()
        
        # 计算条件预测
        pred_cond = transformer(
            x=[latents],  # 保持List[Tensor]格式
            t=timesteps,
            context=context,  # List[Tensor]
            seq_len=seq_len
        )
            
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
        
        # CFG组合
        model_output = model_output_uncond + guide_scale * (model_output_cond - model_output_uncond)
        del model_output_cond, model_output_uncond

    # 确保数据类型一致性
    computation_dtype = torch.float32
    _, _, log_prob = wan_step(
        model_output.to(computation_dtype), 
        latents.to(computation_dtype), 
        args.eta, 
        sigma_schedule,
        i, 
        prev_sample=pre_latents.to(computation_dtype), 
        grpo=True, 
        sde_solver=True
    )
    
    return log_prob

def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)

def train_wan_one_step(
    args,
    llm,
    sampling_params,               
    device,
    transformer,
    vae,
    optimizer,
    lr_scheduler,
    loader,
    max_grad_norm,
    context_null_single,
    reward_model=None,
    tokenizer=None,
    preprocess_val=None,
    qwen_reward_model=None,  # 保持兼容性，但不再直接使用
):
    """WAN的一步训练，支持文件通信的奖励计算和自适应权重调整"""
    total_loss = 0.0
    optimizer.zero_grad()
    
    batch = next(loader)
    contexts = batch['contexts']
    captions = batch['captions']
    
    if args.use_group:
        expanded_contexts = []
        expanded_captions = []
        for context, caption in zip(contexts, captions):
            for _ in range(args.num_generations):
                expanded_contexts.append(context)
                expanded_captions.append(caption)
        contexts = expanded_contexts
        captions = expanded_captions

    # 获取当前步数（用于文件通信）
    current_step = getattr(args, 'current_step', 0)
    args.current_step = current_step
    
    # 使用文件通信的奖励计算
    reward, all_latents, all_log_probs, sigma_schedule = sample_wan_reference_model(
        args, llm, sampling_params, device, transformer, vae, contexts, captions, context_null_single,
        reward_model, tokenizer, preprocess_val, 
        qwen_reward_model=qwen_reward_model,  # 保持兼容性
        use_file_based_reward=getattr(args, 'use_file_based_reward', False)
    )
    with open('./use_file_base_reward.txt','a') as f:
        f.write(f"{getattr(args, 'use_file_based_reward', False)}")        
    
    
    
    batch_size = all_latents.shape[0]
    context_null = [context_null_single[0] for _ in range(batch_size)]
    
    # 将噪声水平sigma映射到整数时间步 (0-1000)
    timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][:args.sampling_steps]
    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    timesteps_tensor = torch.tensor(timestep_values, device=all_latents.device, dtype=torch.long)
    
    
    # 构建samples
    samples = {
        "timesteps": timesteps_tensor[:, :-1],           # (B, steps)
        "latents": all_latents[:, :-1],                  # (B, steps, C, T, H, W) - 当前状态  
        "next_latents": all_latents[:, 1:],              # (B, steps, C, T, H, W) - 下一状态
        "log_probs": all_log_probs,                      # (B, steps)
        "rewards": reward.to(torch.float32),             # (B,)
        "contexts": contexts,                            # List[Tensor]
        "context_null": context_null,                    # List[Tensor]
        "sigma_schedule": sigma_schedule,
    }
    
    # 分布式奖励统计
    gathered_reward = gather_tensor(samples["rewards"])
    if dist.get_rank() == 0:
        if getattr(args, 'use_file_based_reward', False):
            print("gathered_file_based_reward", gathered_reward)
            with open('./wan_file_based_reward.txt', 'a') as f:
                f.write(f"{gathered_reward.mean().item()}\n")
        elif args.use_hpsv2:
            print("gathered_hps_reward", gathered_reward)
            with open('./wan_hps_reward.txt', 'a') as f: 
                f.write(f"{gathered_reward.mean().item()}\n")
        else:
            print("gathered_qwen_reward", gathered_reward)
            with open('./wan_qwen_reward.txt', 'a') as f:
                f.write(f"{gathered_reward.mean().item()}\n")
    
    # 优势计算
    if args.use_group:
        n = len(samples["rewards"]) // args.num_generations
        advantages = torch.zeros_like(samples["rewards"])
        print("看看adv的shape和num_gen的形状",advantages.shape, n, args.num_generations)
        for i in range(n):
            start_idx = i * args.num_generations
            end_idx = (i + 1) * args.num_generations
            group_rewards = samples["rewards"][start_idx:end_idx]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
        
        samples["advantages"] = advantages
    else:
        advantages = (samples["rewards"] - gathered_reward.mean()) / (gathered_reward.std() + 1e-8)
        samples["advantages"] = advantages

    # Best-of-N选择（如果需要）
    if hasattr(args, 'bestofn') and args.bestofn < batch_size:
        total_scores = samples["advantages"]
        sorted_indices = torch.argsort(total_scores)
        top_indices = sorted_indices[-args.bestofn//2:]     
        bottom_indices = sorted_indices[:args.bestofn//2]     
        selected_indices = torch.cat([top_indices, bottom_indices])
        shuffled_order = torch.randperm(len(selected_indices), device=selected_indices.device)
        selected_indices = selected_indices[shuffled_order]
        
        for key in ["timesteps", "latents", "next_latents", "log_probs", "rewards", "advantages"]:
            samples[key] = samples[key][selected_indices]
        
        new_contexts = [contexts[i] for i in selected_indices.cpu().numpy()]
        new_context_null = [context_null[i] for i in selected_indices.cpu().numpy()]
        samples["contexts"] = new_contexts
        samples["context_null"] = new_context_null
        batch_size = len(selected_indices)

    # 时间步随机排列
    perms = torch.stack([
        torch.randperm(len(samples["timesteps"][0])) 
        for _ in range(batch_size)
    ]).to(device)
    
    for key in ["timesteps", "latents", "next_latents", "log_probs"]:
        samples[key] = samples[key][
            torch.arange(batch_size).to(device)[:, None],
            perms,
        ]
    
    # 构建batched样本
    samples_batched = {
        k: v.unsqueeze(1) if k in ["timesteps", "latents", "next_latents", "log_probs"] else v
        for k, v in samples.items()
    }
    
    # 构建样本列表
    samples_batched_list = []
    for i in range(batch_size):
        sample_dict = {
            "timesteps": samples_batched["timesteps"][i],
            "latents": samples_batched["latents"][i],
            "next_latents": samples_batched["next_latents"][i],
            "log_probs": samples_batched["log_probs"][i],
            "rewards": samples["rewards"][i],
            "advantages": samples["advantages"][i],
            "contexts": [samples["contexts"][i]],
            "context_null": [samples["context_null"][i]],
            "sigma_schedule": sigma_schedule,
        }
        samples_batched_list.append(sample_dict)
    
    # 训练循环 - 应用自适应权重调整策略
    train_timesteps = int(len(samples["timesteps"][0]) * args.timestep_fraction)
    grad_norm = None
    print(f"DEBUG: Processing {len(samples_batched_list)} samples")

    # 【自适应权重】预计算 timestep 权重：从 0.5 线性递增到 1.0
    timestep_weights = torch.linspace(1, 0.5, train_timesteps, device=device)
    print(f"DEBUG: Using adaptive timestep weights from 0.5 to 1.0 across {train_timesteps} steps")
    
    collect_time_start=time.time()
    
    for i, sample in enumerate(samples_batched_list):
        # 【自适应策略】收集每个 timestep 的 loss
        #sample_losses = []
        sample_loss_for_log = torch.zeros((),device=device)
        last_ratio = None
        last_weight = None
        for step_idx in range(train_timesteps):
            clip_range = args.clip_range
            adv_clip_max = args.adv_clip_max
            
            # 正确处理维度和索引
            current_latents = sample["latents"][0, step_idx]      # (C, T, H, W)
            next_latents = sample["next_latents"][0, step_idx]    # (C, T, H, W)
            current_timesteps = sample["timesteps"][0, step_idx]  # scalar
            current_log_probs = sample["log_probs"][0, step_idx]  # scalar
            
            # 确保latents维度正确
            if current_latents.shape[0] != 16:
                raise ValueError(f"Expected 16 channels, got {current_latents.shape[0]} channels")

            # 计算序列长度
            latent_shape = current_latents.shape  # (C, T, H, W)
            seq_len = math.ceil(
                (latent_shape[2] * latent_shape[3]) / (2 * 2) * latent_shape[1]
            )
            
            # GRPO单步
            new_log_probs = grpo_wan_one_step(
                args,
                current_latents,
                next_latents,
                sample["contexts"],
                sample["context_null"],
                seq_len,
                transformer,
                current_timesteps.unsqueeze(0),
                perms[i][step_idx],
                sigma_schedule, 
                getattr(args, 'guide_scale', 5.0),
            )

            # PPO loss计算
            advantages = torch.clamp(
                sample["advantages"],
                -adv_clip_max,
                adv_clip_max,
            )

            # timestep weight * ratio
            ratio = torch.exp(new_log_probs - current_log_probs)

            unclipped_loss = -advantages * ratio
            clipped_loss = -advantages * torch.clamp(
                ratio,
                1.0 - clip_range,
                1.0 + clip_range*2,
            )

            # 【自适应策略】计算当前 timestep 的损失
            timestep_loss = torch.maximum(unclipped_loss, clipped_loss).mean()
            
            weight = timestep_weights[step_idx]
            weight_timestep_loss =weight * timestep_loss
            loss_step = weight_timestep_loss / (args.gradient_accumulation_steps * train_timesteps)
            loss_step.backward()
            # 仅用于日志
            sample_loss_for_log += loss_step.detach()
            last_ratio = ratio.detach()
            last_weight = weight.detach()
        
        # 统计（样本级的时序平均损失，已包含1/train_timesteps缩放）
        avg_loss = sample_loss_for_log.clone()
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        total_loss += avg_loss.item()
        

        collect_time_end=time.time()
        print(f"训练用时：{collect_time_end-collect_time_start}s")
        
        if (i + 1) % args.gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
        # 修改打印信息
        if dist.get_rank() % 8 == 0:
            if getattr(args, 'use_file_based_reward', False):
                print("file_based reward", sample["rewards"].item())
            elif args.use_hpsv2:
                print("hps reward", sample["rewards"].item())
            else:
                print("qwen reward", sample["rewards"].item())
            if last_ratio is not None:    
                print("ratio", last_ratio.mean().item())
            print("advantage", sample["advantages"].item())
            print("avg loss (per-sample after time-avg)", sample_loss_for_log.item())
            if last_weight is not None:
                print(f"adaptive weight (last step): {last_weight.item():.3f}")

        dist.barrier()
    
    # 打印自适应策略统计信息
    if dist.get_rank() == 0:
        print(f"🎯 Adaptive training completed:")
        print(f"  - Used timestep weights: {timestep_weights[0]:.3f} → {timestep_weights[-1]:.3f}")
        print(f"  - Total samples processed: {len(samples_batched_list)}")
        print(f"  - Average loss: {total_loss:.6f}")
    
    return total_loss, grad_norm.item() if grad_norm is not None else 0.0, samples["rewards"]

def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

# 【新增】GPU分配逻辑
    if hasattr(args, 'train_gpus') and args.train_gpus:
        # 解析GPU列表
        train_gpu_list = list(map(int, args.train_gpus.split(",")))
        
        # 验证环境变量
        if "LOCAL_RANK" not in os.environ:
            raise ValueError("请使用 torchrun 启动分布式训练")
        
        # 获取分布式参数
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        # 验证GPU数量与进程数匹配
        if len(train_gpu_list) != world_size:
            raise ValueError(f"指定的GPU数量({len(train_gpu_list)})与world_size({world_size})不匹配")
        
        # 设置当前进程使用的GPU
        gpu_id = train_gpu_list[local_rank]
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        
        print(f"进程 rank={rank}, local_rank={local_rank} 使用 GPU {gpu_id}")
        
        # 初始化分布式进程组
        dist.init_process_group("nccl")
        
    else:
        # 原有的GPU分配逻辑（兼容性保持）
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = torch.cuda.current_device()
        
        print(f"使用默认GPU分配: rank={rank}, local_rank={local_rank}, device={device}")

    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed + rank)

    if args.use_bf16:
        model_dtype = torch.bfloat16
        autocast_dtype = torch.bfloat16
        precision_name = "BF16"
    else:
        model_dtype = torch.float32
        autocast_dtype = torch.float32
        precision_name = "FP32"

    # 创建输出目录
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    # 初始化reward跟踪器（只在rank 0上）
    reward_tracker = None
    if rank == 0:
        reward_tracker = RewardTracker(
            save_dir=os.path.join(args.output_dir, "realtime_plots"),
            save_every=1,  # 每步都保存
            max_points=10000  # 最多保存10000个数据点
        )
        print(f"Real-time reward tracker initialized, plots will be saved to: {reward_tracker.save_dir}")

    print(f"--> loading WAN model from {args.pretrained_model_name_or_path}")

    
    # 加载WAN配置
    from wan.configs import t2v_1_3B
    config = t2v_1_3B
    
    # 加载WAN模型 - 使用内存优化配置
    transformer = WanModel.from_pretrained(args.pretrained_model_name_or_path)

    # 根据args.use_bf16决定数据类型
    model_dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    transformer = transformer.to(dtype=model_dtype)
    transformer = transformer.to(device)
    
    # 添加参数量统计
    total_params = sum(p.numel() for p in transformer.parameters())
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    
    print(f"=== WAN Transformer Model Statistics ===")
    print(f"  Total parameters: {total_params / 1e9:.2f} B")
    print(f"  Trainable parameters: {trainable_params / 1e9:.2f} B")
    print(f"  Trainable ratio: {trainable_params / total_params * 100:.1f}%")
    print(f"  Model dtype: {model_dtype}")
    print(f"  Precision: {precision_name}")

    # 使用FSDP包装模型 - 优化配置
    from torch.distributed.fsdp.api import CPUOffload, MixedPrecision, BackwardPrefetch, ShardingStrategy
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy, transformer_auto_wrap_policy
    from torch.distributed._composable.fsdp import fully_shard
    
    # 配置CPU offload（如果启用）
    cpu_offload = None
    if args.cpu_offload:
        cpu_offload = CPUOffload(offload_params=True)
        print("CPU offload enabled for parameters")
    
    # 配置混合精度
    if args.use_bf16:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,      # 参数使用BF16
            reduce_dtype=torch.bfloat16,      # 梯度聚合使用FP32（更稳定）
            buffer_dtype=torch.bfloat16,     # buffer使用BF16
            cast_forward_inputs=True,        # 自动转换输入类型
        )
        print("Using BF16 mixed precision with FP32 gradient reduction")
    else:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
        print("Using FP32 precision")
    
    # 【关键修改】完全分片的包装策略 - 包装所有模块
    def wan_full_shard_wrap_policy(module, recurse, nonwrapped_numel):
        """
        WAN模型完全分片策略 - 包装所有可能的模块以实现参数完全平分
        """
        # 1. 降低参数阈值，包装更多模块
        if nonwrapped_numel >= 1e3:  # 10K参数就包装，而不是100K
            return True
        # 2.l 包装所有WAN的主要组件
        module_name = str(type(module).__name__)
        # 包装所有attention相关模块
        attention_modules = [
            'WanAttentionBlock', 'WanSelfAttention', 'WanT2VCrossAttention', 
            'WanI2VCrossAttention', 'attention', 'Attention'
        ]
        if any(name in module_name for name in attention_modules):
            return True
        # 包装所有Linear层（如果参数足够多）
        if module_name == 'Linear' and nonwrapped_numel >= 1e3:  # 5K参数以上的Linear层
            return True
        # 包装所有卷积层
        conv_modules = ['Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d']
        if any(name in module_name for name in conv_modules):
            return True
        # 包装embedding和projection层
        embed_modules = ['Embedding', 'MLPProj', 'Head']
        if any(name in module_name for name in embed_modules):
            return True
        # 包装normalization层（如果参数足够多）
        norm_modules = ['LayerNorm', 'WanLayerNorm', 'WanRMSNorm', 'RMSNorm', 'GroupNorm', 'BatchNorm']
        if any(name in module_name for name in norm_modules) and nonwrapped_numel >= 1e3:
            return True
        # 包装FFN/MLP相关
        ffn_modules = ['Sequential', 'GELU', 'SiLU', 'ReLU']
        if any(name in module_name for name in ffn_modules) and nonwrapped_numel >= 1e3:
            return True
        # 如果模块有超过1000个参数，就包装
        if nonwrapped_numel >= 1e3:
            return True
        return False

    # 【强制】使用FULL_SHARD策略
    if args.no_sharding:
        sharding_strategy = ShardingStrategy.NO_SHARD
        print("FSDP is running in NO_SHARD (DDP-like) mode. Sampling will be fast.")
    else:
        sharding_strategy = ShardingStrategy.FULL_SHARD
        print(f"Using ZeRO-3 (FULL_SHARD) - each GPU will store ~1/{world_size} of parameters. Sampling may be slow.")
    
    # 清理显存
    torch.cuda.empty_cache()
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy, transformer_auto_wrap_policy
    from torch.distributed._composable.fsdp import fully_shard

    # 【关键】导入梯度检查点包装
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing,
    )
    from wan.modules.model import WanAttentionBlock # 确保导入了核心模块
    check_fn = lambda m: isinstance(m, WanAttentionBlock)
    if args.enable_gradient_checkpointing:
        print("Enabling gradient checkpointing for WanAttentionBlock...")
        apply_activation_checkpointing(
            transformer,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=check_fn
        )

    print("Applying FSDP with optimized configuration...")
    transformer = FSDP(
        transformer,
        auto_wrap_policy=wan_full_shard_wrap_policy,
        mixed_precision=mixed_precision_policy,
        device_id=local_rank,
        cpu_offload=cpu_offload,
        sharding_strategy=sharding_strategy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        use_orig_params=False,
        sync_module_states=True,
        # forward_prefetch=True if args.enable_sequence_parallel else False,
        ignored_modules=None,
    )
    print("✓ FSDP successfully applied to transformer")
    # 【验证】检查分片效果
    if rank == 0:
        # 统计实际的参数分布
        local_params = sum(p.numel() for p in transformer.parameters() if p.is_meta == False)
        total_params_check = sum(p.numel() for name, p in transformer.named_parameters())
            
        print(f"=== Full Shard Verification ===")
        print(f"  Original total parameters: {total_params / 1e9:.2f}B")
        print(f"  Local parameters per GPU: {local_params / 1e9:.2f}B")
        print(f"  Target per GPU: {total_params / world_size / 1e9:.2f}B")
        print(f"  Sharding efficiency: {(local_params / (total_params / world_size)) * 100:.1f}%")
    
    # 清理显存
    torch.cuda.empty_cache()
    
    # 加载VAE - 如果启用fp16也使用fp16
    vae_dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    vae = WanVAE(
        vae_pth=os.path.join(args.pretrained_model_name_or_path, config.vae_checkpoint),
        device=device
    )
    vae.model = vae.model.to(dtype=vae_dtype)
    print(f"VAE dtype: {vae_dtype}")

    print(f"--> WAN model loaded with memory optimizations")

    # 【优化】预先计算全局不变的 context_null，然后删除T5编码器
    print("Pre-calculating global context_null...")
    text_encoder = T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=torch.device('cpu'),  # 始终在CPU上加载
        checkpoint_path=os.path.join(args.pretrained_model_name_or_path, config.t5_checkpoint),
        tokenizer_path=os.path.join(args.pretrained_model_name_or_path, config.t5_tokenizer),
    )
    
    neg_prompt = getattr(args, 'neg_prompt', "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
    
    # 只计算一次，batch size为1
    if args.use_bf16:
        text_encoder.model.to(device) # 临时移到GPU计算
        context_null_single = text_encoder([neg_prompt], device)
        text_encoder.model.cpu() # 移回CPU
    else:
        context_null_single = text_encoder([neg_prompt], torch.device('cpu'))
    
    # 将其移动到目标设备上，以便后续高效复制
    context_null_single = [t.to(device) for t in context_null_single]
    
    # 删除T5编码器以释放CPU内存
    del text_encoder
    torch.cuda.empty_cache() # 清理因T5移动产生的缓存
    print("✓ Global context_null pre-calculated and T5 encoder has been deleted.")

    reward_model = None
    tokenizer = None
    preprocess_val = None

    from vllm import LLM, SamplingParams
    qwen_model_path = "/gemini/space/Qwen/Qwen2___5-VL-72B-Instruct"
    sampling_params = SamplingParams(temperature=0.8, top_p=0.90)
    print(f"Loading Qwen model from {qwen_model_path}...")
    videos_dir = "/gemini/space/ljm/Wan2.1-main_72btest_for_more/Reward/videos"
    llm = LLM(model = qwen_model_path, tensor_parallel_size = 8, gpu_memory_utilization = 0.90, allowed_local_media_path = videos_dir,)
    print(f"Successfully load Qwen model")
    print("="*40)
    
    # 添加 HPSv2 初始化
    if args.use_hpsv2:
        print(f"hps")

    transformer.train()

    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    print("Using standard AdamW optimizer (no CPU offload to avoid gradient issues)")
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    # 学习率调度器
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=-1,
    )

    # 使用预处理数据集
    train_dataset = WanPreprocessedDataset(args.data_json_path)
    sampler = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed
    )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=wan_preprocessed_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    # 训练信息
    total_batch_size = args.train_batch_size * world_size * args.gradient_accumulation_steps
    print("***** Running WAN GRPO training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Dataloader size = {len(train_dataloader)}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps per epoch = {args.max_train_steps}")

    progress_bar = tqdm(
        range(0, 100000),
        initial=0,
        desc="Steps",
        disable=local_rank > 0,
    )

    # 创建数据加载器迭代器
    def get_loader_iterator():
        while True:
            for batch in train_dataloader:
                yield batch

    loader = get_loader_iterator()

    for epoch in range(1000000):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        for step in range(1, args.max_train_steps + 1):
            # 在第1步和每5步保存checkpoint
            args.current_step = step  # 添加这一行
            if step == 1 or step % 5 == 0:
                save_checkpoint_simple(
                    model=transformer,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    step=step,
                    epoch=epoch,
                    rank=rank,
                    output_dir=args.output_dir,
                    args=args
                )
            #save_checkpoint(transformer, rank, args.output_dir, step, epoch)
            # 执行训练步骤
            loss, grad_norm, batch_rewards = train_wan_one_step(
                args,
                llm,
                sampling_params,
                device, 
                transformer,
                vae,
                optimizer,
                lr_scheduler,
                loader,
                args.max_grad_norm,
                context_null_single,
                reward_model,
                tokenizer,
                preprocess_val,
                qwen_reward_model=None,
            )

            # 实时更新图表（只在rank 0上）
            if rank == 0 and reward_tracker is not None:
                # 计算 advantage 统计信息（仿照 train_grpo_wan_fsdp.py 的逻辑）
                if args.use_group:
                    n = len(batch_rewards) // args.num_generations
                    group_means = []
                    group_stds = []
                    advantages = []
                    for i in range(n):
                        start_idx = i * args.num_generations
                        end_idx = (i + 1) * args.num_generations
                        group_rewards = batch_rewards[start_idx:end_idx]
                        group_mean = group_rewards.mean().item()
                        group_std = group_rewards.std().item()
                        group_means.append(group_mean)
                        group_stds.append(group_std)
                        advantages.extend(((group_rewards - group_mean) / (group_std + 1e-8)).cpu().numpy())
                    
                    # 记录组平均值和标准差
                    group_mean = np.mean(group_means)
                    group_std = np.mean(group_stds)
                    advantage = np.mean(advantages)
                else:
                    # 不使用分组时的 advantage 计算
                    gathered_reward = gather_tensor(batch_rewards)
                    advantages = (batch_rewards - gathered_reward.mean()) / (gathered_reward.std() + 1e-8)
                    advantage = advantages.mean().item()
                
                reward_tracker.add_data(
                    step=step,
                    loss=loss,
                    reward_tensor=batch_rewards,
                    grad_norm=grad_norm,
                    advantage=advantage,  # 添加 advantage 参数
                )
                # 显示保存信息
                print(f"Step {step}: Plots saved to {reward_tracker.save_dir}")
            # 更新进度条
            if rank == 0:
                # 计算平均reward用于显示
                if batch_rewards is not None:
                    if hasattr(batch_rewards, 'mean'):
                        avg_reward = batch_rewards.mean().item()
                    else:
                        avg_reward = float(batch_rewards)
                else:
                    avg_reward = 0.0
                    
                progress_bar.set_postfix({
                    "loss": f"{loss:.4f}",
                    "grad_norm": f"{grad_norm:.4f}",
                    "reward": f"{avg_reward:.4f}",
                })
                progress_bar.update(1)

    # 训练结束时保存最终总结图表
    if rank == 0 and reward_tracker is not None:
        reward_tracker.save_summary_plot("final")
        print(f"Final training plots saved to {reward_tracker.save_dir}")
    
    # 清理分布式进程组
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 数据集和数据加载器
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=16)
    
    # 模型路径
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # 验证和日志
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    # 优化器和训练
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=10)
    parser.add_argument("--max_grad_norm", default=2.0, type=float)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # 学习率调度器
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)

    # GRPO训练参数
    parser.add_argument("--h", type=int, default=512, help="video height")
    parser.add_argument("--w", type=int, default=512, help="video width")
    parser.add_argument("--t", type=int, default=25, help="video length")
    parser.add_argument("--sampling_steps", type=int, default=20, help="sampling steps")
    parser.add_argument("--eta", type=float, default=0.3, help="noise eta")
    parser.add_argument("--sampler_seed", type=int, default=42, help="seed of sampler")
    parser.add_argument("--use_group", action="store_true", default=False)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--init_same_noise", action="store_true", default=False)
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--timestep_fraction", type=float, default=1.0)
    parser.add_argument("--clip_range", type=float, default=1e-4)
    parser.add_argument("--adv_clip_max", type=float, default=5.0)

    # 内存优化参数
    parser.add_argument("--use_bf16", action="store_true", default=False, 
                       help="Use BF16 mixed precision training")
    parser.add_argument("--cpu_offload", action="store_true", default=False,
                       help="Offload model parameters to CPU")
    parser.add_argument("--use_zero2", action="store_true", default=False,
                       help="Use ZeRO-2 instead of ZeRO-3")
    
    # 视频保存相关参数
    parser.add_argument("--video_fps", type=int, default=25, help="保存视频的帧率")
    parser.add_argument("--save_video_every_steps", type=int, default=100, help="每隔多少步保存一次视频")

    parser.add_argument("--save_plot_every_steps", type=int, default=50, help="每隔多少步保存一次reward图表")
    
    parser.add_argument("--guide_scale", type=float, default=5.0, 
                   help="Classifier-free guidance scale")
    parser.add_argument("--neg_prompt", type=str, 
                   default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                   help="Negative prompt for CFG")
    # 添加 HPSv2 相关参数
    parser.add_argument(
        "--use_hpsv2",
        action="store_true",
        default=False,
        help="whether use hpsv2 as reward model for video generation",
    )
    
    parser.add_argument(
        "--hps_aggregation",
        type=str,
        default="mean",
        choices=["mean", "max", "min", "weighted"],
        help="How to aggregate HPS scores across video frames: mean, max, min, or weighted (later frames have higher weight)",
    )

    # 序列并行和内存优化参数
    parser.add_argument("--use_sequential_cfg", action="store_true", default=False,
                       help="Use sequential CFG to save memory")
    parser.add_argument("--enable_gradient_checkpointing", action="store_true", default=True,
                       help="Enable gradient checkpointing")
    # 【新增参数】
    parser.add_argument("--no_sharding", action="store_true", default=False,
                       help="Run FSDP in DDP-like mode (NO_SHARD) for fast sampling, disables parameter sharding.")
    # 添加 Qwen 奖励模型参数
    parser.add_argument(
        "--use_qwen_reward",
        action="store_true",
        default=False,
        help="Use Qwen video quality reward model instead of detection reward"
    )
    
    parser.add_argument(
        "--qwen_reward_model_path",
        type=str,
        default="/gemini/space/Wan2.1-main/Qwen72b",
        help="Path to Qwen reward model"
    )
    parser.add_argument(
        "--use_file_based_reward",
        action="store_true",
        default=False,
        help="Use file-based communication for reward computation (cuda:7 service)"
    )
    parser.add_argument("--train_gpus", type=str, default="0,1,2,3,4,5",
            help="指定训练使用的GPU列表，用逗号分隔，如 '0,1,2,3,4,5'")
    args = parser.parse_args()
    main(args)
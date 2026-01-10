import argparse
import math
import os
from pathlib import Path
import time
from torch.utils.data import DataLoader
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data.distributed import DistributedSampler
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

# 导入WAN相关模块
import wan
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel  
from wan.modules.vae import WanVAE
from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")

def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)

def compute_video_color_diversity_reward(videos):
    """
    计算视频颜色多样性奖励 - 奖励颜色较少的视频（基于FLUX的color_diversity_reward）
    
    Args:
        videos: torch.Tensor, shape (B, C, T, H, W) 或 (C, T, H, W), 范围 [-1, 1]
    
    Returns:
        rewards: torch.Tensor, shape (B,), 奖励分数
    """
    # 处理不同的输入格式
    if videos.dim() == 4:  # (C, T, H, W)
        videos = videos.unsqueeze(0)  # -> (1, C, T, H, W)
    
    # 确保输入是5维的 (B, C, T, H, W)
    if videos.dim() != 5:
        raise ValueError(f"Expected 5D input (B, C, T, H, W), got {videos.dim()}D")
    
    # 将视频从 [-1, 1] 转换到 [0, 1]
    videos = (videos + 1.0) / 2.0
    videos = torch.clamp(videos, 0, 1)
    
    # 转换数据类型为float32，避免BFloat16转numpy的问题
    videos = videos.float()
    
    batch_size = videos.shape[0]
    rewards = []
    
    for b in range(batch_size):
        video = videos[b]  # (C, T, H, W)
        
        if video.shape[0] == 3:  # RGB视频
            frame_rewards = []
            
            # 对每一帧计算颜色多样性奖励
            for t in range(video.shape[1]):
                frame = video[:, t, :, :]  # (C, H, W)
                
                # 下采样帧以减少计算量
                frame_small = F.interpolate(
                    frame.unsqueeze(0), 
                    size=(64, 64), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                
                # 转换为HSV颜色空间来更好地分析颜色
                frame_np = frame_small.permute(1, 2, 0).cpu().numpy()
                frame_np = (frame_np * 255).astype(np.uint8)
                
                # 使用PIL转换为HSV
                pil_image = Image.fromarray(frame_np)
                hsv_image = pil_image.convert('HSV')
                hsv_np = np.array(hsv_image)
                
                # 分析色相(Hue)的多样性
                hue = hsv_np[:, :, 0]  # 色相通道
                saturation = hsv_np[:, :, 1]  # 饱和度通道
                
                # 初始化hist变量
                hist = np.array([])
                
                # 只考虑饱和度足够高的像素（忽略灰色像素）
                high_saturation_mask = saturation > 50  # 饱和度阈值
                
                if high_saturation_mask.sum() > 0:
                    valid_hues = hue[high_saturation_mask]
                    
                    # 计算色相的直方图
                    hist, _ = np.histogram(valid_hues, bins=36, range=(0, 360))  # 每10度一个bin
                    
                    # 计算非零bin的数量（不同颜色的数量）
                    num_colors = np.count_nonzero(hist)
                    
                    # 计算色相的标准差作为颜色分散度的指标
                    hue_std = np.std(valid_hues)
                    
                    # 奖励函数：颜色种类越少，奖励越高
                    # 基础奖励从颜色数量计算
                    color_reward = max(0, 10 - num_colors)  # 10种颜色以下有奖励
                    
                    # 根据色相标准差调整奖励
                    diversity_penalty = hue_std / 50.0  # 标准差越大，惩罚越大
                    
                    frame_reward = color_reward - diversity_penalty
                else:
                    # 如果帧主要是灰色，给予中等奖励
                    frame_reward = 5.0
                
                # 额外检查：如果帧主要是单一颜色，给予额外奖励
                # 计算主导颜色的比例
                if len(hist) > 0:  # 检查hist是否为空
                    dominant_color_ratio = np.max(hist) / np.sum(hist)
                    if dominant_color_ratio > 0.7:  # 如果某种颜色占70%以上
                        frame_reward += 3.0
                
                frame_rewards.append(frame_reward)
            
            # 对所有帧的奖励取平均值作为视频的最终奖励
            # 也可以选择取最小值、最大值或加权平均
            final_reward = np.mean(frame_rewards)
            
        else:
            # 如果不是RGB视频，给予默认奖励
            final_reward = 0.0
        
        rewards.append(final_reward)
    
    return torch.tensor(rewards, dtype=torch.float32, device=videos.device)


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

def wan_collate_function(batch):
    """WAN数据集的collate函数"""
    captions = [item['caption'] for item in batch]
    return captions

def run_wan_sample_step(
    args,
    latents,  # [(16, 7, 64, 64)]
    progress_bar,
    transformer,
    context,
    seq_len,
    grpo_sample,
):
    """WAN采样步骤 - 使用原生调度器"""
    
    # 使用WAN原生的调度器而不是简单的linspace
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    
    # 创建调度器 - 参考text2video.py的参数
    sample_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,  # 使用WAN的默认值
        shift=1,
        use_dynamic_shifting=False,
        solver_order=2,
        prediction_type="flow_prediction"
    )
    
    # 设置采样步数和时间偏移
    sample_scheduler.set_timesteps(
        args.sampling_steps, 
        device=latents[0].device, 
        shift=args.shift  # 使用args中的shift参数
    )
    
    timesteps = sample_scheduler.timesteps
    print(f"WAN timesteps: {timesteps}")
    
    if grpo_sample:
        all_latents = [latents[0]]  # 存储初始latent (16, 7, 64, 64)
        all_log_probs = []
        
        current_latents = latents[0]
        
        # 确定batch_size - 这个很重要！
        B = len(context) if isinstance(context, list) else context.shape[0]
        device = current_latents.device
        
        for i, t in enumerate(progress_bar):
            # 使用调度器的时间步
            timestep = torch.full([B], timesteps[i], device=device, dtype=torch.long)
            
            transformer.eval()
            with torch.autocast("cuda", torch.float32):
                # WAN模型前向传播
                pred = transformer(
                    x=[current_latents],  # 传入 (16, 7, 64, 64) 格式
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

            if grpo_sample:
                # 对于GRPO，我们需要手动计算以获得log概率
                if i == len(timesteps) - 1:
                    # 最后一步，直接使用调度器
                    next_latents = sample_scheduler.step(
                        model_output.unsqueeze(0), 
                        timesteps[i], 
                        current_latents.unsqueeze(0), 
                        return_dict=False
                    )[0].squeeze(0)
                    
                    # 最后一步的log概率设为0（因为没有噪声）- 修复：确保维度一致
                    log_prob = torch.zeros(B, device=device, dtype=torch.float32)
                else:
                    # 使用调度器计算确定性步骤
                    next_latents_det = sample_scheduler.step(
                        model_output.unsqueeze(0), 
                        timesteps[i], 
                        current_latents.unsqueeze(0), 
                        return_dict=False
                    )[0].squeeze(0)
                    
                    # 添加噪声以支持GRPO
                    # 计算噪声标准差（基于时间步差异）
                    if i + 1 < len(timesteps):
                        sigma_curr = timesteps[i].float() / 1000.0
                        sigma_next = timesteps[i + 1].float() / 1000.0
                        sigma_diff = abs(sigma_curr - sigma_next)
                        std_dev = args.eta * math.sqrt(sigma_diff)
                    else:
                        std_dev = 0.0
                    
                    # 添加噪声
                    if std_dev > 0:
                        noise = torch.randn_like(next_latents_det)
                        next_latents = next_latents_det + noise * std_dev
                        
                        # 计算log概率
                        log_prob_raw = (
                            -((next_latents.detach() - next_latents_det) ** 2) / (2 * std_dev**2)
                            - math.log(std_dev + 1e-8) 
                            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
                        )
                        # 修复：确保log_prob维度一致，在空间维度上求平均，保持batch维度
                        log_prob = log_prob_raw.mean(dim=tuple(range(1, log_prob_raw.ndim)))
                        
                        # 确保log_prob的维度是(B,)
                        if log_prob.dim() == 0:  # 如果是标量，扩展为(B,)
                            log_prob = log_prob.unsqueeze(0).expand(B)
                        elif log_prob.shape[0] != B:  # 如果batch维度不匹配
                            log_prob = log_prob.mean().unsqueeze(0).expand(B)
                    else:
                        next_latents = next_latents_det
                        log_prob = torch.zeros(B, device=device, dtype=torch.float32)
                
                print(f"Step {i}: log_prob= {log_prob}")
                
            else:
                # 非GRPO模式，直接使用调度器
                next_latents = sample_scheduler.step(
                    model_output.unsqueeze(0), 
                    timesteps[i], 
                    current_latents.unsqueeze(0), 
                    return_dict=False
                )[0].squeeze(0)
                log_prob = None
            
            current_latents = next_latents
            all_latents.append(current_latents)
            if log_prob is not None:
                all_log_probs.append(log_prob)
        
        # 最终结果
        final_latents = current_latents
        
        # 堆叠结果 - 注意维度
        all_latents = torch.stack(all_latents, dim=0)  # (num_steps+1, 16, 7, 64, 64)
        
        if all_log_probs:
            # 修复：确保所有log_prob都有相同的形状
            print(f"all_log_probs shapes: {[lp.shape for lp in all_log_probs]}")
            
            # 检查并修复不一致的维度
            consistent_log_probs = []
            for lp in all_log_probs:
                if lp.dim() == 0:  # 标量
                    lp = lp.unsqueeze(0).expand(B)
                elif lp.shape[0] != B:  # batch维度不匹配
                    lp = lp.mean().unsqueeze(0).expand(B)
                consistent_log_probs.append(lp)
            
            all_log_probs = torch.stack(consistent_log_probs, dim=0)  # (num_steps, B)
        else:
            all_log_probs = torch.zeros(len(all_latents)-1, B, device=device, dtype=torch.float32)
        
        print(f"WAN final shapes: all_latents={all_latents.shape}, all_log_probs={all_log_probs.shape}")
        
        return [final_latents], final_latents, all_latents, all_log_probs
    
    else:
        # 非GRPO模式的简化实现
        current_latents = latents[0]
        
        for i, t in enumerate(progress_bar):
            B = len(context) if isinstance(context, list) else context.shape[0]
            timestep = torch.full([B], timesteps[i], device=current_latents.device, dtype=torch.long)
            
            with torch.autocast("cuda", torch.float32):
                pred = transformer(
                    x=[current_latents],
                    t=timestep,
                    context=context,
                    seq_len=seq_len
                )
                
                if isinstance(pred, dict) and 'rgb' in pred:
                    model_output = pred['rgb'][0]
                elif isinstance(pred, list):
                    model_output = pred[0]
                else:
                    model_output = pred
            
            # 使用调度器步骤
            current_latents = sample_scheduler.step(
                model_output.unsqueeze(0), 
                timesteps[i], 
                current_latents.unsqueeze(0), 
                return_dict=False
            )[0].squeeze(0)
        
        return [current_latents], current_latents, None, None

def grpo_wan_one_step(
    args,
    latents,         # (1, 16, 7, 64, 64)
    pre_latents,     # (1, 16, 7, 64, 64)
    context,
    seq_len,
    transformer,
    timesteps,
    i,
):
    """GRPO的单步训练 - 使用WAN原生调度器重新计算"""
    B = len(context) if isinstance(context, list) else context.shape[0]
    transformer.train()
    
    # 确保latents维度正确：(16, 7, 64, 64)
    if latents.dim() == 5:  # (1, 16, 7, 64, 64)
        latents = latents.squeeze(0)  # -> (16, 7, 64, 64)
    
    if pre_latents.dim() == 5:
        pre_latents = pre_latents.squeeze(0)  # -> (16, 7, 64, 64)
    
    # 检查通道数
    if latents.shape[0] != 16:
        raise ValueError(f"Expected 16 channels, got {latents.shape[0]} channels")
    
    with torch.autocast("cuda", torch.float32):
        pred = transformer(
            x=[latents],  # [tensor(16, 7, 64, 64)]
            t=timesteps,
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

    # 创建临时调度器来重新计算log概率
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    
    temp_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1,
        use_dynamic_shifting=False,
        solver_order=2,
        prediction_type="flow_prediction"
    )
    temp_scheduler.set_timesteps(
        args.sampling_steps, 
        device=latents.device, 
        shift=args.shift
    )
    
    # 计算确定性步骤
    current_timestep = timesteps[0].item()  # 假设batch中所有timestep相同
    timestep_index = None
    for idx, t in enumerate(temp_scheduler.timesteps):
        if abs(t.item() - current_timestep) < 1:  # 找到最接近的timestep
            timestep_index = idx
            break
    
    if timestep_index is None:
        # 如果找不到对应的timestep，使用简化计算
        log_prob = torch.zeros(B, device=latents.device, dtype=torch.float32)
    else:
        # 使用调度器计算确定性结果
        temp_scheduler._step_index = timestep_index
        next_latents_det = temp_scheduler.step(
            model_output.unsqueeze(0), 
            temp_scheduler.timesteps[timestep_index], 
            latents.unsqueeze(0), 
            return_dict=False
        )[0].squeeze(0)
        
        # 计算噪声参数
        if timestep_index + 1 < len(temp_scheduler.timesteps):
            sigma_curr = temp_scheduler.timesteps[timestep_index].float() / 1000.0
            sigma_next = temp_scheduler.timesteps[timestep_index + 1].float() / 1000.0
            sigma_diff = abs(sigma_curr - sigma_next)
            std_dev = args.eta * math.sqrt(sigma_diff)
        else:
            std_dev = 0.0
        
        # 计算log概率
        if std_dev > 0:
            log_prob_raw = (
                -((pre_latents.detach() - next_latents_det) ** 2) / (2 * std_dev**2)
                - math.log(std_dev + 1e-8) 
                - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
            )
            # 修复：确保log_prob维度一致
            log_prob = log_prob_raw.mean(dim=tuple(range(1, log_prob_raw.ndim)))
            
            # 确保log_prob的维度是(B,)
            if log_prob.dim() == 0:  # 如果是标量，扩展为(B,)
                log_prob = log_prob.unsqueeze(0).expand(B)
            elif log_prob.shape[0] != B:  # 如果batch维度不匹配
                log_prob = log_prob.mean().unsqueeze(0).expand(B)
        else:
            log_prob = torch.zeros(B, device=latents.device, dtype=torch.float32)
    
    return log_prob

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
            print(f"First frame saved: {image_path}")
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
            print(f"Error saving video {video_path}: {e}")
            # 如果视频保存失败，至少保存第一帧作为图像
            first_frame = video_np[0]  # (H, W, C)
            if C == 3:
                first_frame_pil = Image.fromarray(first_frame)
            else:
                first_frame_pil = Image.fromarray(first_frame[:,:,0], mode='L')
            
            image_filename = f"wan_frame_rank{rank}_batch{index}_{timestamp}.png"
            image_path = os.path.join("./images", image_filename)
            first_frame_pil.save(image_path)
            print(f"First frame saved as image: {image_path}")
        
        # 保存对应的prompt
        prompt_filename = f"wan_prompt_rank{rank}_batch{index}.txt"
        prompt_path = os.path.join("./videos", prompt_filename)
        # try:
        #     with open(prompt_path, 'w', encoding='utf-8') as f:
        #         f.write(f"Video: {video_filename}\n")
        #         f.write(f"Timestamp: {timestamp}\n")
        #         f.write(f"Rank: {rank}, Batch: {index}\n")
        #         f.write(f"Video shape: (C={C}, T={T}, H={H}, W={W})\n")
        #         f.write(f"Prompt: {caption}\n")
                
        #         # 如果有额外的配置信息，也可以保存
        #         f.write(f"\n=== Generation Config ===\n")
        #         f.write(f"Sampling steps: {args.sampling_steps}\n")
        #         f.write(f"ETA: {args.eta}\n")
        #         f.write(f"Shift: {args.shift}\n")
        #         f.write(f"Video size: {args.w}x{args.h}x{args.t}\n")
            
        #     print(f"Prompt saved: {prompt_path}")
            
        # except Exception as e:
        #     print(f"Error saving prompt {prompt_path}: {e}")
    
    else:
        print(f"Unexpected video_frames shape: {video_frames.shape}")

def create_video_grid(video_list, captions_list, output_path, fps=8):
    """
    创建视频网格，将多个视频排列在一起
    
    Args:
        video_list: List[torch.Tensor], 每个tensor shape (C, T, H, W)
        captions_list: List[str], 对应的caption列表
        output_path: str, 输出视频路径
        fps: int, 视频帧率
    """
    if not video_list:
        return
    
    # 确定网格大小
    num_videos = len(video_list)
    grid_size = int(np.ceil(np.sqrt(num_videos)))
    
    # 获取视频尺寸
    C, T, H, W = video_list[0].shape
    
    # 创建网格画布
    grid_H = H * grid_size
    grid_W = W * grid_size
    
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (grid_W, grid_H))
        
        for t in range(T):
            # 创建当前帧的网格
            grid_frame = np.zeros((grid_H, grid_W, 3), dtype=np.uint8)
            
            for i, video in enumerate(video_list):
                row = i // grid_size
                col = i % grid_size
                
                # 获取当前视频的当前帧
                frame = video[:, t, :, :].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
                frame = (frame * 255).astype(np.uint8)
                
                if C == 1:
                    frame = np.repeat(frame, 3, axis=-1)
                
                # 将帧放置到网格中
                start_h = row * H
                end_h = start_h + H
                start_w = col * W
                end_w = start_w + W
                
                grid_frame[start_h:end_h, start_w:end_w] = frame
            
            # 转换为BGR格式并写入视频
            grid_frame_bgr = cv2.cvtColor(grid_frame, cv2.COLOR_RGB2BGR)
            out.write(grid_frame_bgr)
        
        out.release()
        print(f"Video grid saved: {output_path}")
        
        # 保存对应的captions
        caption_path = output_path.replace('.mp4', '_captions.txt')
        with open(caption_path, 'w', encoding='utf-8') as f:
            for i, caption in enumerate(captions_list):
                f.write(f"Video {i+1}: {caption}\n")
        
    except Exception as e:
        print(f"Error creating video grid: {e}")

def sample_wan_reference_model(
    args,
    device, 
    transformer,
    vae,
    contexts,  # 现在接收List[Tensor]格式
    captions,  # 保留captions用于调试
):
    """WAN参考模型采样 - 使用原生调度器"""
    # 视频参数
    frame_num = args.t
    size = (args.w, args.h)
    
    # 创建sigma调度器以获取sigma_schedule
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    
    main_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1,
        use_dynamic_shifting=False,
        solver_order=2,
        prediction_type="flow_prediction"
    )
    
    main_scheduler.set_timesteps(
        args.sampling_steps, 
        device=device, 
        shift=args.shift
    )
    
    # 获取sigma_schedule用于后续使用
    sigma_schedule = main_scheduler.sigmas  # 这是实际的sigma值
    print(f"sigma_schedule shape: {sigma_schedule.shape}, values: {sigma_schedule}")
    
    B = len(captions)
    batch_size = 1
    batch_indices = torch.chunk(torch.arange(B), B // batch_size)

    all_latents = []
    all_log_probs = []
    all_rewards = []

    # VAE参数
    vae_stride = [4, 8, 8]  # WAN的VAE下采样率
    patch_size = [1, 2, 2]  # WAN的patch size
    
    if args.init_same_noise:
        # 计算latent尺寸
        latent_shape = (
            16,  # WAN使用16通道
            (frame_num - 1) // vae_stride[0] + 1,
            size[1] // vae_stride[1],
            size[0] // vae_stride[2]
        )
        input_latents = torch.randn(latent_shape, device=device, dtype=torch.float32)
        print(f"WAN input_latents shape: {input_latents.shape}")

    for index, batch_idx in enumerate(batch_indices):
        batch_captions = [captions[i] for i in batch_idx]
        
        # 修复：正确提取batch的contexts，保持List[Tensor]格式
        batch_contexts = [contexts[i].to(device) for i in batch_idx]  # List[Tensor]
        
        if not args.init_same_noise:
            latent_shape = (
                16,
                (frame_num - 1) // vae_stride[0] + 1,
                size[1] // vae_stride[1],
                size[0] // vae_stride[2]
            )
            input_latents = torch.randn(latent_shape, device=device, dtype=torch.float32)

        # 计算序列长度
        seq_len = math.ceil(
            (latent_shape[2] * latent_shape[3]) / (patch_size[1] * patch_size[2]) * latent_shape[1]
        )

        grpo_sample = True
        progress_bar = tqdm(range(0, args.sampling_steps), desc="WAN Sampling Progress")
        with torch.no_grad():
            _, final_latents, batch_latents, batch_log_probs = run_wan_sample_step(
                args,
                [input_latents],
                progress_bar,
                transformer,
                batch_contexts,  # 传入List[Tensor]格式
                seq_len,
                grpo_sample,
            )

        print(f"WAN sampling results:")
        print(f"  batch_latents shape: {batch_latents.shape}")  # 应该是 (num_steps+1, 16, 7, 64, 64)
        print(f"  batch_log_probs shape: {batch_log_probs.shape}")  # 应该是 (num_steps, B)

        # 调整维度以匹配训练需要
        # 从 (num_steps+1, 16, 7, 64, 64) 转换为 (1, num_steps+1, 16, 7, 64, 64)
        batch_latents = batch_latents.unsqueeze(0)  # 添加batch维度
        
        # 修复：处理log_probs的维度
        # 如果batch_log_probs是 (8, 1)，需要转换为 (1, 8)
        if batch_log_probs.dim() == 2 and batch_log_probs.shape[1] == 1:
            batch_log_probs = batch_log_probs.transpose(0, 1)  # (8, 1) -> (1, 8)
        else:
            batch_log_probs = batch_log_probs.unsqueeze(0)  # 添加batch维度
        
        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)

        # VAE解码部分保持不变...
        rank = int(os.environ.get("RANK", 0))
        
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.float32):
                # WAN VAE解码，输入是 (16, 7, 64, 64)
                decoded_videos = vae.decode([final_latents])
                video_frames = decoded_videos[0]  # (C, T, H, W)
                
                # 后处理
                video_frames = (video_frames + 1.0) / 2.0  # [-1, 1] -> [0, 1]
                video_frames = torch.clamp(video_frames, 0, 1)
                
                # 创建输出目录
                os.makedirs("./videos", exist_ok=True)
                os.makedirs("./images", exist_ok=True)
                
                # 保存视频
                save_video_and_prompt(
                    video_frames,
                    batch_captions[0],  # 当前batch的caption
                    rank,
                    index,
                    args
                )

        # # 计算奖励
        # if args.use_simple_color_reward:
        #     color_reward = compute_simple_color_reward(video_frames[:, :1, :, :])  # 只用第一帧
        color_reward = compute_video_color_diversity_reward(video_frames.unsqueeze(0))  # 添加batch维度
        all_rewards.append(color_reward)

    # 拼接结果
    if len(all_latents) > 1:
        all_latents = torch.cat(all_latents, dim=0)  # (B, num_steps+1, 16, 7, 64, 64)
        all_log_probs = torch.cat(all_log_probs, dim=0)  # (B, num_steps)
        all_rewards = torch.cat(all_rewards, dim=0)
    else:
        all_latents = all_latents[0]  # (1, num_steps+1, 16, 7, 64, 64)
        all_log_probs = all_log_probs[0]  # (1, num_steps)
        all_rewards = all_rewards[0]
    
    print(f"Final WAN sampling results:")
    print(f"  all_latents: {all_latents.shape}")     # (B, num_steps+1, 16, 7, 64, 64)
    print(f"  all_log_probs: {all_log_probs.shape}") # (B, num_steps)
    print(f"  all_rewards: {all_rewards.shape}")     # (B,)
    
    # 修复：返回实际的sigma_schedule而不是None
    return all_rewards, all_latents, all_log_probs, sigma_schedule

def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)

def train_wan_one_step(
    args,
    device,
    transformer,
    vae,
    optimizer,
    lr_scheduler,
    loader,
    max_grad_norm,
):
    """WAN的一步训练，处理(B, T, C, F, H, W)维度"""
    total_loss = 0.0
    optimizer.zero_grad()
    
    batch = next(loader)
    contexts = batch['contexts']  # List[Tensor]格式
    captions = batch['captions']  # 原始文本
    
    if args.use_group:
        # 扩展contexts和captions
        expanded_contexts = []
        expanded_captions = []
        for context, caption in zip(contexts, captions):
            for _ in range(args.num_generations):
                expanded_contexts.append(context)
                expanded_captions.append(caption)
        contexts = expanded_contexts
        captions = expanded_captions

    # 采样参考模型
    reward, all_latents, all_log_probs, sigma_schedule = sample_wan_reference_model(
        args, device, transformer, vae, contexts, captions,
    )
    
    print(f"WAN training data shapes:")
    print(f"  all_latents: {all_latents.shape}")     # (B, 9, 16, 7, 64, 64)
    print(f"  all_log_probs: {all_log_probs.shape}") # (B, 8)
    print(f"  reward: {reward.shape}")               # (B,)
    print(f"  sigma_schedule type: {type(sigma_schedule)}, shape: {sigma_schedule.shape if sigma_schedule is not None else 'None'}")
    
    batch_size = all_latents.shape[0]
    
    # 修复：使用sigma_schedule创建timestep_values
    if sigma_schedule is not None:
        # 从sigma_schedule生成timestep values
        timestep_value = [int(sigma.item() * 1000) for sigma in sigma_schedule[:args.sampling_steps]]
    else:
        # 备用方案：使用线性插值
        timestep_value = [int(1000 * (1.0 - i / args.sampling_steps)) for i in range(args.sampling_steps)]
    
    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    timesteps_tensor = torch.tensor(timestep_values, device=all_latents.device, dtype=torch.long)
    
    # print(f"timestep_values shape: {len(timestep_values)} x {len(timestep_values[0])}")
    # print(f"timesteps_tensor shape: {timesteps_tensor.shape}")
    
    samples = {
        "timesteps": timesteps_tensor[:, :-1],
        "latents": all_latents[:, :-1],
        "next_latents": all_latents[:, 1:],
        "log_probs": all_log_probs,
        "rewards": reward.to(torch.float32),
    }
    
    # 计算advantages等保持不变...
    gathered_reward = gather_tensor(samples["rewards"])
    if dist.get_rank() == 0:
        print("gathered_color_reward", gathered_reward)
        with open('./wan_color_reward.txt', 'a') as f: 
            f.write(f"{gathered_reward.mean().item()}\n")

    if args.use_group:
        n = len(samples["rewards"]) // args.num_generations
        advantages = torch.zeros_like(samples["rewards"])
        
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

    # 其余训练逻辑保持不变...
    perms = torch.stack([
        torch.randperm(len(samples["timesteps"][0])) 
        for _ in range(batch_size)
    ]).to(device)
    
    for key in ["timesteps", "latents", "next_latents", "log_probs"]:
        samples[key] = samples[key][
            torch.arange(batch_size).to(device)[:, None],
            perms,
        ]
    
    samples_batched = {k: v.unsqueeze(1) for k, v in samples.items() if isinstance(v, torch.Tensor)}
    
    samples_batched_list = []
    for i in range(batch_size):
        sample_dict = {}
        for key, value in samples_batched.items():
            sample_dict[key] = value[i]
        
        # 修复：传入List[Tensor]格式的context
        sample_dict["contexts"] = [contexts[i]]  # List[Tensor]格式
        sample_dict["captions"] = [captions[i % len(captions)]]
        samples_batched_list.append(sample_dict)
    
    train_timesteps = int(len(samples["timesteps"][0]) * args.timestep_fraction)
    grad_norm = None
    
    for i, sample in enumerate(samples_batched_list):
        for step_idx in range(train_timesteps):
            clip_range = args.clip_range
            adv_clip_max = args.adv_clip_max
            
            # 直接使用预处理的context List[Tensor]
            context = [tensor.to(device) for tensor in sample["contexts"]]
            
            latent_shape = sample["latents"][:, step_idx].shape
            seq_len = math.ceil(
                (latent_shape[3] * latent_shape[4]) / (2 * 2) * latent_shape[2]
            )
            
            new_log_probs = grpo_wan_one_step(
                args,
                sample["latents"][:, step_idx],
                sample["next_latents"][:, step_idx],
                context,  # List[Tensor]格式
                seq_len,
                transformer,
                sample["timesteps"][:, step_idx],
                perms[i][step_idx],
            )

            # 其余训练逻辑保持不变...
            advantages = torch.clamp(
                sample["advantages"],
                -adv_clip_max,
                adv_clip_max,
            )

            ratio = torch.exp(new_log_probs - sample["log_probs"][:, step_idx])

            unclipped_loss = -advantages * ratio
            clipped_loss = -advantages * torch.clamp(
                ratio,
                1.0 - clip_range,
                1.0 + clip_range,
            )
            loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (args.gradient_accumulation_steps * train_timesteps)

            loss.backward()
            avg_loss = loss.detach().clone()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            total_loss += avg_loss.item()
            
        if (i + 1) % args.gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
        if dist.get_rank() % 8 == 0:
            print("color reward", sample["rewards"].item())
            print("ratio", ratio.mean().item())
            print("advantage", sample["advantages"].item())
            print("final loss", loss.item())
        
        dist.barrier()
    
    return total_loss, grad_norm.item() if grad_norm is not None else 0.0

def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed + rank)

    # 创建输出目录
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    print(f"--> loading WAN model from {args.pretrained_model_name_or_path}")
    
    # 加载WAN配置
    from wan.configs import t2v_1_3B
    config = t2v_1_3B
    
    # 加载WAN模型
    transformer = WanModel.from_pretrained(args.pretrained_model_name_or_path)
    transformer = transformer.to(dtype=torch.float32)
    # 重要：在FSDP包装之前，先将模型移动到正确的GPU设备
    transformer = transformer.to(device)
    # 添加参数量统计
    total_params = sum(p.numel() for p in transformer.parameters())
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print(f"=== WAN Transformer Model Statistics ===")
    print(f"  Total parameters: {total_params / 1e9:.2f} B")
    print(f"  Trainable parameters: {trainable_params / 1e9:.2f} B")
    print(f"  Non-trainable parameters: {non_trainable_params / 1e9:.2f} B")
    print(f"  Trainable ratio: {trainable_params / total_params * 100:.1f}%")

    # 使用FSDP包装模型，并指定正确的设备策略
    from torch.distributed.fsdp.api import CPUOffload, MixedPrecision
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    import torch.distributed.fsdp as fsdp
    
    # 配置FSDP策略
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    
    transformer = FSDP(
        transformer,
        auto_wrap_policy=None,
        mixed_precision=mixed_precision_policy,
        device_id=local_rank,  # 明确指定设备ID
        cpu_offload=None,      # 不使用CPU offload
    )
    # 确保模型在正确的设备上
    print(f"Transformer model is on device: {next(transformer.parameters()).device}")
    
    # 加载VAE
    vae = WanVAE(
        vae_pth=os.path.join(args.pretrained_model_name_or_path, config.vae_checkpoint),
        device=device
    )
    vae.model = vae.model.to(dtype=torch.float32)

    print(f"--> WAN model loaded")

    # 设置模型为训练模式
    transformer.train()

    # 优化器
    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    print(f"optimizer: {optimizer}")

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
        collate_fn=wan_preprocessed_collate_function,  # 使用修复的collate函数
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    # if rank <= 0:
    #     project = "wan_grpo"
    #     wandb.init(project=project, config=args)

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
    step_times = deque(maxlen=100)

    for epoch in range(1000000):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        for step in range(1, args.max_train_steps + 1):
            start_time = time.time()
            
            if step % args.checkpointing_steps == 0:
                # 保存检查点
                save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
                os.makedirs(save_path, exist_ok=True)
                
                if rank == 0:
                    # 保存模型状态
                    model_state = transformer.state_dict()
                    torch.save({
                        'step': step,
                        'model_state_dict': model_state,
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, os.path.join(save_path, "model.pth"))
                    
                    print(f"Checkpoint saved to {save_path}")
                
                dist.barrier()

            loss, grad_norm = train_wan_one_step(
                args,
                device, 
                transformer,
                vae,
                optimizer,
                lr_scheduler,
                loader,
                args.max_grad_norm,
            )

            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)

            progress_bar.set_postfix({
                "loss": f"{loss:.4f}",
                "step_time": f"{step_time:.2f}s",
                "grad_norm": grad_norm,
            })
            progress_bar.update(1)
            
            # if rank <= 0:
            #     wandb.log({
            #         "train_loss": loss,
            #         "learning_rate": lr_scheduler.get_last_lr()[0],
            #         "step_time": step_time,
            #         "avg_step_time": avg_step_time,
            #         "grad_norm": grad_norm,
            #     }, step=step)

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
    parser.add_argument("--checkpointing_steps", type=int, default=500)

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
    # parser.add_argument("--use_simple_color_reward", action="store_true", default=False)

    # 视频保存相关参数
    parser.add_argument("--video_fps", type=int, default=8, help="保存视频的帧率")
    parser.add_argument("--save_video_every_steps", type=int, default=100, help="每隔多少步保存一次视频")
    parser.add_argument("--max_videos_per_checkpoint", type=int, default=4, help="每个检查点最多保存多少个视频")

    args = parser.parse_args()
    main(args)
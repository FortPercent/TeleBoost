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
import os
# 导入WAN相关模块
import wan
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel  
from wan.modules.vae import WanVAE
from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler

from fastvideo.reward_tracker import RewardTracker,create_reward_summary_plot

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")

def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)

def compute_red_intensity_reward(videos):
    """
    计算红色强度奖励 - 奖励包含更多红色的视频
    这个reward会让生成的视频倾向于包含更多红色元素，效果非常明显
    
    Args:
        videos: torch.Tensor, shape (B, C, T, H, W) 或 (C, T, H, W), 范围 [0, 1]
    
    Returns:
        rewards: torch.Tensor, shape (B,), 奖励分数
    """
    # 处理不同的输入格式
    if videos.dim() == 4:  # (C, T, H, W)
        videos = videos.unsqueeze(0)  # -> (1, C, T, H, W)
    
    # 确保输入是5维的 (B, C, T, H, W)
    if videos.dim() != 5:
        raise ValueError(f"Expected 5D input (B, C, T, H, W), got {videos.dim()}D")
    
    # 确保视频在[0, 1]范围内
    videos = torch.clamp(videos, 0, 1)
    videos = videos.float()
    
    batch_size = videos.shape[0]
    rewards = []
    
    for b in range(batch_size):
        video = videos[b]  # (C, T, H, W)
        C, T, H, W = video.shape
        
        if C < 3:  # 如果不是RGB视频，给予最低奖励
            rewards.append(0.0)
            continue
        
        frame_red_scores = []
        
        for t in range(T):
            frame = video[:, t, :, :]  # (C, H, W)
            
            # 提取RGB通道
            red_channel = frame[0]    # (H, W)
            green_channel = frame[1]  # (H, W)
            blue_channel = frame[2]   # (H, W)
            
            # 1. 基础红色强度（红色通道的平均值）
            red_intensity = red_channel.mean().item()
            
            # 2. 红色纯度（红色相对于绿色和蓝色的优势）
            # 计算每个像素的红色优势：R - max(G, B)
            red_dominance = red_channel - torch.maximum(green_channel, blue_channel)
            red_dominance = torch.clamp(red_dominance, 0, 1)  # 只保留正值
            red_purity = red_dominance.mean().item()
            
            # 3. 鲜艳红色像素比例
            # 定义鲜艳红色：R > 0.7 且 R > G+0.3 且 R > B+0.3
            bright_red_mask = (
                (red_channel > 0.7) & 
                (red_channel > green_channel + 0.3) & 
                (red_channel > blue_channel + 0.3)
            )
            bright_red_ratio = bright_red_mask.float().mean().item()
            
            # 4. 深红色像素比例
            # 定义深红色：R > 0.4 且 R > 1.5*G 且 R > 1.5*B
            deep_red_mask = (
                (red_channel > 0.4) & 
                (red_channel > green_channel * 1.5) & 
                (red_channel > blue_channel * 1.5)
            )
            deep_red_ratio = deep_red_mask.float().mean().item()
            
            # 5. 红色区域连通性奖励
            # 奖励大块的红色区域而不是零散的红色像素
            red_regions_score = 0.0
            if bright_red_mask.sum() > 0:
                # 简单的连通性评估：计算红色像素的空间聚集度
                # 使用形态学操作来评估连通性
                kernel_size = 3
                padding = kernel_size // 2
                
                # 扩张操作
                bright_red_float = bright_red_mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                kernel = torch.ones(1, 1, kernel_size, kernel_size, device=videos.device) / (kernel_size * kernel_size)
                dilated = F.conv2d(bright_red_float, kernel, padding=padding)
                
                # 连通性分数：扩张后高值区域的比例
                connectivity = (dilated > 0.3).float().mean().item()
                red_regions_score = connectivity * bright_red_ratio  # 连通性 × 红色比例
            
            # 6. 综合当前帧的红色分数
            frame_score = (
                red_intensity * 30 +      # 30% 基础红色强度
                red_purity * 40 +         # 40% 红色纯度（最重要）
                bright_red_ratio * 50 +   # 50分 鲜艳红色比例
                deep_red_ratio * 30 +     # 30分 深红色比例
                red_regions_score * 20    # 20分 红色区域连通性
            )
            
            frame_red_scores.append(frame_score)
        
        # 7. 计算整个视频的红色奖励
        if frame_red_scores:
            # 平均红色分数
            avg_red_score = np.mean(frame_red_scores)
            
            # 最高红色分数（奖励峰值红色）
            max_red_score = np.max(frame_red_scores)
            
            # 红色一致性（奖励持续的红色）
            red_consistency = 1.0 / (1.0 + np.std(frame_red_scores))  # 方差越小，一致性越高
            
            # 综合视频红色奖励
            video_red_reward = (
                avg_red_score * 0.6 +      # 60% 平均红色分数
                max_red_score * 0.3 +      # 30% 峰值红色分数
                red_consistency * 10       # 10分 红色一致性奖励
            )
            
            # 额外奖励：如果红色非常突出，给予奖励加成
            if avg_red_score > 10:  # 如果平均红色分数很高
                video_red_reward += 15  # 额外奖励
            
            if max_red_score > 20:  # 如果有帧红色非常突出
                video_red_reward += 10  # 额外峰值奖励
            
            # 设置合理的上限
            video_red_reward = min(video_red_reward, 100.0)
        else:
            video_red_reward = 0.0
        
        rewards.append(video_red_reward)
    
    return torch.tensor(rewards, dtype=torch.float32, device=videos.device)
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
        
        # 保存对应的prompt
        # prompt_filename = f"wan_prompt_rank{rank}_batch{index}.txt"
        # prompt_path = os.path.join("./videos", prompt_filename)
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


def run_wan_sample_step(
    args,
    latents,  # [(16, 7, 64, 64)]
    progress_bar, 
    sigma_schedule,  # 添加sigma_schedule
    transformer,
    context,
    context_null,
    seq_len,
    grpo_sample,
    guide_scale=5.0,
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
                # 正向预测（有条件）
                pred_cond = transformer(
                    x=latents,
                    t=timestep,
                    context=context,
                    seq_len=seq_len
                )
                
                # 负向预测（无条件）
                pred_uncond = transformer(
                    x=latents,
                    t=timestep,
                    context=context_null,
                    seq_len=seq_len
                )
                
                # 处理模型输出
                if isinstance(pred_cond, dict) and 'rgb' in pred_cond:
                    model_output_cond = pred_cond['rgb'][0]
                    model_output_uncond = pred_uncond['rgb'][0]
                elif isinstance(pred_cond, list):
                    model_output_cond = pred_cond[0]
                    model_output_uncond = pred_uncond[0]
                else:
                    model_output_cond = pred_cond
                    model_output_uncond = pred_uncond
                
                # CFG组合：uncond + guide_scale * (cond - uncond)
                model_output = model_output_uncond + guide_scale * (model_output_cond - model_output_uncond)

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
        
        # 修正：WAN的all_latents维度是 (num_steps+1, 16, 7, 64, 64)
        all_latents = torch.stack(all_latents, dim=0)  # (9, 16, 7, 64, 64)
        all_log_probs = torch.stack(all_log_probs, dim=0)  # (8, B) -> (8,)
        
        # print(f"WAN after stack: all_latents={all_latents.shape}, all_log_probs={all_log_probs.shape}")
        # (9, 16, 7, 64, 64), (8,)
        
        return latents, final_latents, all_latents, all_log_probs

def sample_wan_reference_model(
    args,
    device, 
    transformer,
    vae,
    contexts,
    captions,
    text_encoder,
):
    """WAN参考模型采样，支持FP16优化"""
    # 视频参数
    frame_num = args.t
    size = (args.w, args.h)
    
    # 创建sigma调度（类似FLUX）
    sample_steps = args.sampling_steps
    sigma_schedule = torch.linspace(1, 0, sample_steps + 1).to(device)
    
    # 应用时间偏移（如果需要）
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
    
    # 根据是否使用FP16选择数据类型
    latent_dtype = torch.float16 if args.use_fp16 else torch.float32
    
    # 创建负向条件（CFG需要）
    neg_prompt = getattr(args, 'neg_prompt', "worst quality, low quality, bad anatomy, bad hands, bad body, missing fingers, extra digit, fewer digits")
    # 为每个样本创建负向prompt列表
    neg_prompts = [neg_prompt] * B

    # 使用T5编码器编码负向prompt
    if args.use_fp16:
        # 如果使用FP16，在GPU上编码然后移回CPU
        text_encoder.model.to(device)
        context_null = text_encoder(neg_prompts, device)
        text_encoder.model.cpu()  # 编码完成后移回CPU节省显存
    else:
        # 在CPU上编码
        context_null = text_encoder(neg_prompts, torch.device('cpu'))
        # 将结果移到GPU
        context_null = [t.to(device) for t in context_null]

    if args.init_same_noise:
        latent_shape = (
            16,
            (frame_num - 1) // vae_stride[0] + 1,
            size[1] // vae_stride[1],
            size[0] // vae_stride[2]
        )
        input_latents = torch.randn(latent_shape, device=device, dtype=latent_dtype)

    for index, batch_idx in enumerate(batch_indices):
        batch_captions = [captions[i] for i in batch_idx]
        batch_contexts = [contexts[i].to(device) for i in batch_idx]
        batch_context_null = [context_null[i].to(device) for i in batch_idx]
        
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
        
        # 获取CFG引导强度
        guide_scale = getattr(args, 'guide_scale', 5.0)

        # 使用梯度检查点减少内存使用
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
            # 使用适当的数据类型进行autocast
            autocast_dtype = torch.float16 if args.use_fp16 else torch.float32
            with torch.autocast("cuda", dtype=autocast_dtype):
                # 确保final_latents的数据类型正确
                final_latents_vae = final_latents.to(dtype=autocast_dtype)
                decoded_videos = vae.decode([final_latents_vae])
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

        color_reward = compute_red_intensity_reward(video_frames.unsqueeze(0))
        all_rewards.append(color_reward)

        del final_latents, video_frames, color_reward
        torch.cuda.empty_cache()

    # 拼接结果
    if len(all_latents) > 1:
        all_latents = torch.cat(all_latents, dim=0)
        all_log_probs = torch.cat(all_log_probs, dim=0)
        all_rewards = torch.cat(all_rewards, dim=0)
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
    
    # 使用适当的数据类型进行autocast
    autocast_dtype = torch.float16 if args.use_fp16 else torch.float32
    with torch.autocast("cuda", dtype=autocast_dtype):
        # 正向预测
        pred_cond = transformer(
            x=[latents],
            t=timesteps,
            context=context,
            seq_len=seq_len
        )
        
        # 负向预测
        pred_uncond = transformer(
            x=[latents],
            t=timesteps,
            context=context_null,
            seq_len=seq_len
        )
        
        # 处理输出
        if isinstance(pred_cond, dict) and 'rgb' in pred_cond:
            model_output_cond = pred_cond['rgb'][0]
            model_output_uncond = pred_uncond['rgb'][0]
        elif isinstance(pred_cond, list):
            model_output_cond = pred_cond[0]
            model_output_uncond = pred_uncond[0]
        else:
            model_output_cond = pred_cond
            model_output_uncond = pred_uncond
        
        # CFG组合
        model_output = model_output_uncond + guide_scale * (model_output_cond - model_output_uncond)

    # 确保数据类型一致性
    computation_dtype = torch.float32  # 关键计算使用FP32以保持精度 这里也暂时改成fp16吧
    # computation_dtype = torch.float16
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
    print('log_prob', log_prob)
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
    device,
    transformer,
    vae,
    text_encoder,
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
        args, device, transformer, vae, contexts, captions,text_encoder
    )
    
    # print(f"WAN training data shapes:")
    # print(f"  all_latents: {all_latents.shape}")     # (B, 9, 16, 7, 64, 64)
    # print(f"  all_log_probs: {all_log_probs.shape}") # (B, 8)
    # print(f"  reward: {reward.shape}")               # (B,)
    
    batch_size = all_latents.shape[0]

    # 【修正】使用T5编码器创建负向条件
    neg_prompt = getattr(args, 'neg_prompt', "worst quality, low quality, bad anatomy, bad hands, bad body, missing fingers, extra digit, fewer digits")
    neg_prompts = [neg_prompt] * batch_size
    
    if args.use_fp16:
        text_encoder.model.to(device)
        context_null = text_encoder(neg_prompts, device)
        text_encoder.model.cpu()
    else:
        context_null = text_encoder(neg_prompts, torch.device('cpu'))
        context_null = [t.to(device) for t in context_null]
    
    # 创建timestep values（基于sigma_schedule）
    timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][:args.sampling_steps]
    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    timesteps_tensor = torch.tensor(timestep_values, device=all_latents.device, dtype=torch.long)
    
    # 构建samples字典
    samples = {
        "timesteps": timesteps_tensor[:, :-1],           # (B, 8)
        "latents": all_latents[:, :-1],                  # (B, 8, 16, 7, 64, 64) - 当前状态
        "next_latents": all_latents[:, 1:],              # (B, 8, 16, 7, 64, 64) - 下一状态
        "log_probs": all_log_probs,                      # (B, 8)
        "rewards": reward.to(torch.float32),             # (B,)
        "sigma_schedule": sigma_schedule,                # 添加sigma_schedule
    }
    
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
            context_null_batch = [context_null[i].to(device)]  # 对应的负向条件

            latent_shape = sample["latents"][:, step_idx].shape
            seq_len = math.ceil(
                (latent_shape[3] * latent_shape[4]) / (2 * 2) * latent_shape[2]
            )
            
            new_log_probs = grpo_wan_one_step(
                args,
                sample["latents"][:, step_idx],
                sample["next_latents"][:, step_idx],
                context,  # List[Tensor]格式
                context_null_batch,  # List[Tensor]格式
                seq_len,
                transformer,
                sample["timesteps"][:, step_idx],
                perms[i][step_idx],
                sigma_schedule, 
                getattr(args, 'guide_scale', 5.0),
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
    
    return total_loss, grad_norm.item() if grad_norm is not None else 0.0, samples["rewards"]

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
    
    # 根据args.use_fp16决定数据类型
    model_dtype = torch.float16 if args.use_fp16 else torch.float32
    transformer = transformer.to(dtype=model_dtype)
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
    print(f"  Model dtype: {model_dtype}")

    # 使用FSDP包装模型 - 优化配置
    from torch.distributed.fsdp.api import CPUOffload, MixedPrecision, BackwardPrefetch, ShardingStrategy
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    import torch.distributed.fsdp as fsdp
    
    # 配置CPU offload（如果启用）
    cpu_offload = None
    if args.cpu_offload:
        cpu_offload = CPUOffload(offload_params=True)
        print("CPU offload enabled for parameters")
    
    # 配置混合精度
    if args.use_fp16:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float16,      # 参数使用fp16
            reduce_dtype=torch.float16,     # 梯度聚合使用fp32
            buffer_dtype=torch.float16,     # buffer使用fp16
            cast_forward_inputs=True,       # 自动转换输入类型
        )
        print("Using FP16 mixed precision")
    else:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
        print("Using FP32 precision")
    
    # 配置分片策略
    sharding_strategy = ShardingStrategy.FULL_SHARD
    if args.use_zero2:
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        print("Using ZeRO-2 (SHARD_GRAD_OP)")
    else:
        print("Using ZeRO-3 (FULL_SHARD)")
    
    transformer = FSDP(
        transformer,
        auto_wrap_policy=None,
        mixed_precision=mixed_precision_policy,
        device_id=local_rank,
        cpu_offload=cpu_offload,
        sharding_strategy=sharding_strategy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # 预取优化
        limit_all_gathers=True,  # 限制all_gather操作
        use_orig_params=False,   # 使用展平参数以节省内存
    )
    
    # 加载VAE - 如果启用fp16也使用fp16
    vae_dtype = torch.float16 if args.use_fp16 else torch.float32
    vae = WanVAE(
        vae_pth=os.path.join(args.pretrained_model_name_or_path, config.vae_checkpoint),
        device=device
    )
    vae.model = vae.model.to(dtype=vae_dtype)
    print(f"VAE dtype: {vae_dtype}")

    print(f"--> WAN model loaded with memory optimizations")
    print("Loading T5 text encoder for CFG...")
    text_encoder = T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=torch.device('cpu'),  # 放在CPU上节省显存
        checkpoint_path=os.path.join(args.pretrained_model_name_or_path, config.t5_checkpoint),
        tokenizer_path=os.path.join(args.pretrained_model_name_or_path, config.t5_tokenizer),
    )
    print("T5 text encoder loaded")
    # 设置模型为训练模式
    transformer.train()

    # 优化器 - 支持CPU offload
    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))
    use_zero_optimizer = args.optimizer_cpu_offload

    if use_zero_optimizer:
        # 使用CPU offload的优化器
        from torch.distributed.optim import ZeroRedundancyOptimizer
        print("Using ZeroRedundancyOptimizer with CPU offload")
        
        optimizer = ZeroRedundancyOptimizer(
            params_to_optimize,
            optimizer_class=torch.optim.AdamW,
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
            eps=1e-8,
        )
    else:
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

    for epoch in range(1000000):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        for step in range(1, args.max_train_steps + 1):
            # if step % args.checkpointing_steps == 0:
            #     save_checkpoint_safely(step, transformer, optimizer, lr_scheduler, args.output_dir, rank, use_zero_optimizer)

            # 执行训练步骤
            loss, grad_norm, batch_rewards = train_wan_one_step(
                args,
                device, 
                transformer,
                vae,
                text_encoder,
                optimizer,
                lr_scheduler,
                loader,
                args.max_grad_norm,
            )
            # 实时更新图表（只在rank 0上）
            if rank == 0 and reward_tracker is not None:
                try:
                    # 使用新的add_data方法，会立即保存图表
                    reward_tracker.add_data(
                        step=step,
                        loss=loss,
                        reward_tensor=batch_rewards,
                        grad_norm=grad_norm
                    )
                    # 显示保存信息
                    print(f"Step {step}: Plots saved to {reward_tracker.save_dir}")
                except Exception as e:
                    print(f"Error updating real-time plots: {e}")

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

    # 训练结束时保存最终检查点
    # if rank == 0:
    #     save_final_checkpoint_safely(
    #         step, transformer, optimizer, lr_scheduler, 
    #         args.output_dir, use_zero_optimizer
    #     )

    # 训练结束时保存最终总结图表
    if rank == 0 and reward_tracker is not None:
        try:
            reward_tracker.save_summary_plot("final")
            print(f"Final training plots saved to {reward_tracker.save_dir}")
        except Exception as e:
            print(f"Error saving final plots: {e}")
    
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

    # 内存优化参数
    parser.add_argument("--use_fp16", action="store_true", default=False, 
                       help="Use FP16 mixed precision training")
    parser.add_argument("--cpu_offload", action="store_true", default=False,
                       help="Offload model parameters to CPU")
    parser.add_argument("--optimizer_cpu_offload", action="store_true", default=False,
                       help="Offload optimizer states to CPU")
    parser.add_argument("--use_zero2", action="store_true", default=False,
                       help="Use ZeRO-2 instead of ZeRO-3")
    
    # 视频保存相关参数
    parser.add_argument("--video_fps", type=int, default=25, help="保存视频的帧率")
    parser.add_argument("--save_video_every_steps", type=int, default=100, help="每隔多少步保存一次视频")
    parser.add_argument("--max_videos_per_checkpoint", type=int, default=4, help="每个检查点最多保存多少个视频")

    parser.add_argument("--save_plot_every_steps", type=int, default=50, help="每隔多少步保存一次reward图表")
    
    parser.add_argument("--guide_scale", type=float, default=5.0, 
                   help="Classifier-free guidance scale")
    parser.add_argument("--neg_prompt", type=str, 
                   default="worst quality, low quality, bad anatomy, bad hands, bad body, missing fingers, extra digit, fewer digits",
                   help="Negative prompt for CFG")
    args = parser.parse_args()
    main(args)
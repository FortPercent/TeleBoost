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
from fastvideo.utils.checkpoint import save_checkpoint

# 加入hps
from HPSv2.hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    # 转换数据类型为float32，避免Bfloat16转numpy的问题
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
    """WAN采样步骤，修正CFG实现避免通道数错误"""
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
            
            # 【修正】序列化CFG，避免在latents维度合并
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
        all_latents = torch.stack(all_latents, dim=0)  # (9, 16, 7, 64, 64)
        all_log_probs = torch.stack(all_log_probs, dim=0)  # (8, B) -> (8,)
        
        return latents, final_latents, all_latents, all_log_probs

def sample_wan_reference_model(
    args,
    device, 
    transformer,
    vae,
    contexts,
    captions,
    context_null_single,
    # 新增hpsv2参数
    reward_model=None,
    tokenizer=None,
    preprocess_val=None,
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
    latent_dtype = torch.bfloat16 if args.use_bf16 else torch.float32

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
        # 【优化】直接复制预计算的context_null，以匹配当前批次大小
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
            autocast_dtype = torch.bfloat16 if args.use_bf16 else torch.float32
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

        # 计算奖励
        if args.use_hpsv2 and reward_model is not None and tokenizer is not None and preprocess_val is not None:
            # 对每一帧都进行HPS v2评分
            with torch.no_grad():
                frame_rewards = []
                C, T, H, W = video_frames.shape  # (C, T, H, W)
                
                for t in range(T):  # 遍历每一帧
                    frame = video_frames[:, t, :, :]  # (C, H, W)
                    
                    # 转换为PIL图像
                    frame_np = frame.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
                    frame_np = (frame_np * 255).astype(np.uint8)
                    frame_pil = Image.fromarray(frame_np)
                    
                    # 预处理图像
                    image = preprocess_val(frame_pil).unsqueeze(0).to(device=device, non_blocking=True)
                    # 处理文本prompt
                    text = tokenizer([batch_captions[0]]).to(device=device, non_blocking=True)
                    
                    # 计算HPS分数
                    outputs = reward_model(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T
                    hps_score = torch.diagonal(logits_per_image)
                    frame_rewards.append(hps_score)
                
                # 将所有帧的奖励汇总
                frame_rewards = torch.stack(frame_rewards, dim=0)  # (T, 1)
                
                # 计算视频整体奖励的策略选择
                if args.hps_aggregation == "mean":
                    video_reward = frame_rewards.mean(dim=0)  # 平均值
                elif args.hps_aggregation == "max":
                    video_reward = frame_rewards.max(dim=0)[0]  # 最大值
                elif args.hps_aggregation == "min":
                    video_reward = frame_rewards.min(dim=0)[0]  # 最小值
                elif args.hps_aggregation == "weighted":
                    # 时间加权：后面的帧权重更高
                    weights = torch.linspace(0.5, 1.0, T, device='cpu').unsqueeze(1)  # (T, 1)
                    video_reward = (frame_rewards * weights).sum(dim=0) / weights.sum()
                else:  # 默认使用平均值
                    video_reward = frame_rewards.mean(dim=0)
                
                # 确保video_reward在正确的设备上且为正确的数据类型
                video_reward = video_reward.to(device, dtype=torch.float32)
                all_rewards.append(video_reward)
                
                # 可选：保存每帧的HPS分数用于分析
                if rank == 0:
                    frame_scores_str = ", ".join([f"Frame{t}: {score.item():.3f}" for t, score in enumerate(frame_rewards)])
                    print(f"HPS scores per frame - {frame_scores_str}")
                    print(f"Video overall HPS ({args.hps_aggregation}): {video_reward.item():.3f}")
                    
                    # 保存详细的帧级分数到文件
                    with open('./wan_hps_per_frame.txt', 'a') as f:
                        f.write(f"Video {index}: {frame_scores_str}, Overall: {video_reward.item():.3f}\n")
        else:
            # 使用红色强度奖励
            color_reward = compute_red_intensity_reward(video_frames.unsqueeze(0))
            all_rewards.append(color_reward)

        del final_latents, video_frames
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
        # 先计算条件预测
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
        args, device, transformer, vae, contexts, captions,context_null_single,
        reward_model, tokenizer, preprocess_val,
    )
    
    batch_size = all_latents.shape[0]

    context_null = [context_null_single[0] for _ in range(batch_size)]
    
    # 【关键修正】参考Hunyuan的timesteps构建方式
    timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][:args.sampling_steps]
    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    timesteps_tensor = torch.tensor(timestep_values, device=all_latents.device, dtype=torch.long)
    
    # 【关键修正】参考Hunyuan构建samples，注意维度处理
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
    
    gathered_reward = gather_tensor(samples["rewards"])
    if dist.get_rank() == 0:
        if args.use_hpsv2:
            print("gathered_hps_reward", gathered_reward)
            with open('./wan_hps_reward.txt', 'a') as f: 
                f.write(f"{gathered_reward.mean().item()}\n")
        else:
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

    # 【参考Hunyuan】Best-of-N选择（如果需要）
    if hasattr(args, 'bestofn') and args.bestofn < batch_size:
        total_scores = samples["advantages"]
        sorted_indices = torch.argsort(total_scores)
        top_indices = sorted_indices[-args.bestofn//2:]     
        bottom_indices = sorted_indices[:args.bestofn//2]     
        selected_indices = torch.cat([top_indices, bottom_indices])
        shuffled_order = torch.randperm(len(selected_indices), device=selected_indices.device)
        selected_indices = selected_indices[shuffled_order]
        
        # 选择样本
        for key in ["timesteps", "latents", "next_latents", "log_probs", "rewards", "advantages"]:
            samples[key] = samples[key][selected_indices]
        
        # 更新contexts
        new_contexts = [contexts[i] for i in selected_indices.cpu().numpy()]
        new_context_null = [context_null[i] for i in selected_indices.cpu().numpy()]
        samples["contexts"] = new_contexts
        samples["context_null"] = new_context_null
        batch_size = len(selected_indices)

    # 【参考Hunyuan】随机排列
    perms = torch.stack([
        torch.randperm(len(samples["timesteps"][0])) 
        for _ in range(batch_size)
    ]).to(device)
    
    for key in ["timesteps", "latents", "next_latents", "log_probs"]:
        samples[key] = samples[key][
            torch.arange(batch_size).to(device)[:, None],
            perms,
        ]
    
    # 【关键修正】参考Hunyuan的正确方式构建batched样本
    samples_batched = {
        k: v.unsqueeze(1) if k in ["timesteps", "latents", "next_latents", "log_probs"] else v
        for k, v in samples.items()
    }
    
    # 【参考Hunyuan】构建样本列表的正确方式
    samples_batched_list = []
    for i in range(batch_size):
        sample_dict = {
            "timesteps": samples_batched["timesteps"][i],      # (1, steps)
            "latents": samples_batched["latents"][i],          # (1, steps, C, T, H, W)
            "next_latents": samples_batched["next_latents"][i], # (1, steps, C, T, H, W)
            "log_probs": samples_batched["log_probs"][i],      # (1, steps)
            "rewards": samples["rewards"][i],                   # scalar
            "advantages": samples["advantages"][i],             # scalar
            "contexts": [samples["contexts"][i]],               # List[Tensor]
            "context_null": [samples["context_null"][i]],       # List[Tensor]
            "sigma_schedule": sigma_schedule,
        }
        samples_batched_list.append(sample_dict)
    
    train_timesteps = int(len(samples["timesteps"][0]) * args.timestep_fraction)
    grad_norm = None
    print(f"DEBUG: Processing {len(samples_batched_list)} samples")
    
    for i, sample in enumerate(samples_batched_list):
        for step_idx in range(train_timesteps):
            clip_range = args.clip_range
            adv_clip_max = args.adv_clip_max
            
            # 【修正】正确处理维度和索引
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
                current_latents,           # (C, T, H, W)
                next_latents,             # (C, T, H, W)
                sample["contexts"],       # List[Tensor]
                sample["context_null"],   # List[Tensor]
                seq_len,
                transformer,
                current_timesteps.unsqueeze(0),  # (1,)
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

            ratio = torch.exp(new_log_probs - current_log_probs)

            unclipped_loss = -advantages * ratio
            clipped_loss = -advantages * torch.clamp(
                ratio,
                1.0 - clip_range,
                1.0 + clip_range*2, #据DAPO说这里大点好
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
            if args.use_hpsv2:
                print("hps reward", sample["rewards"].item())
            else:
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
        # 2. 包装所有WAN的主要组件
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

    # 添加 HPSv2 初始化
    if args.use_hpsv2:
        if create_model_and_transforms is None or get_tokenizer is None:
            raise ImportError("HPSv2 modules not found. Please install HPSv2.")
            
        def initialize_model():
            model_dict = {}
            model, preprocess_train, preprocess_val = create_model_and_transforms(
                'ViT-H-14',
                './hps_ckpt/open_clip_pytorch_model.bin',
                precision='amp',
                device=device,
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
        model = model_dict['model']
        preprocess_val = model_dict['preprocess_val']
        cp = "./hps_ckpt/HPS_v2.1_compressed.pt"

        checkpoint = torch.load(cp, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        tokenizer = get_tokenizer('ViT-H-14')
        reward_model = model.to(device)
        reward_model.eval()

        print(f"HPS aggregation method: {args.hps_aggregation}")

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
            # save_checkpoint(transformer, rank, args.output_dir, step, epoch)
            # 执行训练步骤
            loss, grad_norm, batch_rewards = train_wan_one_step(
                args,
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
            )
            # 实时更新图表（只在rank 0上）
            if rank == 0 and reward_tracker is not None:
                reward_tracker.add_data(
                    step=step,
                    loss=loss,
                    reward_tensor=batch_rewards,
                    grad_norm=grad_norm
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
    args = parser.parse_args()
    main(args)
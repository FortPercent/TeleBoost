# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import torch
import numpy as np
import re
from torch.nn import functional as F



def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    if method == "strict":
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score


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

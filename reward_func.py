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
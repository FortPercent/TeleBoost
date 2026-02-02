import torch
import numpy as np
import cv2
import os
import sys
import pickle
from scipy.fft import fft, fftfreq
from scipy.stats import entropy
import torch.nn.functional as F
from pathlib import Path

# 添加RAFT路径
sys.path.append('/gemini/platform/public/zqni/RAFT-master/core')
from raft import RAFT
from utils.utils import InputPadder

class SpectralVolumeRewardCalculator:
    """
    基于光流频谱的奖励计算器
    结合平均频谱匹配和分布一致性的双重奖励机制
    """
    def __init__(self, 
                 reference_stats_path,
                 raft_model_path,
                 target_size=(320, 180),
                 device='cuda',
                 w1=0.6,  # 平均频谱匹配权重
                 w2=0.4,  # 分布一致性权重
                 norm_type='l2',  # 'l1' or 'l2'
                 freq_range=(0.0, 15.0),
                 use_frequency_weighting=True):
        """
        初始化频域奖励计算器
        Args:
            reference_stats_path: 预计算的真实视频统计数据路径
            raft_model_path: RAFT模型权重路径
            target_size: 光流计算的目标尺寸 (width, height)
            device: 计算设备
            w1: 平均频谱匹配奖励权重
            w2: 分布一致性奖励权重
            norm_type: 距离度量类型
            freq_range: 关注的频率范围
            use_frequency_weighting: 是否使用频率加权
        """
        self.device = device
        self.target_size = target_size
        self.w1 = w1
        self.w2 = w2
        self.norm_type = norm_type
        self.freq_range = freq_range
        self.use_frequency_weighting = use_frequency_weighting
        # 加载预计算的参考统计数据
        self._load_reference_stats(reference_stats_path)
        # 初始化RAFT模型
        self._init_raft_model(raft_model_path)
        print(f"✅ SpectralVolumeRewardCalculator initialized")
        print(f"   - Weights: w1={w1:.2f}, w2={w2:.2f}")
        print(f"   - Frequency range: {freq_range[0]:.1f}-{freq_range[1]:.1f} Hz")
        print(f"   - Reference frequencies: {len(self.reference_freqs)} points")

    def _load_reference_stats(self, stats_path):
        """加载预计算的参考统计数据"""
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Reference stats file not found: {stats_path}")
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        # 提取关键统计量
        self.reference_freqs = stats['frequencies']
        self.mean_real_u_spectrum = stats['mean_u_amplitude_l2_norm']
        self.mean_real_v_spectrum = stats['mean_v_amplitude_l2_norm']
        
        # 如果有预计算的差值分布统计，使用之；否则使用默认值
        if 'difference_stats' in stats:
            self.mu_d_real = stats['difference_stats']['mean']
            self.sigma_d_real = stats['difference_stats']['std']
        else:
            # 根据您的分析图，真实视频的KL散度分布大约是这个范围
            self.mu_d_real = 0.1256  # 从您的分析中得出
            self.sigma_d_real = 0.05  # 估计值，可以根据实际数据调整
        
        # 创建频率掩码，只关注指定范围
        self.freq_mask = ((self.reference_freqs >= self.freq_range[0]) & 
                         (self.reference_freqs <= self.freq_range[1]))
        self.analysis_freqs = self.reference_freqs[self.freq_mask]
        self.ref_u_analysis = self.mean_real_u_spectrum[self.freq_mask]
        self.ref_v_analysis = self.mean_real_v_spectrum[self.freq_mask]
        
        # 创建频率权重
        if self.use_frequency_weighting:
            self.freq_weights = self._create_frequency_weights(self.analysis_freqs)
        else:
            self.freq_weights = np.ones_like(self.analysis_freqs)
        
        print(f"✅ Reference stats loaded from {stats_path}")
        print(f"   - Real video difference distribution: μ={self.mu_d_real:.4f}, σ={self.sigma_d_real:.4f}")
    
    def _create_frequency_weights(self, freqs):
        """
        创建频率权重，强调中频动态信息，降低直流和高频噪声的影响
        """
        weights = np.ones_like(freqs)
        # 降低直流分量权重
        dc_mask = freqs < 0.5
        weights[dc_mask] *= 0.3
        # 增强中频动态信息权重 (1-8 Hz)
        mid_freq_mask = (freqs >= 1.0) & (freqs <= 8.0)
        weights[mid_freq_mask] *= 1.5
        # 适度降低高频权重 (> 10 Hz)
        high_freq_mask = freqs > 10.0
        weights[high_freq_mask] *= 0.7
        # 归一化权重
        weights = weights / np.sum(weights) * len(weights)
        return weights
    
    def _init_raft_model(self, model_path):
        """初始化RAFT光流模型"""
        import argparse
        # 创建RAFT模型参数
        raft_args = argparse.Namespace()
        raft_args.small = False
        raft_args.mixed_precision = False
        raft_args.alternate_corr = False
        # 加载模型
        self.raft_model = RAFT(raft_args)
        self.raft_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.raft_model = self.raft_model.to(self.device)
        self.raft_model.eval()
        print(f"✅ RAFT model loaded from {model_path}")
    
    def _video_to_frames(self, video_tensor, num_frames=64):
        """
        将视频张量转换为帧列表
        Args:
            video_tensor: shape (C, T, H, W), 范围 [0, 1]
            num_frames: 提取的帧数
            
        Returns:
            List of numpy arrays, each with shape (H, W, 3)
        """
        C, T, H, W = video_tensor.shape
        # 确保有足够的帧
        if T < num_frames:
            # 如果帧数不够，重复最后一帧
            padding_frames = num_frames - T
            last_frame = video_tensor[:, -1:, :, :].repeat(1, padding_frames, 1, 1)
            video_tensor = torch.cat([video_tensor, last_frame], dim=1)
        elif T > num_frames:
            # 如果帧数太多，均匀采样
            indices = torch.linspace(0, T-1, num_frames).long()
            video_tensor = video_tensor[:, indices, :, :]
        # 转换为numpy并调整尺寸
        video_np = video_tensor.permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, C)
        video_np = (video_np * 255).astype(np.uint8)
        # 确保是3通道
        if C == 1:
            video_np = np.repeat(video_np, 3, axis=-1)
        elif C > 3:
            video_np = video_np[:, :, :, :3]
        # 调整尺寸并转换为列表
        frames = []
        for t in range(num_frames):
            frame = video_np[t]
            if self.target_size is not None:
                frame = cv2.resize(frame, self.target_size)
            frames.append(frame)
        return frames
    
    def _compute_optical_flow_sequence(self, frames):
        """
        使用RAFT计算光流序列
        Args:
            frames: List of numpy arrays, each with shape (H, W, 3)
        Returns:
            numpy array with shape (T-1, H, W, 2)
        """
        flows = []
        with torch.no_grad():
            for i in range(len(frames) - 1):
                # 预处理帧
                frame1 = torch.from_numpy(frames[i]).permute(2, 0, 1).float()
                frame2 = torch.from_numpy(frames[i + 1]).permute(2, 0, 1).float()
                frame1 = frame1[None].to(self.device)  # (1, 3, H, W)
                frame2 = frame2[None].to(self.device)  # (1, 3, H, W)
                # 应用padding
                padder = InputPadder(frame1.shape)
                frame1, frame2 = padder.pad(frame1, frame2)
                # 计算光流
                flow_low, flow_up = self.raft_model(frame1, frame2, iters=20, test_mode=True)
                # 获取光流 (H, W, 2)
                flow = flow_up[0].permute(1, 2, 0).cpu().numpy()
                flows.append(flow)
        return np.array(flows)  # (T-1, H, W, 2)
    
    def _compute_spectral_volume(self, flows, fps=25):
        """
        从光流计算频谱体积
        Args:
            flows: numpy array with shape (T, H, W, 2)
            fps: 帧率
        Returns:
            dict with spectral information
        """
        T, H, W, _ = flows.shape
        u_flows = flows[..., 0]  # (T, H, W)
        v_flows = flows[..., 1]  # (T, H, W)
        # 对时间维度进行FFT
        u_fft = fft(u_flows, axis=0)
        v_fft = fft(v_flows, axis=0)
        # 计算幅度谱
        u_magnitude = np.abs(u_fft)
        v_magnitude = np.abs(v_fft)
        # 对空间维度求平均，得到频率-幅度谱
        u_amplitude = np.mean(u_magnitude, axis=(1, 2))
        v_amplitude = np.mean(v_magnitude, axis=(1, 2))
        # 获取频率轴
        freqs = fftfreq(T, 1/fps)
        positive_freq_idx = freqs >= 0
        freqs = freqs[positive_freq_idx]
        u_amplitude = u_amplitude[positive_freq_idx]
        v_amplitude = v_amplitude[positive_freq_idx]
        return {
            'frequencies': freqs,
            'u_amplitude': u_amplitude,
            'v_amplitude': v_amplitude,
            'fps': fps
        }
    
    def _normalize_spectrum(self, spectrum, method='l2'):
        """归一化频谱"""
        spectrum = spectrum.copy()
        if method == 'l2':
            norm = np.linalg.norm(spectrum)
            return spectrum / (norm + 1e-8)
        elif method == 'l1':
            norm = np.sum(np.abs(spectrum))
            return spectrum / (norm + 1e-8)
        else:
            return spectrum
    
    def _compute_distance(self, spec1, spec2, weights=None):
        """
        计算两个频谱之间的加权距离
        Args:
            spec1, spec2: 频谱数组
            weights: 频率权重，如果为None则使用均匀权重
        Returns:
            距离值
        """
        if weights is None:
            weights = np.ones_like(spec1)
        if self.norm_type == 'l1':
            distance = np.sum(weights * np.abs(spec1 - spec2))
        elif self.norm_type == 'l2':
            distance = np.sqrt(np.sum(weights * (spec1 - spec2) ** 2))
        else:
            raise ValueError(f"Unknown norm type: {self.norm_type}")
        return distance
    
    def _compute_kl_divergence(self, p, q):
        """计算KL散度"""
        p = p + 1e-8
        q = q + 1e-8
        p = p / p.sum()
        q = q / q.sum()
        return entropy(p, q)
    
    def compute_reward(self, video_tensor):
        """
        计算单个视频的频域奖励
        Args:
            video_tensor: torch.Tensor, shape (C, T, H, W), 范围 [0, 1]
        Returns:
            float: 总奖励分数
        """
        try:
            # 1. 提取视频帧
            frames = self._video_to_frames(video_tensor)
            if len(frames) < 2:
                return 0.0
            # 2. 计算光流序列
            flows = self._compute_optical_flow_sequence(frames)
            if flows.shape[0] < 2:
                return 0.0
            # 3. 计算频谱体积
            spectral_data = self._compute_spectral_volume(flows)
            # 4. 处理频谱
            u_amplitude = spectral_data['u_amplitude']
            v_amplitude = spectral_data['v_amplitude']
            gen_freqs = spectral_data['frequencies']
            # 5. 插值到参考频率网格
            u_interp = np.interp(self.reference_freqs, gen_freqs, u_amplitude)
            v_interp = np.interp(self.reference_freqs, gen_freqs, v_amplitude)
            # 6. 归一化
            u_norm = self._normalize_spectrum(u_interp, 'l2')
            v_norm = self._normalize_spectrum(v_interp, 'l2')
            # 7. 裁剪到分析频率范围
            u_analysis = u_norm[self.freq_mask]
            v_analysis = v_norm[self.freq_mask]
            # 8. 计算平均频谱匹配奖励 (R_mean)
            u_distance = self._compute_distance(u_analysis, self.ref_u_analysis, self.freq_weights)
            v_distance = self._compute_distance(v_analysis, self.ref_v_analysis, self.freq_weights)
            # 组合U和V方向的距离
            total_distance = 0.5 * u_distance + 0.5 * v_distance
            # 转换为奖励（距离越小，奖励越高）
            R_mean = np.exp(-total_distance)
            # 9. 计算分布一致性奖励 (R_dist)
            # 计算当前视频与参考谱的差值分数（使用KL散度）
            u_kl = self._compute_kl_divergence(u_analysis, self.ref_u_analysis)
            v_kl = self._compute_kl_divergence(v_analysis, self.ref_v_analysis)
            total_kl = 0.5 * u_kl + 0.5 * v_kl
            # 高斯形式的分布一致性奖励
            R_dist = np.exp(-0.5 * ((total_kl - self.mu_d_real) / self.sigma_d_real) ** 2)
            # 10. 计算总奖励
            R_total = self.w1 * R_mean + self.w2 * R_dist
            # 缩放到合理范围
            R_total = R_total * 100.0  # 0-100分
            return float(R_total)
            
        except Exception as e:
            print(f"Error computing spectral reward: {e}")
            return 0.0
    
    def compute_batch_rewards(self, video_batch):
        """
        计算一批视频的频域奖励
        Args:
            video_batch: torch.Tensor, shape (B, C, T, H, W), 范围 [0, 1]
        Returns:
            torch.Tensor, shape (B,), 奖励分数
        """
        if video_batch.dim() == 4:  # (C, T, H, W)
            video_batch = video_batch.unsqueeze(0)  # -> (1, C, T, H, W)
        batch_size = video_batch.shape[0]
        rewards = []
        for b in range(batch_size):
            video = video_batch[b]  # (C, T, H, W)
            reward = self.compute_reward(video)
            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32, device=video_batch.device)

# 为训练代码集成准备的函数
def compute_spectral_volume_reward(videos, reference_stats_path, raft_model_path, args):
    """
    计算基于光流频谱的奖励函数
    这是集成到训练代码中的主要接口
    Args:
        videos: torch.Tensor, shape (B, C, T, H, W), 范围 [0, 1]
        reference_stats_path: 预计算的真实视频统计数据路径
        raft_model_path: RAFT模型路径
        args: 训练参数
    Returns:
        torch.Tensor, shape (B,), 奖励分数
    """
    # 创建奖励计算器（可以考虑做成全局变量以避免重复初始化）
    if not hasattr(compute_spectral_volume_reward, 'calculator'):
        print("Initializing SpectralVolumeRewardCalculator...")
        
        calculator = SpectralVolumeRewardCalculator(
            reference_stats_path=reference_stats_path,
            raft_model_path=raft_model_path,
            target_size=(320, 180),
            device=videos.device,
            w1=getattr(args, 'spectral_w1', 0.6),
            w2=getattr(args, 'spectral_w2', 0.4),
            norm_type=getattr(args, 'spectral_norm', 'l2'),
            freq_range=getattr(args, 'spectral_freq_range', (0.0, 15.0)),
            use_frequency_weighting=getattr(args, 'spectral_use_freq_weights', True)
        )
        compute_spectral_volume_reward.calculator = calculator
    # 计算奖励
    return compute_spectral_volume_reward.calculator.compute_batch_rewards(videos)

# 预处理真实视频数据，计算参考统计量的函数
def precompute_reference_statistics(real_video_spectral_dir, output_path, max_videos=1000):
    """
    预计算真实视频的参考统计量
    Args:
        real_video_spectral_dir: 真实视频频谱数据目录
        output_path: 输出统计数据的路径
        max_videos: 最大处理视频数量
    """
    import glob
    from tqdm import tqdm
    print("Precomputing reference statistics from real videos...")
    spectral_files = glob.glob(os.path.join(real_video_spectral_dir, '*_spectral.pkl'))
    if len(spectral_files) > max_videos:
        spectral_files = spectral_files[:max_videos]
    print(f"Processing {len(spectral_files)} spectral files...")
    # 收集所有频谱数据
    all_u_power = []
    all_v_power = []
    all_frequencies = []
    for spectral_file in tqdm(spectral_files, desc="Loading spectral data"):
        try:
            with open(spectral_file, 'rb') as f:
                data = pickle.load(f)
            u_power_mean = np.mean(data['u_power'], axis=(1, 2))
            v_power_mean = np.mean(data['v_power'], axis=(1, 2))
            all_u_power.append(u_power_mean)
            all_v_power.append(v_power_mean)
            all_frequencies.append(data['frequencies'])
        except Exception as e:
            print(f"Warning: Could not load {spectral_file}: {e}")
    
    # 创建统一频率网格
    common_fps = 25
    max_len = max(len(f) for f in all_frequencies)
    time_samples = 2 * (max_len - 1) if max_len > 1 else 1
    common_freqs = fftfreq(time_samples, 1/common_fps)
    positive_mask = common_freqs >= 0
    common_freqs = common_freqs[positive_mask]
    
    # 插值并计算均值
    interpolated_u = []
    interpolated_v = []
    for u_power, v_power, freqs in zip(all_u_power, all_v_power, all_frequencies):
        u_amplitude = np.sqrt(u_power)
        v_amplitude = np.sqrt(v_power)
        u_interp = np.interp(common_freqs, freqs, u_amplitude)
        v_interp = np.interp(common_freqs, freqs, v_amplitude)
        # L2归一化
        u_norm = u_interp / (np.linalg.norm(u_interp) + 1e-8)
        v_norm = v_interp / (np.linalg.norm(v_interp) + 1e-8)
        interpolated_u.append(u_norm)
        interpolated_v.append(v_norm)
    # 计算均值频谱
    mean_u_spectrum = np.mean(np.array(interpolated_u), axis=0)
    mean_v_spectrum = np.mean(np.array(interpolated_v), axis=0)
    # 计算差值分布统计
    print("Computing difference distribution statistics...")
    # 使用KL散度计算每个样本与均值的差异
    def compute_kl_safe(p, q):
        p = p + 1e-8
        q = q + 1e-8
        p = p / p.sum()
        q = q / q.sum()
        return entropy(p, q)
    difference_scores = []
    for u_norm, v_norm in zip(interpolated_u, interpolated_v):
        u_kl = compute_kl_safe(u_norm, mean_u_spectrum)
        v_kl = compute_kl_safe(v_norm, mean_v_spectrum)
        total_kl = 0.5 * u_kl + 0.5 * v_kl
        difference_scores.append(total_kl)
    difference_scores = np.array(difference_scores)
    # 创建统计数据字典
    stats_data = {
        'frequencies': common_freqs,
        'mean_u_amplitude_l2_norm': mean_u_spectrum,
        'mean_v_amplitude_l2_norm': mean_v_spectrum,
        'num_videos': len(interpolated_u),
        'fps': common_fps,
        'difference_stats': {
            'mean': np.mean(difference_scores),
            'std': np.std(difference_scores),
            'min': np.min(difference_scores),
            'max': np.max(difference_scores),
            'median': np.median(difference_scores),
        },
        'config': {
            'source_dir': real_video_spectral_dir,
            'max_videos': max_videos,
            'normalization': 'l2',
        }
    }
    # 保存统计数据
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(stats_data, f)
    print(f"✅ Reference statistics saved to: {output_path}")
    print(f"   - Processed videos: {len(interpolated_u)}")
    print(f"   - Frequency points: {len(common_freqs)}")
    print(f"   - Difference distribution: μ={stats_data['difference_stats']['mean']:.4f}, σ={stats_data['difference_stats']['std']:.4f}")
    return stats_data
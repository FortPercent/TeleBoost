# Copyright 2024 Dance-GRPO Team
"""
Qwen Reward Model - Uses Qwen VL model for video quality evaluation.

This model uses a large vision-language model (Qwen) to evaluate video quality
through a structured prompt that asks the model to rate various quality dimensions.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.debug import WorkerProfiler

from .base import BaseRewardModel, RewardConfig, make_reward_batch
from .registry import RewardRegistry

logger = logging.getLogger(__name__)


# Quality evaluation prompt (Chinese)
QUALITY_EVALUATION_PROMPT = """
你是一个专业的视频质量评估专家。请从以下维度对这段视频进行评分：

**评估维度：**
- dim1（视觉伪影）：检查画面是否存在闪烁、马赛克、条纹等伪影
- dim2（局部变形）：检查物体和人物是否有不自然的变形或拉伸
- dim3（噪声质量）：评估画面噪点和颗粒感
- dim4（清晰度）：评估画面整体清晰度和锐度
- dim5（色彩准确性）：评估色彩是否自然、准确

评分规则：
- 每个维度评分在 0 ~ 100 范围内，越好越高分。
- 合计为五项得分的算术平均，保留整数。
- 对于某项严重失真或效果极差（如严重模糊、强伪影等），请大胆给出低分（例如低于30分）。
- 每个视频的打分应充分拉开差距，避免视频之间出现"同分"或"几乎同分"情况。
- 请确保不同维度之间的评分不互相矛盾，确保评分具有可比性与区分度。

输出格式（严格遵守）：
dim1:XX分,dim2:XX分,dim3:XX分,dim4:XX分,dim5:XX分,合计:XX分

风格要求：
- 禁止输出解释性文字或分析过程。
- 禁止使用"我认为"、"可能"、"大致"等模糊词语。
- 输出必须严格按照上述格式，一次性返回评估结果。

请严格按照输出格式要求，输出且只输出输出格式的内容。请依照以上标准、逻辑和格式，对视频进行结构化质量评估。
"""


@RewardRegistry.register("qwen")
class QwenRewardModel(BaseRewardModel):
    """
    Qwen Vision-Language Model reward for video quality evaluation.
    
    Uses Qwen VL model to evaluate video quality through a structured prompt
    that rates multiple quality dimensions and produces an overall score.
    
    Config extra_config keys:
        - model_path: Path to Qwen VL model
        - max_tokens: Maximum output tokens (default: 256)
        - temperature: Sampling temperature (default: 0.1)
        - max_pixels: Maximum pixels for video processing (default: 360*420)
        - fps: Frames per second for video sampling (default: 1.0)
    """
    
    REWARD_KEY = "rewards"
    
    def __init__(self, config: RewardConfig, global_rank: int, world_size: int):
        super().__init__(config, global_rank, world_size)
        self.reward_rollout = None
        self.sampling_params = None
        self.max_pixels: int = 360 * 420
        self.fps: float = 1.0
    
    def init_model(self) -> None:
        """Initialize the Qwen model via vLLM."""
        if not self.is_active:
            logger.info(f"[qwen] Rank {self.global_rank} inactive, skipping init")
            return
        
        extra = self.config.extra_config or {}
        model_path = extra.get("model_path") or self.config.model_path
        max_tokens = extra.get("max_tokens", 256)
        temperature = extra.get("temperature", 0.1)
        self.max_pixels = extra.get("max_pixels", 360 * 420)
        self.fps = extra.get("fps", 1.0)
        
        if not model_path:
            raise ValueError("Qwen model requires 'model_path' in config")
        
        try:
            self._build_rollout(model_path, max_tokens, temperature)
            logger.info(f"Qwen reward model initialized from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen model: {e}")
            raise
    
    def _build_rollout(
        self, 
        model_path: str, 
        max_tokens: int, 
        temperature: float
    ) -> None:
        """Build the vLLM inference engine."""
        from vllm import LLM, SamplingParams
        
        self.reward_rollout = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
        )
        
        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        )
    
    def compute_single_score(
        self, 
        video_frames: torch.Tensor, 
        caption: str
    ) -> float:
        """
        Compute quality score for a single video.
        
        Note: This method is not used directly for Qwen model.
        Instead, compute_batch_score handles batched inference.
        """
        raise NotImplementedError(
            "Qwen model uses batch inference. Use compute_batch_score instead."
        )
    
    def compute_batch_score(self, data: DataProto) -> DataProto:
        """
        Compute quality scores for a batch of videos using Qwen.
        
        Overrides base class to use vLLM batch inference.
        """
        if not self.is_active:
            batch_size = data.batch.batch_size[0] // self.dp_size
            dummy_rewards = torch.zeros(batch_size, device='cpu')
            return make_reward_batch(self.REWARD_KEY, dummy_rewards, batch_size)
        
        start_time = time.time()
        
        # Get video IDs from data
        video_ids = data.non_tensor_batch.get('video_ids', [])
        batch_size = len(video_ids)
        
        # Generate prompts
        prompts = self._generate_batch_prompts(video_ids)
        
        # Run inference
        all_rewards = []
        for prompt in prompts:
            output = self.reward_rollout.generate(prompt, sampling_params=self.sampling_params)
            score = self._parse_output(output[0].outputs[0].text)
            all_rewards.append(score)
        
        rewards = torch.tensor(all_rewards, dtype=torch.float32)
        
        elapsed = time.time() - start_time
        logger.info(f"[qwen] Batch compute time: {elapsed:.2f}s")
        
        return make_reward_batch(self.REWARD_KEY, rewards, batch_size)
    
    def _generate_batch_prompts(self, video_ids: List[str]) -> List[Dict]:
        """Generate prompts for a batch of videos."""
        prompt_text = QUALITY_EVALUATION_PROMPT
        formatted_prompt = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
            f"{prompt_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        messages = []
        for video_id in video_ids:
            message = [{
                "prompt": formatted_prompt,
                "multi_modal_data": {"video": [video_id]},
            }]
            messages.append(message)
        
        return messages
    
    def _parse_output(self, output_text: str) -> float:
        """
        Parse the model output to extract the overall score.
        
        Returns a score in [0, 100] range, or 50.0 as default.
        """
        # Score extraction patterns
        score_patterns = [
            r'合计[：:]\\s*(\\d+(?:\\.\\d+)?)\\s*分',
            r'综合得分[：:]\\s*(\\d+(?:\\.\\d+)?)\\s*分',
            r'总分[：:]\\s*(\\d+(?:\\.\\d+)?)\\s*分',
            r'dim5[：:]\\s*(\\d+(?:\\.\\d+)?)\\s*分.*?合计[：:]\\s*(\\d+(?:\\.\\d+)?)\\s*分',
            r'(\\d+(?:\\.\\d+)?)\\s*分',
            r'(\\d+(?:\\.\\d+)?)/100',
        ]
        
        score = 50.0  # Default
        
        for pattern in score_patterns:
            match = re.search(pattern, output_text)
            if match:
                # For patterns with multiple groups, take the last one
                if len(match.groups()) > 1:
                    found_score = float(match.group(2))
                else:
                    found_score = float(match.group(1))
                
                # Clamp to valid range
                score = min(max(found_score, 0), 100)
                logger.debug(f"Extracted score: {score} from pattern: {pattern}")
                break
        
        if score == 50.0:
            logger.warning(f"No score found, using default. Output: {output_text[:200]}...")
        
        return score
    
    def parse_dimension_scores(self, output_text: str) -> Dict[str, float]:
        """
        Parse individual dimension scores from model output.
        
        Returns dict with keys like 'visual_artifacts', 'local_deformation', etc.
        """
        dim_patterns = [
            (r'dim1[：:]\\s*(\\d+(?:\\.\\d+)?)\\s*分', 'visual_artifacts'),
            (r'dim2[：:]\\s*(\\d+(?:\\.\\d+)?)\\s*分', 'local_deformation'),
            (r'dim3[：:]\\s*(\\d+(?:\\.\\d+)?)\\s*分', 'noise_quality'),
            (r'dim4[：:]\\s*(\\d+(?:\\.\\d+)?)\\s*分', 'clarity_sharpness'),
            (r'dim5[：:]\\s*(\\d+(?:\\.\\d+)?)\\s*分', 'color_accuracy'),
        ]
        
        scores = {}
        for pattern, dim_name in dim_patterns:
            match = re.search(pattern, output_text)
            if match:
                scores[dim_name] = float(match.group(1))
        
        return scores

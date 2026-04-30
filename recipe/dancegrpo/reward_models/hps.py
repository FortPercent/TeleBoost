# Copyright 2024 Dance-GRPO Team
"""
HPS (Human Preference Score) Reward Model - Evaluates image quality based on human preferences.

This model uses HPS v2 which combines CLIP features with a trained preference predictor
to score images based on learned human aesthetic preferences.
"""

import logging
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .base import BaseRewardModel, RewardConfig
from .registry import RewardRegistry

logger = logging.getLogger(__name__)


@RewardRegistry.register("hps")
class HPSRewardModel(BaseRewardModel):
    """
    Human Preference Score (HPS) v2 reward model.
    
    Uses a CLIP-based model fine-tuned on human preference data to score
    how well an image matches text and human aesthetic preferences.
    
    Config extra_config keys:
        - model_path: Path to HPS model checkpoint
        - model_type: Model architecture type (default: "ViT-H-14")
    """
    
    REWARD_KEY = "hps_rewards"
    
    def __init__(self, config: RewardConfig, global_rank: int, world_size: int):
        super().__init__(config, global_rank, world_size)
        self.hps_model: Optional[torch.nn.Module] = None
        self.preprocess = None
        self.tokenizer = None
        self.model_type: str = "ViT-H-14"
    
    def init_model(self) -> None:
        """Initialize the HPS model."""
        if not self.is_active:
            logger.info(f"[hps] Rank {self.global_rank} inactive, skipping init")
            return
        
        extra = self.config.extra_config or {}
        model_path = extra.get("model_path") or self.config.model_path
        self.model_type = extra.get("model_type", "ViT-H-14")
        
        if not model_path:
            raise ValueError("HPS model requires 'model_path' in config")
        
        try:
            from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
            
            # Create model
            model, _, preprocess_val = create_model_and_transforms(
                self.model_type,
                model_path,
                precision='amp',
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
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location="cpu")
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            
            self.hps_model = model
            self.preprocess = preprocess_val
            self.tokenizer = get_tokenizer(self.model_type)
            
            num_params = sum(p.numel() for p in model.parameters())
            logger.info(f"HPS model initialized: {num_params:,} params")
            
        except Exception as e:
            logger.error(f"Failed to load HPS model: {e}")
            raise
    
    def compute_single_score(
        self,
        video_frames: torch.Tensor,
        caption: str
    ) -> float:
        """
        Compute HPS score for a video frame.

        Uses the first frame of the video for scoring.

        Args:
            video_frames: Video tensor, shape (T, C, H, W). This is the layout
                produced by ``split_video_frames(permute_to_tchw=True)`` and matches
                the other reward models (see ``aesthetic.py``).
            caption: Text caption to evaluate alignment with

        Returns:
            HPS score (higher = better alignment with human preferences)
        """
        # Take first frame: (T, C, H, W) -> (C, H, W).
        frame = video_frames[0]

        # Convert to PIL Image
        frame_np = frame.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        frame_np = (frame_np * 255).astype(np.uint8)
        frame_pil = Image.fromarray(frame_np)
        
        # Preprocess
        image = self.preprocess(frame_pil).unsqueeze(0)
        text = self.tokenizer([caption])
        
        # Move to device
        image = image.to(self.get_device())
        text = text.to(self.get_device())
        self.hps_model.to(self.get_device())
        
        try:
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    outputs = self.hps_model(image, text)
                    image_features = outputs["image_features"]
                    text_features = outputs["text_features"]
                    
                    # Compute similarity
                    logits = image_features @ text_features.T
                    score = torch.diagonal(logits).item()
            
            logger.debug(f"HPS score: {score:.4f}")
            return score
            
        finally:
            self.hps_model.to("cpu")

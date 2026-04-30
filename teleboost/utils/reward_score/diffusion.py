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
    """Reward videos that contain more red.

    A simple proxy reward used for sanity-checking the GRPO loop end-to-end:
    pushes generations toward red-heavy frames. Effect is very visible when it
    works.

    Args:
        videos: torch.Tensor, shape (B, C, T, H, W) or (C, T, H, W), values in [0, 1].

    Returns:
        rewards: torch.Tensor, shape (B,).
    """
    # Accept either batched or single-video input.
    if videos.dim() == 4:  # (C, T, H, W)
        videos = videos.unsqueeze(0)  # -> (1, C, T, H, W)

    # Require 5-D (B, C, T, H, W).
    if videos.dim() != 5:
        raise ValueError(f"Expected 5D input (B, C, T, H, W), got {videos.dim()}D")

    # Clamp to [0, 1].
    videos = torch.clamp(videos, 0, 1)
    videos = videos.float()

    batch_size = videos.shape[0]
    rewards = []

    for b in range(batch_size):
        video = videos[b]  # (C, T, H, W)
        C, T, H, W = video.shape

        if C < 3:  # not RGB -> give the lowest reward
            rewards.append(0.0)
            continue

        frame_red_scores = []

        for t in range(T):
            frame = video[:, t, :, :]  # (C, H, W)

            # Extract RGB channels.
            red_channel = frame[0]    # (H, W)
            green_channel = frame[1]  # (H, W)
            blue_channel = frame[2]   # (H, W)

            # 1. Mean red-channel intensity.
            red_intensity = red_channel.mean().item()

            # 2. Red dominance: R - max(G, B), keep only positive.
            red_dominance = red_channel - torch.maximum(green_channel, blue_channel)
            red_dominance = torch.clamp(red_dominance, 0, 1)
            red_purity = red_dominance.mean().item()

            # 3. Bright-red pixel ratio: R > 0.7 AND R > G+0.3 AND R > B+0.3.
            bright_red_mask = (
                (red_channel > 0.7) &
                (red_channel > green_channel + 0.3) &
                (red_channel > blue_channel + 0.3)
            )
            bright_red_ratio = bright_red_mask.float().mean().item()

            # 4. Deep-red pixel ratio: R > 0.4 AND R > 1.5*G AND R > 1.5*B.
            deep_red_mask = (
                (red_channel > 0.4) &
                (red_channel > green_channel * 1.5) &
                (red_channel > blue_channel * 1.5)
            )
            deep_red_ratio = deep_red_mask.float().mean().item()

            # 5. Connectivity bonus: prefer large contiguous red regions over scattered pixels.
            red_regions_score = 0.0
            if bright_red_mask.sum() > 0:
                # Quick proxy via a small-kernel dilation.
                kernel_size = 3
                padding = kernel_size // 2

                bright_red_float = bright_red_mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                kernel = torch.ones(1, 1, kernel_size, kernel_size, device=videos.device) / (kernel_size * kernel_size)
                dilated = F.conv2d(bright_red_float, kernel, padding=padding)

                # Connectivity: ratio of high-value cells after dilation, weighted by the bright-red ratio.
                connectivity = (dilated > 0.3).float().mean().item()
                red_regions_score = connectivity * bright_red_ratio

            # 6. Per-frame red score (weighted sum).
            frame_score = (
                red_intensity * 30 +
                red_purity * 40 +
                bright_red_ratio * 50 +
                deep_red_ratio * 30 +
                red_regions_score * 20
            )

            frame_red_scores.append(frame_score)

        # 7. Aggregate per-video red reward.
        if frame_red_scores:
            avg_red_score = np.mean(frame_red_scores)
            max_red_score = np.max(frame_red_scores)

            # Consistency: lower variance -> higher score.
            red_consistency = 1.0 / (1.0 + np.std(frame_red_scores))

            video_red_reward = (
                avg_red_score * 0.6 +
                max_red_score * 0.3 +
                red_consistency * 10
            )

            # Bonuses for strong red signal.
            if avg_red_score > 10:
                video_red_reward += 15

            if max_red_score > 20:
                video_red_reward += 10

            # Cap.
            video_red_reward = min(video_red_reward, 100.0)
        else:
            video_red_reward = 0.0

        rewards.append(video_red_reward)

    return torch.tensor(rewards, dtype=torch.float32, device=videos.device)

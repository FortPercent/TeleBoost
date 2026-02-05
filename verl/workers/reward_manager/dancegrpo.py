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

from collections import defaultdict

import torch
from tensordict import TensorDict

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.diffusion import compute_red_intensity_reward
from verl.workers.reward_manager import register


@register("dancegrpo") # 把dancegrpo注册到REWARD_MANAGER_REGISTRY中，value是AIGCRewardManager这个类
class AIGCRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score # 如果没有传入compute_score函数，就使用默认的default_compute_score函数
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data sources

    def __call__(self, data: DataProto, return_dict=False): # __call__ 让一个类的实例可以像函数一样被直接调用
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        # reward_tensor = torch.zeros_like(data.batch["video_frames"], dtype=torch.float32)
        # reward_extra_info = defaultdict(list)

        # already_print_data_sources = {}
        
        all_rewards = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            reward = compute_red_intensity_reward(data_item.batch["video_frames"]) # 真正的计算reward的函数
            all_rewards.append(reward)
            # print(all_rewards)

            # score = self.compute_score(
            #     data_source=data_source,
            #     solution_str=response_str,
            #     ground_truth=ground_truth,
            #     extra_info=extra_info,
            # )

            # if isinstance(score, dict):
            #     reward = score["score"]
            #     # Store the information including original reward
            #     for key, value in score.items():
            #         reward_extra_info[key].append(value)
            # else:
            #     reward = score
            
        all_rewards = torch.cat(all_rewards, dim=0) 
        # dim=0 有两种情况，
        # 1是每个reward是一个标量，shape是(1,)，那么cat之后shape是(batch_size,)，
        # 2是每个reward是一个向量，shape是(num_rewards,)，那么cat之后shape是(batch_size, num_rewards)
        all_rewards=all_rewards.to(torch.device('cpu'))
        batch = TensorDict(
            {
                "rewards": all_rewards, # rewards是一个tensor，shape是(batch_size,)，每个元素是一个reward score
            },
            batch_size=len(data)
        )

        non_tensor_batch = data.non_tensor_batch
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
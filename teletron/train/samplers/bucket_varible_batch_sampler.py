import math
import numpy as np
import torch
from teleai_data_tool.schema.clip import Clip, ImageWithCaption
from .bucket import Bucket, get_close_ratio, get_random_resolution_type
from typing import Dict
from collections import defaultdict
from teleai_data_tool.logger import logger
import random

class BucketVariableBatchSampler(torch.utils.data.DistributedSampler):
    def __init__(
        self, 
        dataset, 
        bucket_config: Dict[str, Dict[str, Dict[str, int]]], 
        max_frames: int = 81,
        num_replicas: int | None = None, 
        rank: int | None = None,
        shuffle: bool = True, 
        drop_last: bool = False,     
        seed: int = 6666,
    ):
        self.data_list = dataset.data_list
        self.bucket = Bucket(bucket_config, padding_size=16)
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.last_micro_batch_access_index = 0

        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size() if torch.distributed.is_available() else 1
        if rank is None:
            rank = torch.distributed.get_rank() if torch.distributed.is_available() else 0
        self.num_replicas = num_replicas
        self.rank = rank 

        self._cached_bucket_sample_dict = None
        self._cached_bucket_bs = None
        self._cached_bucket_num = None

        ###
        self.max_frames = max_frames

        self._create_buckets()
        self._print_bucket_info()

        self.epoch = 0
        self.num_iters = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def __len__(self,):
        num = 0
        if self.drop_last:
            for k, v in self._cached_bucket_num.items():
                bs = self._cached_bucket_bs[k]
                num += int(math.ceil(v / bs))
            return int(math.ceil(num / self.num_replicas))
        else:
            for k, v in self._cached_bucket_num.items():
                bs = self._cached_bucket_bs[k]
                num += int(math.ceil(v / bs))
            return int(math.ceil(num / self.num_replicas))

    def __iter__(self):
        # set seed 
        generator = torch.Generator()
        generator.manual_seed(self.seed+self.epoch)
        bucket_sample_dict = self._cached_bucket_sample_dict

        bucket_micro_batch_count = defaultdict(int)
        bucket_last_consumed = defaultdict()

        # drop last and shuffle in every bucket 
        for bucket_info, id_list_per_bucket in bucket_sample_dict.items():
            bucket_bs = self._cached_bucket_bs[bucket_info]
            remainder = len(id_list_per_bucket) % bucket_bs
            if remainder > 0:
                if not self.drop_last:
                    num_padding = bucket_bs-remainder
                    id_list_per_bucket += random.choices(id_list_per_bucket, k=num_padding)
                else:
                    id_list_per_bucket = id_list_per_bucket[:-remainder]
            bucket_sample_dict[bucket_info] = id_list_per_bucket

            if self.shuffle:
                data_indices = torch.randperm(len(id_list_per_bucket), generator=generator).tolist()
                id_list_per_bucket = [id_list_per_bucket[i] for i in data_indices]
                bucket_sample_dict[bucket_info] = id_list_per_bucket
            
            num_micro_batches = len(id_list_per_bucket) // bucket_bs
            bucket_micro_batch_count[bucket_info] = num_micro_batches

        # bucket_info_access_order: num1x['t1xh1xw1'] + num2x['t2xh2xw2'] + num3x['t3xh3xw3'] + ...
        bucket_info_access_order = []
        for bucket_info, num_micro_batch in bucket_micro_batch_count.items():
            bucket_info_access_order.extend([bucket_info] * num_micro_batch)
        
        # drop last and shuffle all bucket 
        if self.shuffle:
            bucket_info_access_order_indices = torch.randperm(len(bucket_info_access_order), generator=generator).tolist()
            bucket_info_access_order = [bucket_info_access_order[i] for i in bucket_info_access_order_indices]

        remainder = len(bucket_info_access_order) % self.num_replicas
        if remainder > 0:
            if self.drop_last:
                bucket_info_access_order = bucket_info_access_order[:-remainder]
            else:
                bucket_info_access_order += bucket_info_access_order[:self.num_replicas-remainder]

                ## add data in bucket sample dict 
                for bucket_info in bucket_info_access_order[:self.num_replicas-remainder]:
                    bucket_sample_dict[bucket_info] += bucket_sample_dict[bucket_info][:self._cached_bucket_bs[bucket_info]]

        # total step per epcoh  
        num_iters = len(bucket_info_access_order) // self.num_replicas
        self.num_iters = num_iters

        start_iter_idx = self.last_micro_batch_access_index // self.num_replicas

        self.last_micro_batch_access_index = start_iter_idx * self.num_replicas

        # delete previous data 
        for i in range(self.last_micro_batch_access_index):
            bucket_info = bucket_info_access_order[i]
            bucket_bs = self._cached_bucket_bs[bucket_info]
            if bucket_info in bucket_last_consumed:
                bucket_last_consumed[bucket_info] += bucket_bs
            else:
                bucket_last_consumed[bucket_info] = bucket_bs

        for i in range(start_iter_idx, num_iters):
            # get global batch 
            bucket_access_list = bucket_info_access_order[i * self.num_replicas : (i + 1) * self.num_replicas]
            self.last_micro_batch_access_index += self.num_replicas

            bucket_access_boundaries = []
            for bucket_info in bucket_access_list:
                bucket_bs = self._cached_bucket_bs[bucket_info]
                last_consumed_index = bucket_last_consumed.get(bucket_info, 0)
                bucket_access_boundaries.append([last_consumed_index, last_consumed_index + bucket_bs])

                # update consumption
                if bucket_info in bucket_last_consumed:
                    bucket_last_consumed[bucket_info] += bucket_bs
                else:
                    bucket_last_consumed[bucket_info] = bucket_bs
            
            bucket_info = bucket_access_list[self.rank]
            boundary = bucket_access_boundaries[self.rank]
            cur_micro_batch = bucket_sample_dict[bucket_info][boundary[0] : boundary[1]]
            yield cur_micro_batch
        self.reset()

    def _create_buckets(self,):
        if self._cached_bucket_sample_dict is not None:
            return 
        
        bucket_sample_dict = defaultdict(list)
        bucket_bs = defaultdict(int)
        for idx, clip in enumerate(self.data_list):
            if isinstance(clip, ImageWithCaption):
                length = 1
                # get resolution type 
                resolution_type = get_random_resolution_type(self.bucket.bucket_prob, torch.rand(1).item(), length)
                # get aspect ratio 
                aspect_ratio = get_close_ratio(clip.image.height, clip.image.width)
                # get height width
                height, width = self.bucket.resolution[resolution_type][aspect_ratio]
                bucket_info = f"{length}x{height}x{width}"
                bucket_sample_dict[bucket_info].append(idx)
                bucket_bs[bucket_info] = self.bucket.bucket_bsz[bucket_info]

                setattr(clip, "video_info", (length, width, height))
            
            elif isinstance(clip, Clip):
                frame_interval = clip.frame_interval
                length = int(clip.length // frame_interval)
                length = (clip.length - 1) // 4 * 4 + 1
                length = int(min(length, self.max_frames))
                # get resolution type 
                resolution_type = get_random_resolution_type(self.bucket.bucket_prob, torch.rand(1).item(), length)
                # get aspect ratio 
                aspect_ratio = get_close_ratio(clip.height, clip.width)
                # get height width
                height, width = self.bucket.resolution[resolution_type][aspect_ratio]
                bucket_info = f"{length}x{height}x{width}"
                bucket_sample_dict[bucket_info].append(idx)
                bucket_bs[bucket_info] = self.bucket.bucket_bsz[bucket_info]

                setattr(clip, "video_info", (length, width, height))
            
            else:
                raise ValueError("Input must be Clip or ImageWithCaption.")

        self._cached_bucket_sample_dict = bucket_sample_dict
        self._cached_bucket_num = {k:len(v) for k, v in bucket_sample_dict.items()}
        self._cached_bucket_bs = bucket_bs

    def reset(self):
        self.last_micro_batch_access_index = 0
    
    def state_dict(self,):
        return {"seed": self.seed, "epoch": self.epoch, "last_micro_batch_access_index": self.last_micro_batch_access_index}

    def load_state_dict(self, state_dict: dict):
        self.__dict__.update(state_dict)

    def _print_bucket_info(self,):
        logger.info(
            f"bucket_type \t bucket number \t bucket_batch"
        )
        for bucket_type, bucket_number in self._cached_bucket_num.items():
            logger.info(
                f'{bucket_type} \t {bucket_number} \t {self._cached_bucket_bs[bucket_type]}\n'
            )